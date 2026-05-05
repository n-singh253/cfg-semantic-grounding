"""Shared graph-building pipeline for structural misalignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.baseline.structural_misalignment.embeddings import encode_texts, serialize_code_node_for_embedding
from src.baseline.structural_misalignment.graph.build import build_canonical_graph, build_pyg_heterodata, write_graph_artifacts
from src.baseline.structural_misalignment.grounding.schemas import normalize_subtasks, serialize_subtask_for_embedding
from src.baseline.structural_misalignment.parsers.registry import get_linker, get_patch_parser, get_prompt_parser

import src.baseline.structural_misalignment.parsers.prompt.deterministic_subtasks  # noqa: F401
import src.baseline.structural_misalignment.parsers.prompt.llm_subtasks  # noqa: F401
import src.baseline.structural_misalignment.parsers.patch.cfg_ast  # noqa: F401
import src.baseline.structural_misalignment.parsers.patch.cfg_ast_scoped  # noqa: F401
import src.baseline.structural_misalignment.parsers.patch.llm_chunks  # noqa: F401
import src.baseline.structural_misalignment.parsers.linking.embedding_similarity  # noqa: F401
import src.baseline.structural_misalignment.parsers.linking.llm_grounding  # noqa: F401


def build_structural_graph(
    *,
    prompt: str,
    patch_text: str,
    repo_code: Dict[str, Any],
    config: Dict[str, Any],
    artifact_root: Path,
    llm_client: Any,
    module_name: str,
    module_config_hash: str,
    fidelity_mode: str,
    graph_label: int = 0,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    parser_config = config.get("parsers", {}) if isinstance(config.get("parsers"), dict) else {}
    prompt_parser_name = str(parser_config.get("prompt", "deterministic_subtasks")).strip()
    patch_parser_name = str(parser_config.get("patch", "cfg_ast")).strip()
    linker_name = str(parser_config.get("linking", "embedding_similarity")).strip()
    prompt_parser = get_prompt_parser(prompt_parser_name)
    patch_parser = get_patch_parser(patch_parser_name)
    linker = get_linker(linker_name)
    llm_config = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    provider = str(llm_config.get("provider", config.get("provider", "none")))
    model = str(llm_config.get("model", config.get("model", "none")))
    temperature = float(llm_config.get("temperature", config.get("temperature", 0.0)))
    seed = llm_config.get("seed", config.get("seed"))
    max_retries = int(llm_config.get("max_retries", config.get("max_retries", 0)))
    backoff_sec = float(llm_config.get("backoff_sec", config.get("backoff_sec", 0.0)))
    allow_provider_fallback = bool(
        llm_config.get(
            "allow_provider_fallback",
            config.get("allow_provider_fallback", False),
        )
    )

    instance_id = str(repo_code.get("instance_id", "unknown"))
    embedding_model_name = str(config.get("embedding_model_name", "microsoft/codebert-base"))
    embedding_pooling = str(config.get("embedding_pooling", "mean"))
    base_repo = Path(str(repo_code.get("path", ""))).resolve()

    subtasks_raw, subtasks_meta = prompt_parser(
        llm_client=llm_client,
        instance_id=instance_id,
        module_name=module_name,
        module_config_hash=module_config_hash,
        fidelity_mode=fidelity_mode,
        provider=provider,
        model=model,
        problem_statement=prompt,
        artifact_dir=artifact_root / "subtasks",
        temperature=temperature,
        seed=seed,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
        allow_provider_fallback=allow_provider_fallback,
        system_prompt=str(llm_config.get("subtasks_system_prompt", config.get("system_prompt", ""))),
        config=config,
    )
    subtasks = normalize_subtasks(list(subtasks_raw))

    cfg_diff, candidate_nodes, cfg_diagnostics = patch_parser(
        patch_text,
        base_repo=base_repo if base_repo.exists() else None,
        allow_hunk_fallback=bool(config.get("allow_hunk_fallback", True)),
        config=config,
    )
    links, link_meta = linker(
        llm_client=llm_client,
        instance_id=instance_id,
        module_name=module_name,
        module_config_hash=module_config_hash,
        fidelity_mode=fidelity_mode,
        provider=provider,
        model=model,
        problem_statement=prompt,
        subtasks=subtasks,
        candidate_nodes=candidate_nodes,
        artifact_dir=artifact_root / "grounding",
        temperature=temperature,
        seed=seed,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
        allow_provider_fallback=allow_provider_fallback,
        config=config,
    )

    subtask_embeddings = encode_texts(
        [serialize_subtask_for_embedding(subtask) for subtask in subtasks],
        model_name=embedding_model_name,
        pooling=embedding_pooling,
    )
    code_embeddings = encode_texts(
        [serialize_code_node_for_embedding(node) for node in candidate_nodes],
        model_name=embedding_model_name,
        pooling=embedding_pooling,
    )
    graph_payload = build_canonical_graph(
        instance_id=instance_id,
        graph_label=graph_label,
        subtasks=subtasks,
        candidate_nodes=candidate_nodes,
        code_edges=list(cfg_diff.get("candidate_edges", [])),
        links=links,
        subtask_features=subtask_embeddings.vectors,
        code_features=code_embeddings.vectors,
    )
    artifact_paths = write_graph_artifacts(artifact_root / "graph", graph_payload)
    hetero_graph = build_pyg_heterodata(graph_payload)

    metadata = {
        "prompt_parser": prompt_parser_name,
        "patch_parser": patch_parser_name,
        "linker": linker_name,
        "subtasks_metadata": subtasks_meta,
        "cfg_diagnostics": cfg_diagnostics,
        "link_metadata": link_meta,
        "artifact_paths": artifact_paths,
        "embedding_model_name": embedding_model_name,
        "embedding_pooling": embedding_pooling,
        "llm_provider": provider,
        "llm_model": model,
    }
    return graph_payload, hetero_graph, metadata
