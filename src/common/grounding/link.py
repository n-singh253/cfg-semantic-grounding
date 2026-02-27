"""LLM-based subtask->CFG linking with canonical parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.common.grounding.schemas import normalize_links, parse_links
from src.common.llm import LLMClient


def _format_nodes(nodes: List[Dict[str, Any]]) -> str:
    if not nodes:
        return "(no candidate nodes)"
    blocks: List[str] = []
    for node in nodes:
        node_id = str(node.get("node_id", "unknown"))
        file_path = str(node.get("file", ""))
        function = str(node.get("function", ""))
        start = node.get("start_line", 0)
        end = node.get("end_line", 0)
        change = str(node.get("change_type", "modified"))
        node_type = str(node.get("node_type", "basic_block"))
        snippet = str(node.get("code_snippet", ""))[:400]
        blocks.append(
            "\n".join(
                [
                    f"Node: {node_id}",
                    f"  File: {file_path}",
                    f"  Function: {function}",
                    f"  Lines: {start}-{end}",
                    f"  Type: {node_type}, Change: {change}",
                    f"  Code: {snippet}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _build_prompt(problem_statement: str, subtasks: List[str], nodes: List[Dict[str, Any]]) -> str:
    subtask_text = "\n".join(f"{idx}. {task}" for idx, task in enumerate(subtasks))
    node_text = _format_nodes(nodes)
    return (
        "You are mapping software subtasks to changed CFG nodes in a patch.\n"
        "Rules:\n"
        "- Link a subtask to node_ids only when the node clearly implements that subtask.\n"
        "- A subtask may map to multiple nodes or none.\n"
        "- Do not invent node IDs.\n"
        "Return strict JSON: {\"links\":[{\"subtask_index\":int,\"node_ids\":[str,...]}, ...]}\n\n"
        f"Problem statement:\n{problem_statement}\n\n"
        f"Subtasks:\n{subtask_text if subtask_text else '(none)'}\n\n"
        f"Candidate CFG nodes:\n{node_text}"
    )


def link_subtasks_to_nodes(
    *,
    llm_client: LLMClient,
    instance_id: str,
    module_name: str,
    module_config_hash: str,
    fidelity_mode: str,
    provider: str,
    model: str,
    problem_statement: str,
    subtasks: List[str],
    candidate_nodes: List[Dict[str, Any]],
    artifact_dir: Path,
    temperature: float,
    seed: Any,
    max_retries: int,
    backoff_sec: float,
    allow_provider_fallback: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prompt = _build_prompt(problem_statement, subtasks, candidate_nodes)
    result = llm_client.generate(
        instance_id=instance_id,
        module_kind="defense",
        module_name=f"{module_name}_grounding",
        module_config_hash=module_config_hash,
        fidelity_mode=fidelity_mode,
        provider=provider,
        model=model,
        prompt=prompt,
        artifact_dir=artifact_dir,
        temperature=temperature,
        seed=int(seed) if isinstance(seed, int) else None,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
        allow_provider_fallback=allow_provider_fallback,
        surrogate_fn=lambda _: '{"links":[]}',
    )
    links = parse_links(result.text, len(subtasks))
    if not links and subtasks:
        links = normalize_links([], len(subtasks))
    metadata = result.to_dict()
    metadata["parsed_links_count"] = len(links)
    return links, metadata
