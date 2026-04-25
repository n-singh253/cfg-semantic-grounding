"""Iterative LLM-based subtask-to-node grounding."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.baseline.structural_misalignment.grounding.link_iterative import (
    link_subtasks_to_nodes as _link_iterative_impl,
)
from src.baseline.structural_misalignment.parsers.registry import register_linker
from src.common.llm import LLMClient


def llm_grounding_iterative_linker(
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
    **kwargs: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    return _link_iterative_impl(
        llm_client=llm_client,
        instance_id=instance_id,
        module_name=module_name,
        module_config_hash=module_config_hash,
        fidelity_mode=fidelity_mode,
        provider=provider,
        model=model,
        problem_statement=problem_statement,
        subtasks=subtasks,
        candidate_nodes=candidate_nodes,
        artifact_dir=artifact_dir,
        temperature=temperature,
        seed=seed,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
        allow_provider_fallback=allow_provider_fallback,
        config=kwargs.get("config"),
    )


register_linker("llm_grounding_iterative")(llm_grounding_iterative_linker)
