"""LLM-based subtask-to-node grounding (default linker)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.baseline.structural_misalignment.grounding.link import link_subtasks_to_nodes as _link_impl
from src.baseline.structural_misalignment.parsers.registry import register_linker
from src.common.llm import LLMClient


def llm_grounding_linker(
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
    """Link subtasks to CFG nodes using LLM.
    
    This is the default linker that uses LLM to map which nodes implement which subtasks.
    
    Args:
        llm_client: Shared LLM client
        instance_id: Unique instance identifier
        module_name: Name of the calling module
        module_config_hash: Config hash for caching
        fidelity_mode: "full" or "surrogate"
        provider: LLM provider
        model: Model name
        problem_statement: Original problem statement
        subtasks: List of subtask strings from prompt parser
        candidate_nodes: List of candidate nodes from patch parser
        artifact_dir: Directory to save artifacts
        temperature: LLM temperature
        seed: Random seed
        max_retries: Max retry attempts
        backoff_sec: Backoff duration
        allow_provider_fallback: Whether to fallback to other providers
        **kwargs: Additional arguments (ignored for extensibility)
    
    Returns:
        Tuple of (links, metadata)
        - links: List of dicts with subtask_index and node_ids
        - metadata: Dict with prompt_hash, cache_hit, token_usage, etc.
    """
    return _link_impl(
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
    )


# Register as default linker
register_linker("llm_grounding")(llm_grounding_linker)
