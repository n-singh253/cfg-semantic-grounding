"""LLM-based subtask generation (default prompt parser)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.baseline.structural_misalignment.grounding.subtasks import (
    DEFAULT_SYSTEM_PROMPT,
    generate_subtasks as _generate_subtasks_impl,
)
from src.baseline.structural_misalignment.parsers.registry import register_prompt_parser
from src.common.llm import LLMClient


def llm_subtasks_parser(
    *,
    llm_client: LLMClient,
    instance_id: str,
    module_name: str,
    module_config_hash: str,
    fidelity_mode: str,
    provider: str,
    model: str,
    problem_statement: str,
    artifact_dir: Path,
    temperature: float,
    seed: Any,
    max_retries: int,
    backoff_sec: float,
    allow_provider_fallback: bool,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    **kwargs: Any,
) -> Tuple[List[str], Dict[str, Any]]:
    """Parse problem statement into subtasks using LLM.
    
    This is the default prompt parser that uses LLM to generate a JSON array of subtask strings.
    
    Args:
        llm_client: Shared LLM client
        instance_id: Unique instance identifier
        module_name: Name of the calling module
        module_config_hash: Config hash for caching
        fidelity_mode: "full" or "surrogate"
        provider: LLM provider (openai, anthropic, etc.)
        model: Model name
        problem_statement: The issue/problem description to parse
        artifact_dir: Directory to save artifacts
        temperature: LLM temperature
        seed: Random seed
        max_retries: Max retry attempts
        backoff_sec: Backoff duration between retries
        allow_provider_fallback: Whether to fallback to other providers
        system_prompt: System prompt for LLM (optional)
        **kwargs: Additional arguments (ignored for extensibility)
    
    Returns:
        Tuple of (subtasks, metadata)
        - subtasks: List of subtask strings
        - metadata: Dict with prompt_hash, cache_hit, token_usage, etc.
    """
    return _generate_subtasks_impl(
        llm_client=llm_client,
        instance_id=instance_id,
        module_name=module_name,
        module_config_hash=module_config_hash,
        fidelity_mode=fidelity_mode,
        provider=provider,
        model=model,
        problem_statement=problem_statement,
        artifact_dir=artifact_dir,
        temperature=temperature,
        seed=seed,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
        allow_provider_fallback=allow_provider_fallback,
        system_prompt=system_prompt,
    )


# Register as default prompt parser
register_prompt_parser("llm_subtasks")(llm_subtasks_parser)
