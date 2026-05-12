"""LLM-based subtask generation (default prompt parser)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.baseline.structural_misalignment.grounding.schemas import (
    analyze_subtask_requirement_retention,
    ensure_suspicious_requirement_retention,
    split_problem_statement,
)
from src.baseline.structural_misalignment.grounding.subtasks import (
    DEFAULT_SYSTEM_PROMPT,
    generate_subtasks as _generate_subtasks_impl,
    generate_subtasks_iterative as _generate_subtasks_iterative_impl,
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
    try:
        config = kwargs.get("config") if isinstance(kwargs.get("config"), dict) else {}
        chunking_enabled = bool(config.get("subtask_chunking_enabled", True))
        chunk_max_chars = int(config.get("subtask_chunk_max_chars", 2000))
        generator = _generate_subtasks_iterative_impl if chunking_enabled else _generate_subtasks_impl
        generator_kwargs: Dict[str, Any] = {}
        if chunking_enabled:
            generator_kwargs["chunk_max_chars"] = chunk_max_chars
        result = generator(
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
            system_prompt=system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
            **generator_kwargs,
        )
        fallback_file = artifact_dir / "fallback_metadata.json"
        if fallback_file.exists():
            fallback_file.unlink()
        return result
    except Exception as exc:
        config = kwargs.get("config") if isinstance(kwargs.get("config"), dict) else {}
        fallback_enabled = bool(config.get("fallback_to_deterministic_subtasks_on_llm_failure", False))
        if not fallback_enabled:
            raise

        chunks = split_problem_statement(problem_statement)
        subtasks = chunks[:12] if chunks else ["Read changed code", "Map changes to requested behavior"]
        subtasks, repair_diagnostics = ensure_suspicious_requirement_retention(problem_statement, subtasks)
        retention_diagnostics = analyze_subtask_requirement_retention(problem_statement, subtasks)
        parse_diagnostics: Dict[str, Any] = {
            "parse_mode": "deterministic_fallback_after_llm_failure",
            "parsed_subtasks_count": len(subtasks),
            "response_chars": 0,
            "response_lines": 0,
        }
        metadata: Dict[str, Any] = {
            "parser": "llm_subtasks",
            "fallback_used": True,
            "fallback_parser": "deterministic_subtasks",
            "fallback_reason": f"{type(exc).__name__}: {exc}",
            "provider": provider,
            "model": model,
            "cache_hit": False,
            "call_count": max_retries + 1,
            "parse_diagnostics": parse_diagnostics,
            "requirement_retention_diagnostics": retention_diagnostics,
            "requirement_retention_repair": repair_diagnostics,
            "parsed_subtasks_count": len(subtasks),
        }
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "parser_diagnostics.json").write_text(
            json.dumps(
                {
                    "parse_diagnostics": parse_diagnostics,
                    "requirement_retention_diagnostics": retention_diagnostics,
                    "requirement_retention_repair": repair_diagnostics,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (artifact_dir / "fallback_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return subtasks, metadata


# Register as default prompt parser
register_prompt_parser("llm_subtasks")(llm_subtasks_parser)
