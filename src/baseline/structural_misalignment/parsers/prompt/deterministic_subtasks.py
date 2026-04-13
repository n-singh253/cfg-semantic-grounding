"""Deterministic prompt parser that emits structured benign/malicious subtasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.baseline.structural_misalignment.grounding.schemas import (
    normalize_subtask_text,
    normalize_subtasks,
    split_problem_statement,
)
from src.baseline.structural_misalignment.parsers.registry import register_prompt_parser


def _expand_chunk(chunk: str) -> List[str]:
    lowered = chunk.lower()
    separators = [" while ", " but ", " and ", " then ", " also "]
    for separator in separators:
        if separator in lowered:
            parts = [normalize_subtask_text(part) for part in chunk.split(separator) if normalize_subtask_text(part)]
            if len(parts) > 1:
                return parts
    return [normalize_subtask_text(chunk)] if normalize_subtask_text(chunk) else []


def deterministic_subtasks_parser(
    *,
    problem_statement: str,
    artifact_dir: Path,
    **kwargs: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    del kwargs
    chunks = split_problem_statement(problem_statement)
    expanded: List[str] = []
    for chunk in chunks:
        expanded.extend(_expand_chunk(chunk))

    subtasks = normalize_subtasks(expanded)
    metadata: Dict[str, Any] = {
        "parser": "deterministic_subtasks",
        "subtask_count": len(subtasks),
        "artifact_dir": str(artifact_dir),
        "deterministic": True,
    }
    return subtasks, metadata


register_prompt_parser("deterministic_subtasks")(deterministic_subtasks_parser)
