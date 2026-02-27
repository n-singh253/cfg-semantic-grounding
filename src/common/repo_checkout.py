"""Repository checkout helpers for run-local snapshot preparation."""

from __future__ import annotations

from pathlib import Path


def ensure_repo_checkout(_repo_id: str, _base_commit: str, target_dir: Path) -> Path:
    """Ensure target checkout directory exists for this harness run."""
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir
