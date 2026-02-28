"""Training-time helpers for structural misalignment classifiers."""

from __future__ import annotations

from typing import Any


def assert_eval_only_no_refit(eval_only: bool, obj: Any, stage: str) -> None:
    """Guardrail helper: prohibit fitting steps during eval-only runs."""
    if not eval_only:
        return
    if stage.strip().lower().startswith("fit"):
        raise RuntimeError("eval_only guard violated: fitting is not allowed during evaluation")
    _ = obj
