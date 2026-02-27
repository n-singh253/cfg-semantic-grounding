"""Prompt rewrite defense."""

from __future__ import annotations

from typing import Any, Dict, List

from src.baseline.base import BaseDefense
from src.baseline.registry import register_baseline


class PromptRewriteDefense(BaseDefense):
    name = "prompt_rewrite"

    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ):
        _ = (code_or_patch, all_tests, repo_code)
        replacements = self.config.get("replacements") or {
            "bypass validation": "preserve validation",
            "disable checks": "keep checks enabled",
            "ignore sanitization": "retain sanitization",
        }
        edited = prompt
        changed = False
        for src, dst in replacements.items():
            if src.lower() in edited.lower():
                edited = edited.replace(src, dst)
                changed = True
        self.last_signals = {
            "changed": changed,
            "replacement_count": len(replacements),
        }
        if changed:
            return edited
        return True


register_baseline("prompt_rewrite")(PromptRewriteDefense)
