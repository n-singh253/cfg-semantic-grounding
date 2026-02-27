"""Prompt filtering baseline."""

from __future__ import annotations

from typing import Any, Dict, List

from src.baseline.base import BaseDefense
from src.baseline.registry import register_baseline


class PromptFilterDefense(BaseDefense):
    name = "prompt_filter"

    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ):
        _ = (code_or_patch, all_tests, repo_code)
        banned = [str(p).lower() for p in self.config.get("banned_patterns", [])]
        matches = [p for p in banned if p and p in prompt.lower()]
        default_decision = str(self.config.get("default_decision", "accept")).lower()
        if matches:
            self.last_signals = {"matched_patterns": matches, "policy": "reject_on_match"}
            return False
        self.last_signals = {"matched_patterns": [], "policy_default": default_decision}
        return False if default_decision == "reject" else True


register_baseline("prompt_filter")(PromptFilterDefense)
