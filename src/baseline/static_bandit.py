"""Bandit baseline registration.

Bandit execution for the publication pipeline is implemented in
``src.eval.defense`` so it can apply patches in isolated temporary
repo copies and avoid contaminating shared benchmark checkouts.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.baseline.base import BaseDefense
from src.baseline.registry import register_baseline


class StaticBanditDefense(BaseDefense):
    name = "static_bandit"

    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ):
        _ = (prompt, code_or_patch, all_tests, repo_code)
        self.last_signals = {
            "tool": "bandit",
            "failure_reason": "use_cli_run_defense_for_isolated_static_scanning",
        }
        return False


register_baseline("static_bandit")(StaticBanditDefense)
