"""Semgrep baseline registration.

Semgrep execution lives in ``src.eval.defense``. It scans an isolated clean
repository, applies the patch, scans again, and compares finding counts.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.baseline.base import BaseDefense
from src.baseline.registry import register_baseline


class StaticSemgrepDefense(BaseDefense):
    name = "static_semgrep"

    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ):
        _ = (prompt, code_or_patch, all_tests, repo_code)
        self.last_signals = {
            "tool": "semgrep",
            "failure_reason": "use_cli_run_defense_for_isolated_static_scanning",
        }
        return False


register_baseline("static_semgrep")(StaticSemgrepDefense)
