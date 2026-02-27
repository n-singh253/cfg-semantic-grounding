"""Bandit-based defense baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.baseline.base import BaseDefense
from src.baseline.registry import register_baseline
from src.common.subprocess import command_exists, run_command


class StaticBanditDefense(BaseDefense):
    name = "static_bandit"

    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ):
        _ = (prompt, code_or_patch, all_tests)
        repo_path = Path(str(repo_code.get("path", ".")))
        command = self.config.get("command", ["bandit", "-r", ".", "-f", "json"])
        if not command_exists(str(command[0])):
            behavior = str(self.config.get("missing_tool_behavior", "skip"))
            self.last_signals = {"tool": "bandit", "available": False, "behavior": behavior}
            return True if behavior == "skip" else False
        result = run_command([str(x) for x in command], cwd=repo_path)
        findings = 0
        try:
            payload = json.loads(result.stdout or "{}")
            findings = len(payload.get("results", []))
        except json.JSONDecodeError:
            findings = 0
        max_findings = int(self.config.get("max_findings", 0))
        self.last_signals = {
            "tool": "bandit",
            "available": True,
            "returncode": result.returncode,
            "findings": findings,
        }
        if findings > max_findings:
            return False
        return True


register_baseline("static_bandit")(StaticBanditDefense)
