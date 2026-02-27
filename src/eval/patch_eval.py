"""Patch application and evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.common.artifact_store import atomic_write_text
from src.common.diff import apply_unified_diff
from src.common.subprocess import command_exists, run_command
from src.common.tests import run_tests
from src.common.types import TestSpec


def apply_patch(repo_dir: Path, diff_text: str) -> Tuple[bool, str]:
    return apply_unified_diff(repo_dir, diff_text)


def run_repo_tests(test_specs: List[TestSpec], repo_dir: Path, log_path: Path) -> Tuple[bool, int]:
    return run_tests(test_specs, repo_dir, log_path)


def _tool_version(tool: str) -> str:
    if not command_exists(tool):
        return ""
    result = run_command([tool, "--version"])
    text = (result.stdout or result.stderr or "").strip()
    first = text.splitlines()[0].strip() if text else ""
    return first


def run_static_checks(repo_dir: Path, output_path: Path) -> Tuple[int, Path, Dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"tools": {}}
    total_findings = 0

    bandit_cmd = ["bandit", "-r", ".", "-f", "json"]
    if command_exists("bandit"):
        bandit_version = _tool_version("bandit")
        res = run_command(bandit_cmd, cwd=repo_dir)
        findings = 0
        parsed = None
        try:
            parsed = json.loads(res.stdout or "{}")
            findings = len(parsed.get("results", []))
            total_findings += findings
        except json.JSONDecodeError:
            pass
        payload["tools"]["bandit"] = {
            "available": True,
            "version": bandit_version,
            "command": bandit_cmd,
            "returncode": res.returncode,
            "findings": findings,
            "raw_stdout_preview": (res.stdout or "")[:500],
            "raw_stderr_preview": (res.stderr or "")[:500],
        }
    else:
        payload["tools"]["bandit"] = {"available": False}

    semgrep_cmd = ["semgrep", "--config", "auto", "--json", "."]
    if command_exists("semgrep"):
        semgrep_version = _tool_version("semgrep")
        res = run_command(semgrep_cmd, cwd=repo_dir)
        findings = 0
        parsed = None
        try:
            parsed = json.loads(res.stdout or "{}")
            findings = len(parsed.get("results", []))
            total_findings += findings
        except json.JSONDecodeError:
            pass
        payload["tools"]["semgrep"] = {
            "available": True,
            "version": semgrep_version,
            "command": semgrep_cmd,
            "returncode": res.returncode,
            "findings": findings,
            "raw_stdout_preview": (res.stdout or "")[:500],
            "raw_stderr_preview": (res.stderr or "")[:500],
        }
    else:
        payload["tools"]["semgrep"] = {"available": False}

    payload["total_findings"] = total_findings
    atomic_write_text(output_path, json.dumps(payload, indent=2, sort_keys=True))
    return total_findings, output_path, payload
