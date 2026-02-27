"""Helpers for command-driven agent wrappers."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict, List

from src.common.artifact_store import atomic_write_json, atomic_write_text
from src.common.subprocess import command_exists, run_command
from src.common.types import Patch


def _agent_artifact_dir(agent_name: str, repo_code: Dict[str, Any]) -> Path:
    run_root = repo_code.get("run_root")
    instance_id = str(repo_code.get("instance_id", "unknown"))
    if run_root:
        return Path(str(run_root)) / "artifacts" / "agents" / instance_id / agent_name
    return Path(str(repo_code.get("path", "."))) / ".agent_artifacts" / agent_name


def _write_invocation_logs(
    artifact_dir: Path,
    command: List[str],
    prompt: str,
    stdout: str,
    stderr: str,
) -> Dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command_path = artifact_dir / "command.txt"
    prompt_path = artifact_dir / "prompt.txt"
    stdout_path = artifact_dir / "stdout.log"
    stderr_path = artifact_dir / "stderr.log"
    atomic_write_text(command_path, " ".join(shlex.quote(x) for x in command))
    atomic_write_text(prompt_path, prompt)
    atomic_write_text(stdout_path, stdout or "")
    atomic_write_text(stderr_path, stderr or "")
    return {
        "artifact_path": str(artifact_dir),
        "command_path": str(command_path),
        "prompt_path": str(prompt_path),
        "stdout_log_path": str(stdout_path),
        "stderr_log_path": str(stderr_path),
    }


def run_cli_agent(
    *,
    agent_name: str,
    config: Dict[str, Any],
    repo_code: Dict[str, Any],
    prompt: str,
    all_tests: List[Any],
) -> Patch:
    command_template = config.get("command")
    if not isinstance(command_template, list) or not command_template:
        raise ValueError(f"{agent_name}: config.command must be a non-empty list")

    executable = str(command_template[0])
    behavior = str(config.get("missing_tool_behavior", "fail")).lower()
    artifact_dir = _agent_artifact_dir(agent_name, repo_code)
    if not command_exists(executable):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        missing_tool_path = artifact_dir / "missing_tool.json"
        atomic_write_json(
            missing_tool_path,
            {
                "agent": agent_name,
                "missing_tool": executable,
                "behavior": behavior,
            },
        )
        if behavior == "skip":
            return Patch(
                unified_diff="",
                metadata={
                    "agent": agent_name,
                    "tool_available": False,
                    "missing_tool_behavior": behavior,
                    "artifact_path": str(artifact_dir),
                    "missing_tool_path": str(missing_tool_path),
                },
            )
        raise RuntimeError(
            f"{agent_name}: missing required CLI tool '{executable}'. "
            f"Details: {missing_tool_path}"
        )

    tests_serialized = json.dumps([getattr(t, "name", "test") for t in all_tests], ensure_ascii=True)
    fmt = {
        "repo_path": str(repo_code.get("path", ".")),
        "prompt": prompt,
        "tests": tests_serialized,
    }
    command = [str(part).format(**fmt) for part in command_template]
    cwd = Path(str(repo_code.get("path", ".")))
    timeout_sec = int(config.get("timeout_sec", 120))
    result = run_command(command, cwd=cwd, timeout_sec=timeout_sec)
    log_paths = _write_invocation_logs(artifact_dir, command, prompt, result.stdout, result.stderr)

    output_mode = str(config.get("output_mode", "stdout"))
    if output_mode == "file":
        output_file = str(config.get("output_file", "patch.diff"))
        patch_path = cwd / output_file
        diff_text = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
    else:
        diff_text = result.stdout or ""

    if not diff_text.strip():
        if behavior == "skip":
            return Patch(
                unified_diff="",
                metadata={
                    "agent": agent_name,
                    "tool_available": True,
                    "returncode": result.returncode,
                    "empty_output": True,
                    "command": " ".join(shlex.quote(x) for x in command),
                    **log_paths,
                },
            )
        raise RuntimeError(
            f"{agent_name}: command produced no patch output (returncode={result.returncode}). "
            f"See logs: {log_paths['stdout_log_path']} and {log_paths['stderr_log_path']}"
        )

    return Patch(
        unified_diff=diff_text,
        metadata={
            "agent": agent_name,
            "tool_available": True,
            "returncode": result.returncode,
            "command": " ".join(shlex.quote(x) for x in command),
            "stderr_preview": (result.stderr or "")[:500],
            **log_paths,
        },
    )
