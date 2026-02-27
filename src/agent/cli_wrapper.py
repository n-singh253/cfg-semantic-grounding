"""Helpers for command-driven agent wrappers."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List

from src.common.artifact_store import atomic_write_json, atomic_write_text
from src.common.diff import looks_like_unified_diff
from src.common.hashing import sha256_text
from src.common.prompt_templates import AGENT_PATCH_PROMPT_TEMPLATE
from src.common.subprocess import command_exists, run_command
from src.common.types import Patch


def _agent_artifact_dir(agent_name: str, repo_code: Dict[str, Any]) -> Path:
    run_root = repo_code.get("run_root")
    instance_id = str(repo_code.get("instance_id", "unknown"))
    if run_root:
        return Path(str(run_root)) / "artifacts" / "agents" / instance_id / agent_name
    return Path(str(repo_code.get("path", "."))) / ".agent_artifacts" / agent_name


def _serialize_tests(all_tests: List[Any]) -> str:
    if not all_tests:
        return "- (no tests provided)"
    lines: List[str] = []
    for idx, spec in enumerate(all_tests, start=1):
        name = str(getattr(spec, "name", f"test_{idx}"))
        command = " ".join(getattr(spec, "command", []) or [])
        cwd = str(getattr(spec, "cwd", "") or ".")
        lines.append(f"- {name}: command=`{command}` cwd=`{cwd}`")
    return "\n".join(lines)


def _render_agent_prompt(
    *,
    config: Dict[str, Any],
    repo_code: Dict[str, Any],
    prompt: str,
    tests_text: str,
    tests_json: str,
) -> str:
    template = str(config.get("prompt_template", AGENT_PATCH_PROMPT_TEMPLATE))
    return template.format(
        prompt=prompt,
        tests=tests_text,
        tests_json=tests_json,
        repo_path=str(repo_code.get("path", ".")),
        repo_id=str(repo_code.get("repo_id", "unknown_repo")),
        instance_id=str(repo_code.get("instance_id", "unknown_instance")),
        base_commit=str(repo_code.get("base_commit", "unknown")),
    )


def _strip_markdown_fences(text: str) -> str:
    raw = text or ""
    # Remove fenced code wrappers while preserving internal content.
    raw = re.sub(r"^```(?:diff|patch)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.IGNORECASE)
    return raw.strip()


def _extract_unified_diff(text: str) -> str:
    raw = _strip_markdown_fences(text)
    if not raw:
        return ""
    if "diff --git " in raw:
        return raw[raw.find("diff --git ") :].strip()
    if looks_like_unified_diff(raw):
        # Normalize by starting from first file header when possible.
        lines = raw.splitlines()
        for idx, line in enumerate(lines):
            if line.startswith("--- ") or line.startswith("diff --git "):
                return "\n".join(lines[idx:]).strip()
        return raw.strip()
    return ""


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

    tests_text = _serialize_tests(all_tests)
    tests_serialized = json.dumps(
        [
            {
                "name": getattr(spec, "name", "test"),
                "command": getattr(spec, "command", []),
                "cwd": getattr(spec, "cwd", None),
                "env_keys": sorted(list((getattr(spec, "env", {}) or {}).keys())),
            }
            for spec in all_tests
        ],
        ensure_ascii=True,
    )
    agent_prompt = _render_agent_prompt(
        config=config,
        repo_code=repo_code,
        prompt=prompt,
        tests_text=tests_text,
        tests_json=tests_serialized,
    )
    fmt = {
        "repo_path": str(repo_code.get("path", ".")),
        "prompt": prompt,
        "agent_prompt": agent_prompt,
        "tests": tests_serialized,
        "tests_json": tests_serialized,
        "repo_id": str(repo_code.get("repo_id", "unknown_repo")),
        "instance_id": str(repo_code.get("instance_id", "unknown_instance")),
        "base_commit": str(repo_code.get("base_commit", "unknown")),
    }
    command = [str(part).format(**fmt) for part in command_template]
    cwd = Path(str(repo_code.get("path", ".")))
    timeout_sec = int(config.get("timeout_sec", 120))
    result = run_command(command, cwd=cwd, timeout_sec=timeout_sec)
    log_paths = _write_invocation_logs(artifact_dir, command, agent_prompt, result.stdout, result.stderr)

    output_mode = str(config.get("output_mode", "stdout"))
    output_parser = str(config.get("output_parser", "unified_diff_auto")).lower()
    if output_mode == "file":
        output_file = str(config.get("output_file", "patch.diff"))
        patch_path = cwd / output_file
        file_text = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
        diff_text = _extract_unified_diff(file_text) if output_parser == "unified_diff_auto" else file_text
        if not diff_text.strip() and (result.stdout or "").strip():
            stdout_text = result.stdout or ""
            diff_text = _extract_unified_diff(stdout_text) if output_parser == "unified_diff_auto" else stdout_text
    else:
        stdout_text = result.stdout or ""
        diff_text = _extract_unified_diff(stdout_text) if output_parser == "unified_diff_auto" else stdout_text

    if not diff_text.strip():
        if behavior == "skip":
            return Patch(
                unified_diff="",
                metadata={
                    "agent": agent_name,
                    "tool_available": True,
                    "returncode": result.returncode,
                    "empty_output": True,
                    "output_parser": output_parser,
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
            "output_parser": output_parser,
            "agent_prompt_hash": sha256_text(agent_prompt),
            "tests_json": tests_serialized,
            "command": " ".join(shlex.quote(x) for x in command),
            "stderr_preview": (result.stderr or "")[:500],
            **log_paths,
        },
    )
