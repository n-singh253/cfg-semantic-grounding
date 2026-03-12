"""Helpers for command-driven agent wrappers."""

from __future__ import annotations

import os
import json
import re
import shlex
import shutil
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
    artifact_tag = str(repo_code.get("agent_artifact_tag", "") or "").strip()
    if run_root:
        base = Path(str(run_root)) / "artifacts" / "agents" / instance_id / agent_name
        return base / artifact_tag if artifact_tag else base
    base = Path(str(repo_code.get("path", "."))) / ".agent_artifacts" / agent_name
    return base / artifact_tag if artifact_tag else base


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


def _first_diff_start_index(lines: List[str]) -> int:
    for idx, line in enumerate(lines):
        if line.startswith("diff --git ") or line.startswith("--- "):
            return idx
    return -1


def _extract_diff_block(raw: str) -> str:
    lines = raw.splitlines()
    start = _first_diff_start_index(lines)
    if start < 0:
        return ""

    header_prefixes = (
        "diff --git ",
        "index ",
        "--- ",
        "+++ ",
        "@@",
        "new file mode ",
        "deleted file mode ",
        "similarity index ",
        "rename from ",
        "rename to ",
        "old mode ",
        "new mode ",
        "Binary files ",
    )

    out: List[str] = []
    for line in lines[start:]:
        if line.startswith("```"):
            break
        if (
            line.startswith(header_prefixes)
            or line.startswith("+")
            or line.startswith("-")
            or line.startswith(" ")
            or line == r"\ No newline at end of file"
        ):
            out.append(line)
            continue
        if out:
            # Stop once prose/non-diff text begins after diff started.
            break
    return "\n".join(out).strip()


def _extract_unified_diff(text: str) -> str:
    raw = text or ""
    if not raw:
        return ""
    extracted = _extract_diff_block(raw)
    if extracted and looks_like_unified_diff(extracted):
        return extracted

    fenced = _strip_markdown_fences(raw)
    extracted_fenced = _extract_diff_block(fenced)
    if extracted_fenced and looks_like_unified_diff(extracted_fenced):
        return extracted_fenced

    if looks_like_unified_diff(fenced):
        return fenced.strip()
    return ""


def _nvm_node_major_from_path(path: str) -> int:
    match = re.search(r"/\.nvm/versions/node/v(\d+)\.", path)
    if not match:
        return -1
    try:
        return int(match.group(1))
    except Exception:
        return -1


def _resolve_cli_executable(executable: str) -> str:
    resolved = shutil.which(executable)
    if not resolved:
        return executable

    if executable not in {"gemini", "gemini-cli"}:
        return resolved

    current_major = _nvm_node_major_from_path(resolved)
    if current_major >= 20:
        return resolved

    nvm_root = Path.home() / ".nvm" / "versions" / "node"
    if not nvm_root.exists():
        return resolved

    best = resolved
    best_major = current_major
    for candidate in sorted(nvm_root.glob("v*/bin/gemini")):
        major = _nvm_node_major_from_path(str(candidate))
        if major >= 20 and major > best_major:
            best = str(candidate)
            best_major = major
    return best


def _node_for_nvm_gemini(gemini_executable: str) -> str:
    # /home/<user>/.nvm/versions/node/v20.20.0/bin/gemini -> .../bin/node
    path = Path(gemini_executable)
    if path.name not in {"gemini", "gemini-cli"}:
        return ""
    node_path = path.parent / "node"
    if node_path.exists():
        return str(node_path)
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
    resolved_executable = _resolve_cli_executable(executable)
    behavior = str(config.get("missing_tool_behavior", "fail")).lower()
    artifact_dir = _agent_artifact_dir(agent_name, repo_code)
    if not command_exists(resolved_executable):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        missing_tool_path = artifact_dir / "missing_tool.json"
        atomic_write_json(
            missing_tool_path,
            {
                "agent": agent_name,
                "missing_tool": resolved_executable,
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
            f"{agent_name}: missing required CLI tool '{resolved_executable}'. "
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
    if command:
        if executable in {"gemini", "gemini-cli"}:
            node_bin = _node_for_nvm_gemini(resolved_executable)
            if node_bin:
                command = [node_bin, resolved_executable, *command[1:]]
            else:
                command[0] = resolved_executable
        else:
            command[0] = resolved_executable
    cwd = Path(str(repo_code.get("path", ".")))
    timeout_sec = int(config.get("timeout_sec", 120))
    # Gemini CLI reads GEMINI_API_KEY, but this harness commonly uses GOOGLE_API_KEY.
    # Forward it automatically for command-driven Gemini agents when needed.
    env_override: Dict[str, str] = {}
    if executable in {"gemini", "gemini-cli"}:
        if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
            env_override["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

    result = run_command(
        command,
        cwd=cwd,
        env=env_override or None,
        timeout_sec=timeout_sec,
    )
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
                    "env_overrides": sorted(env_override.keys()),
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
            "env_overrides": sorted(env_override.keys()),
            "output_parser": output_parser,
            "agent_prompt_hash": sha256_text(agent_prompt),
            "tests_json": tests_serialized,
            "command": " ".join(shlex.quote(x) for x in command),
            "stderr_preview": (result.stderr or "")[:500],
            **log_paths,
        },
    )
