"""Safe subprocess runner with captured stdout/stderr."""

from __future__ import annotations

import subprocess
import time
import shutil
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CommandResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    runtime_sec: float


def run_command(
    command: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_sec: Optional[int] = None,
) -> CommandResult:
    start = time.time()
    merged_env = os.environ.copy()
    merged_env.setdefault("LANG", "C.UTF-8")
    merged_env.setdefault("LC_ALL", "C.UTF-8")
    if env is not None:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        runtime = time.time() - start
        return CommandResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            runtime_sec=runtime,
        )
    except subprocess.TimeoutExpired as exc:
        runtime = time.time() - start

        def _to_text(value: object) -> str:
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)

        stdout = _to_text(exc.stdout)
        stderr_base = _to_text(exc.stderr)
        timeout_note = f"Command timed out after {timeout_sec} seconds."
        stderr = f"{stderr_base}\n{timeout_note}".strip() if stderr_base else timeout_note
        return CommandResult(
            command=command,
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            runtime_sec=runtime,
        )


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None
