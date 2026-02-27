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
    merged_env = None
    if env is not None:
        merged_env = os.environ.copy()
        merged_env.update(env)
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


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None
