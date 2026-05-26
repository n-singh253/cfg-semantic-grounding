"""Test execution wrapper used by patch evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from src.common.artifact_store import atomic_write_text
from src.common.subprocess import run_command
from src.common.types import TestSpec


def _test_timeout_sec() -> int:
    raw_value = os.environ.get("CFG_TEST_TIMEOUT_SEC", "120")
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 120


def run_tests(test_specs: List[TestSpec], repo_dir: Path, log_path: Path) -> Tuple[bool, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    all_passed = True
    total_runtime = 0.0
    lines: List[str] = []
    timeout_sec = _test_timeout_sec()

    for spec in test_specs:
        cwd = Path(spec.cwd) if spec.cwd else repo_dir
        result = run_command(
            spec.command,
            cwd=cwd,
            env=spec.env or None,
            timeout_sec=timeout_sec or None,
        )
        total_runtime += result.runtime_sec
        lines.append(f"$ {' '.join(result.command)}")
        lines.append(f"returncode={result.returncode}")
        if result.stdout:
            lines.append("[stdout]")
            lines.append(result.stdout)
        if result.stderr:
            lines.append("[stderr]")
            lines.append(result.stderr)
        lines.append("-" * 60)
        if result.returncode != 0:
            all_passed = False

    atomic_write_text(log_path, "\n".join(lines))
    return all_passed, int(total_runtime)
