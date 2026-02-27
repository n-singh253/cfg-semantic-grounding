"""Report builders for integration_spec and dataset_report artifacts."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit_short() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def build_integration_spec(
    *,
    run_id: str,
    schema_version: str,
    fidelity_mode: str,
    cli_invocation: str,
    selected_plugins: Dict[str, str],
    selected_configs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    created_at = utc_now()
    return {
        "schema_version": schema_version,
        "run_id": run_id,
        "created_at": created_at,
        "timestamps": {"created_at": created_at},
        "fidelity_mode": fidelity_mode,
        "cli_invocation": cli_invocation,
        "git_commit": git_commit_short(),
        "selected_plugins": selected_plugins,
        "selected_configs": selected_configs,
    }


def build_dataset_report(
    *,
    dataset: str,
    split: str,
    total_loaded: int,
    total_selected: int,
    skipped: int,
    failed: int,
    failure_reasons: List[str],
    warnings: List[str],
    tool_availability: Dict[str, bool],
) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "created_at": utc_now(),
        "dataset": dataset,
        "split": split,
        "counts": {
            "loaded": total_loaded,
            "selected": total_selected,
            "skipped": skipped,
            "failed": failed,
        },
        "failure_reasons": failure_reasons,
        "warnings": warnings,
        "tool_availability": tool_availability,
    }
