#!/usr/bin/env python3
"""Validate whether a run directory is experiment-ready."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _collect_result_rows(run_dir: Path) -> List[Dict]:
    direct = run_dir / "results.jsonl"
    if direct.exists():
        return _load_rows(direct)
    rows: List[Dict] = []
    for path in sorted(run_dir.glob("*/results.jsonl")):
        rows.extend(_load_rows(path))
    return rows


def _find_incomplete_subruns(run_dir: Path) -> List[str]:
    incomplete: List[str] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        has_markers = (child / "dataset_report.json").exists() or (child / "integration_spec.json").exists()
        has_results = (child / "results.jsonl").exists()
        if has_markers and not has_results:
            incomplete.append(str(child))
    return incomplete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check run readiness for experiment execution.")
    parser.add_argument("--run-dir", required=True, help="Run output directory (matrix root or single run root).")
    parser.add_argument("--expected-instances", type=int, required=True, help="Expected number of completed rows.")
    parser.add_argument(
        "--min-apply-rate",
        type=float,
        default=0.60,
        help="Minimum fraction of rows with apply_ok=true (default: 0.60).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"[error] run-dir not found: {run_dir}")
        return 1

    rows = _collect_result_rows(run_dir)
    total = len(rows)
    apply_ok = sum(1 for row in rows if bool(row.get("apply_ok", False)))
    apply_rate = (apply_ok / total) if total else 0.0

    summary_exists = (run_dir / "summary.csv").exists()
    incomplete_subruns = _find_incomplete_subruns(run_dir)
    expected_match = total == args.expected_instances
    apply_ok_threshold = apply_rate >= args.min_apply_rate

    print(f"[metrics] run_dir={run_dir}")
    print(f"[metrics] total_rows={total}")
    print(f"[metrics] apply_ok_rows={apply_ok}")
    print(f"[metrics] apply_rate={apply_rate:.3f}")
    print(f"[metrics] expected_instances={args.expected_instances}")
    print(f"[metrics] min_apply_rate={args.min_apply_rate:.3f}")
    print(f"[checks] summary_exists={summary_exists}")
    print(f"[checks] expected_match={expected_match}")
    print(f"[checks] incomplete_subruns={len(incomplete_subruns)}")
    print(f"[checks] apply_rate_ok={apply_ok_threshold}")

    if incomplete_subruns:
        print("[detail] incomplete_subrun_paths:")
        for path in incomplete_subruns:
            print(f"  - {path}")

    ok = summary_exists and expected_match and not incomplete_subruns and apply_ok_threshold
    if ok:
        print("[result] experiment-ready")
        return 0

    print("[result] NOT experiment-ready")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
