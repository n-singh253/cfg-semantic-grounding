#!/usr/bin/env python3
"""Export matrix summary.csv from nested results.jsonl files."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.runner import collect_matrix_rows
from src.eval.report import write_summary_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Export summary CSV from matrix outputs")
    parser.add_argument("--out", required=True, help="Matrix output root")
    args = parser.parse_args()
    out_dir = Path(args.out)
    rows = collect_matrix_rows(out_dir)
    write_summary_csv(out_dir / "summary.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
