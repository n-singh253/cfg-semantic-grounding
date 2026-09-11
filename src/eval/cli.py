"""Minimal publication CLI entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from src.baseline.registry import list_baselines


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="CFG semantic grounding publication harness")
    sub = parser.add_subparsers(dest="command", required=True)

    row_defense = sub.add_parser("run_defense", help="Run a row-level or dataset-level defense baseline.")
    row_defense.add_argument(
        "--rows",
        default=None,
        help="Path to rows.jsonl or rows directory. Required for row-level baselines.",
    )
    row_defense.add_argument("--baseline", required=True)
    row_defense.add_argument("--fidelity-mode", default="llm", choices=["llm", "surrogate_debug"])
    row_defense.add_argument("--out", required=True)
    row_defense.add_argument("--config-dir", default="configs")
    row_defense.add_argument("--repos-root", default=None)
    row_defense.add_argument("--workers", type=int, default=1)
    row_defense.add_argument("--limit", type=int, default=None)
    row_defense.add_argument("--code-base", action="append", default=[])
    row_defense.add_argument("--code-base-file", default=None)
    row_defense.add_argument("--recursive", action="store_true")
    row_defense.add_argument("--strict-rows", action="store_true")
    row_defense.add_argument("--no-resume", dest="resume", action="store_false")
    row_defense.set_defaults(resume=True)
    row_defense.add_argument("--scanner-timeout-sec", type=int, default=120)

    sub.add_parser("list_baselines", help="List registered defense/baseline plugins.")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "list_baselines":
        for name in sorted(list_baselines()):
            print(name)
        return 0

    if args.command == "run_defense":
        from src.eval.defense import run_defense

        code_bases = list(args.code_base)
        if args.code_base_file:
            code_bases.extend(
                line.strip()
                for line in Path(args.code_base_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        results = run_defense(
            rows_path=Path(args.rows) if args.rows else None,
            baseline_name=args.baseline,
            fidelity_mode=args.fidelity_mode,
            out_dir=Path(args.out),
            config_dir=Path(args.config_dir),
            repos_root=Path(args.repos_root) if args.repos_root else None,
            workers=args.workers,
            limit=args.limit,
            code_bases=code_bases,
            recursive=bool(args.recursive),
            strict_rows=bool(args.strict_rows),
            resume=bool(args.resume),
            scanner_timeout_sec=int(args.scanner_timeout_sec),
        )
        return 1 if any(row.get("status") != "success" for row in results) else 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
