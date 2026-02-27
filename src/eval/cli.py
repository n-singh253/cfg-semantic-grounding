"""CLI entrypoints for running one experiment or a matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from src.common.config import load_yaml
from src.baseline.registry import list_baselines
from src.eval.runner import run_matrix, run_one


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="SWE-Bench attack/defense harness")
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("run_one", help="Run a single experiment.")
    one.add_argument("--dataset", required=True, choices=["toy", "swebench_lite", "swebench_pro", "swebench_plus"])
    one.add_argument("--split", default="test")
    one.add_argument("--instance-id", default=None, help="Single instance id or comma-separated ids")
    one.add_argument("--limit", type=int, default=None)
    one.add_argument("--agent", required=True)
    one.add_argument("--attack", required=True)
    one.add_argument("--baseline", required=True)
    one.add_argument("--fidelity-mode", default="llm", choices=["llm", "surrogate_debug"])
    one.add_argument(
        "--swexploit-adv-patches",
        default=None,
        help="Optional path to SWExploit prebuilt adversarial patches JSON/JSONL.",
    )
    one.add_argument("--out", required=True)
    one.add_argument("--config-dir", default="configs")
    one.add_argument("--run-judges", action="store_true")

    matrix = sub.add_parser("run_matrix", help="Run a matrix experiment.")
    matrix.add_argument("--config", required=True)
    matrix.add_argument("--out", required=True)
    matrix.add_argument("--config-dir", default="configs")
    matrix.add_argument(
        "--swexploit-adv-patches",
        default=None,
        help="Optional global SWExploit prebuilt adversarial patches JSON/JSONL path.",
    )

    sub.add_parser("list_baselines", help="List registered defense/baseline plugins.")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cli_invocation = " ".join(["python", "-m", "src.eval.cli", *sys.argv[1:]])
    if args.command == "list_baselines":
        for name in sorted(list_baselines()):
            print(name)
        return 0

    if args.command == "run_one":
        instance_ids = None
        if args.instance_id:
            instance_ids = [x.strip() for x in args.instance_id.split(",") if x.strip()]
        run_one(
            dataset_name=args.dataset,
            split=args.split,
            instance_ids=instance_ids,
            limit=args.limit,
            agent_name=args.agent,
            attack_name=args.attack,
            baseline_name=args.baseline,
            fidelity_mode=args.fidelity_mode,
            out_dir=Path(args.out),
            config_dir=Path(args.config_dir),
            cli_invocation=cli_invocation,
            run_judges=bool(args.run_judges),
            swexploit_adv_patches=args.swexploit_adv_patches,
        )
        return 0

    matrix_cfg = load_yaml(Path(args.config))
    if args.swexploit_adv_patches:
        matrix_cfg = dict(matrix_cfg)
        matrix_cfg["swexploit_adv_patches"] = args.swexploit_adv_patches
    run_matrix(
        matrix_config=matrix_cfg,
        out_dir=Path(args.out),
        config_dir=Path(args.config_dir),
        cli_invocation=cli_invocation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
