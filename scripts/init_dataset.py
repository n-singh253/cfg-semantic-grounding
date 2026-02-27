#!/usr/bin/env python3
"""Initialize local SWE-Bench dataset config + sample JSONL row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import yaml


DATASET_CFG_MAP = {
    "swebench_lite": "lite.yaml",
    "swebench_pro": "pro.yaml",
    "swebench_plus": "plus.yaml",
}


def _load_yaml(path: Path) -> Dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize local dataset file and config path")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["swebench_lite", "swebench_pro", "swebench_plus"],
    )
    parser.add_argument("--repo-path", required=True, help="Absolute path to local checkout")
    parser.add_argument("--base-commit", default="HEAD", help="Base commit for the sample row")
    parser.add_argument(
        "--instance-id",
        default="demo-1",
        help="Sample instance id for the generated JSONL row",
    )
    parser.add_argument(
        "--output",
        default="data/swebench_local.jsonl",
        help="JSONL output path (relative to repo root unless absolute)",
    )
    parser.add_argument(
        "--skip-config-update",
        action="store_true",
        help="Only write JSONL and do not patch dataset config",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "instance_id": args.instance_id,
        "problem_statement": "Fix the issue and keep tests passing.",
        "repo_path": args.repo_path,
        "base_commit": args.base_commit,
        "test_command": ["python3", "-m", "unittest", "discover"],
    }
    output_path.write_text(json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[init-dataset] wrote sample jsonl: {output_path}")

    if args.skip_config_update:
        return 0

    cfg_file = root / "configs" / "datasets" / DATASET_CFG_MAP[args.dataset]
    cfg = _load_yaml(cfg_file)
    cfg["data_path"] = str(output_path if output_path.is_absolute() else output_path.relative_to(root))
    cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[init-dataset] updated dataset config: {cfg_file}")
    print(f"[init-dataset] data_path={cfg['data_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
