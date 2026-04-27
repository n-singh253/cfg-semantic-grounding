#!/usr/bin/env python3
"""Download FeatureBench metadata from HuggingFace and write local JSONL files.

Repos are NOT cloned here — the dataset loader lazy-clones on demand.
"""

from __future__ import annotations

import json
from pathlib import Path

REPOS_ROOT = Path.home() / "featurebench_repos"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build_jsonl(ds, variant: str, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            repo_id = row["repo"]
            org, name = repo_id.split("/", 1)
            repo_path = REPOS_ROOT / org / name

            fail_to_pass = row.get("FAIL_TO_PASS") or []
            test_cmd = ["python3", "-m", "pytest", "-xvs"] + (fail_to_pass if isinstance(fail_to_pass, list) else [])

            record = {
                "instance_id": row["instance_id"],
                "repo": repo_id,
                "repo_id": repo_id,
                "repo_path": str(repo_path),
                "base_commit": row["base_commit"],
                "problem_statement": row["problem_statement"],
                "test_command": test_cmd,
                "patch": row.get("patch", ""),
                "test_patch": row.get("test_patch", ""),
                "FAIL_TO_PASS": fail_to_pass,
                "PASS_TO_PASS": row.get("PASS_TO_PASS") or [],
                "image_name": row.get("image_name", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> int:
    from datasets import load_dataset

    print("[1/2] Loading FeatureBench from HuggingFace...")
    ds_all = load_dataset("LiberCoders/FeatureBench")

    for variant in ("lite", "full"):
        if variant not in ds_all:
            print(f"  [skip] split '{variant}' not found")
            continue
        ds = ds_all[variant]
        out_path = DATA_DIR / f"featurebench_{variant}.jsonl"
        print(f"[2/2] Writing {out_path} ({len(ds)} instances)")
        n = build_jsonl(ds, variant, out_path)
        print(f"       Wrote {n} rows")

    print(f"       Repos will be lazy-cloned to {REPOS_ROOT} on first use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
