#!/usr/bin/env python3
"""Download SWE-bench Lite metadata from HuggingFace and write local JSONL."""

from __future__ import annotations

import json
from pathlib import Path

REPOS_ROOT = Path.home() / "swebench_lite_repos"
DATA_OUT = Path(__file__).resolve().parents[1] / "data" / "swebench_lite.jsonl"


def main() -> int:
    from datasets import load_dataset

    print("[1/2] Loading SWE-bench Lite from HuggingFace...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(f"       {len(ds)} instances loaded")

    print(f"[2/2] Writing {DATA_OUT}")
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with open(DATA_OUT, "w", encoding="utf-8") as f:
        for row in ds:
            repo_id = row["repo"]
            org, name = repo_id.split("/", 1)
            repo_path = REPOS_ROOT / org / name

            fail_to_pass = row.get("FAIL_TO_PASS") or []
            if isinstance(fail_to_pass, str):
                fail_to_pass = json.loads(fail_to_pass)
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
                "repo_language": "python",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1

    print(f"       Wrote {rows_written} rows to {DATA_OUT}")
    print(f"       Repos will be lazy-cloned to {REPOS_ROOT} on first use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
