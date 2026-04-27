#!/usr/bin/env python3
"""Download SWE-PolyBench metadata from HuggingFace and write local JSONL files.

Repos are NOT cloned here — the dataset loader lazy-clones on demand.
"""

from __future__ import annotations

import json
from pathlib import Path

REPOS_ROOT = Path.home() / "polybench_repos"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

HF_DATASETS = {
    "verified": "AmazonScience/SWE-PolyBench_Verified",
    "500": "AmazonScience/SWE-PolyBench_500",
    "full": "AmazonScience/SWE-PolyBench",
}


def build_jsonl(ds, variant: str, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            repo_id = row["repo"]
            org, name = repo_id.split("/", 1)
            repo_path = REPOS_ROOT / org / name

            test_cmd_raw = row.get("test_command") or ""
            if isinstance(test_cmd_raw, str) and test_cmd_raw.strip():
                test_cmd = ["bash", "-c", test_cmd_raw]
            else:
                lang = (row.get("language") or "python").lower()
                if lang == "python":
                    test_cmd = ["python3", "-m", "pytest", "-xvs"]
                elif lang == "java":
                    test_cmd = ["mvn", "test"]
                elif lang in ("javascript", "typescript"):
                    test_cmd = ["npm", "test"]
                else:
                    test_cmd = ["python3", "-m", "pytest", "-q"]

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
                "language": row.get("language", ""),
                "task_category": row.get("task_category", ""),
                "num_nodes": row.get("num_nodes"),
                "modified_nodes": row.get("modified_nodes", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> int:
    from datasets import load_dataset

    for variant, hf_name in HF_DATASETS.items():
        out_path = DATA_DIR / f"polybench_{variant}.jsonl"
        print(f"[{variant}] Loading {hf_name}...")
        try:
            ds = load_dataset(hf_name, split="test")
        except Exception as e:
            print(f"  [skip] {hf_name}: {e}")
            continue
        print(f"  Writing {out_path} ({len(ds)} instances)")
        n = build_jsonl(ds, variant, out_path)
        print(f"  Wrote {n} rows")

    print(f"\nRepos will be lazy-cloned to {REPOS_ROOT} on first use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
