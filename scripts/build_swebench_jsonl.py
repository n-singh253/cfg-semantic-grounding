#!/usr/bin/env python3
"""Build harness JSONL from SWE-Bench metadata + local repo snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List


def _parse_tests(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            return []
    return []


def _iter_repo_filters(values: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for raw in values:
        piece = str(raw).strip()
        if piece:
            out.add(piece)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create local SWE-Bench JSONL for cfg-semantic-grounding runs."
    )
    parser.add_argument(
        "--hf-dataset",
        default="princeton-nlp/SWE-bench_Lite",
        help="Hugging Face dataset id to read metadata from.",
    )
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument(
        "--repos-root",
        default="~/repos/swe_bench",
        help="Root with local snapshots laid out as <repo_short>/<base_commit>/",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of rows to emit.")
    parser.add_argument(
        "--max-tests",
        type=int,
        default=8,
        help="Max number of FAIL_TO_PASS tests to include per row.",
    )
    parser.add_argument(
        "--output",
        default="data/swebench_lite_real10.jsonl",
        help="Output JSONL path (repo-relative unless absolute).",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Optional repo short-name filter (repeat flag). Example: --repo django",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with `pip install datasets`."
        ) from exc

    root = Path(__file__).resolve().parents[1]
    repos_root = Path(args.repos_root).expanduser().resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (root / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wanted_repos = _iter_repo_filters(args.repo)
    ds = load_dataset(args.hf_dataset, split=args.split)

    selected: List[dict[str, Any]] = []
    skipped_missing_repo = 0
    skipped_repo_filter = 0

    for row in ds:
        repo_full = str(row.get("repo", "")).strip()
        repo_short = repo_full.split("/")[-1] if repo_full else ""
        if wanted_repos and repo_short not in wanted_repos:
            skipped_repo_filter += 1
            continue

        base_commit = str(row.get("base_commit", "")).strip()
        if not repo_short or not base_commit:
            continue

        repo_path = repos_root / repo_short / base_commit
        if not repo_path.exists():
            skipped_missing_repo += 1
            continue

        tests = _parse_tests(row.get("FAIL_TO_PASS"))
        if args.max_tests >= 0:
            tests = tests[: args.max_tests]
        if tests:
            test_command = ["python3", "-m", "pytest", "-q", *tests]
        else:
            test_command = ["python3", "-m", "pytest", "-q"]

        selected.append(
            {
                "instance_id": str(row.get("instance_id", "")),
                "problem_statement": str(row.get("problem_statement", "")),
                "repo_path": str(repo_path),
                "base_commit": base_commit,
                "repo_id": repo_full or repo_short,
                "test_command": test_command,
            }
        )

        if len(selected) >= max(0, int(args.limit)):
            break

    with output_path.open("w", encoding="utf-8") as fh:
        for entry in selected:
            fh.write(json.dumps(entry, ensure_ascii=True) + "\n")

    print(f"[build-swebench-jsonl] output={output_path}")
    print(f"[build-swebench-jsonl] rows={len(selected)}")
    print(f"[build-swebench-jsonl] skipped_missing_repo={skipped_missing_repo}")
    if wanted_repos:
        print(f"[build-swebench-jsonl] skipped_repo_filter={skipped_repo_filter}")

    if len(selected) == 0:
        print(
            "[build-swebench-jsonl] WARNING: no rows selected. "
            "Check --repos-root and optional --repo filters."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
