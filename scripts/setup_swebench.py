#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DATASET_OPTIONS: Dict[str, Dict[str, str]] = {
    "swebench_lite": {
        "hf_dataset": "princeton-nlp/SWE-bench_Lite",
        "output_name": "swebench_lite_local.jsonl",
        "repos_env": "CFG_SWEBENCH_LITE_REPOS_DIR",
        "repos_dir": "data/repos/swebench_lite",
        "limit": "300",
    },
    "swebench_pro": {
        "hf_dataset": "princeton-nlp/SWE-bench",
        "output_name": "swebench_pro_local.jsonl",
        "repos_env": "CFG_SWEBENCH_PRO_REPOS_DIR",
        "repos_dir": "data/repos/swebench_pro",
        "limit": "0",
    },
    "swebench_plus": {
        "hf_dataset": "princeton-nlp/SWE-bench_Multimodal",
        "output_name": "swebench_plus_local.jsonl",
        "repos_env": "CFG_SWEBENCH_PLUS_REPOS_DIR",
        "repos_dir": "data/repos/swebench_plus",
        "limit": "0",
    },
}


def log(event: str, **kwargs: Any) -> None:
    suffix = ""
    if kwargs:
        suffix = " | " + " ".join(f"{key}={value}" for key, value in sorted(kwargs.items()))
    print(f"[setup-swebench] {event}{suffix}", flush=True)


def env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up local SWE-Bench checkouts")
    parser.add_argument("--dataset", default=env_value("CFG_SWEBENCH_DATASET", default="swebench_lite"), choices=sorted(DATASET_OPTIONS))
    parser.add_argument("--split", default=env_value("CFG_SWEBENCH_SPLIT", default="test"))
    parser.add_argument("--offset", type=int, default=int(env_value("CFG_SWEBENCH_OFFSET", default="0")))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repos-dir", default=env_value("CFG_SWEBENCH_REPOS_DIR"))
    parser.add_argument("--hf-dataset", default=env_value("CFG_SWEBENCH_HF_DATASET"))
    parser.add_argument("--force-reclone", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel checkout workers.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_under_root(path: str, root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = root / resolved
    return resolved.resolve()


def load_dataset_rows(hf_dataset: str, split: str, offset: int, limit: int) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'datasets'. Install it with: pip install datasets") from exc

    if limit <= 0:
        selected_split = split if offset <= 0 else f"{split}[{offset}:]"
    else:
        end = offset + limit
        selected_split = f"{split}[{offset}:{end}]" if offset > 0 else f"{split}[:{end}]"
    dataset = load_dataset(hf_dataset, split=selected_split)
    return [dict(row) for row in dataset]


def run_git(command: List[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(command, cwd=str(cwd) if cwd else None, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "unknown git failure"
        raise RuntimeError(f"{' '.join(command)} failed: {details}")


def is_valid_git_checkout(path: Path) -> bool:
    if not path.exists():
        return False
    completed = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def ensure_checkout(repo_slug: str, base_commit: str, target_dir: Path, force_reclone: bool) -> None:
    repo_url = f"https://github.com/{repo_slug}.git"

    if target_dir.exists() and force_reclone:
        remove_path(target_dir)

    if target_dir.exists() and not is_valid_git_checkout(target_dir):
        log("reclone-invalid-checkout", path=target_dir)
        remove_path(target_dir)

    if not target_dir.exists():
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        run_git(["git", "clone", repo_url, str(target_dir)])
    else:
        try:
            run_git(["git", "fetch", "--all", "--tags", "--prune"], cwd=target_dir)
        except RuntimeError as exc:
            log("reclone-fetch-failed", path=target_dir, reason=str(exc).splitlines()[0])
            remove_path(target_dir)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            run_git(["git", "clone", repo_url, str(target_dir)])

    run_git(["git", "checkout", "--force", base_commit], cwd=target_dir)


def parse_test_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(payload, list):
            return [str(item) for item in payload if str(item).strip()]
    return []


def build_test_command(row: Dict[str, Any]) -> List[str]:
    fail_to_pass = parse_test_list(row.get("FAIL_TO_PASS"))
    pass_to_pass = parse_test_list(row.get("PASS_TO_PASS"))
    selected_tests = fail_to_pass or pass_to_pass[:25]
    command = ["python3", "-m", "pytest", "-q"]
    if selected_tests:
        command.extend(selected_tests)
    return command


def normalize_row(row: Dict[str, Any], repo_path: Path) -> Dict[str, Any]:
    instance_id = str(row.get("instance_id") or row.get("id") or repo_path.name)
    repo_slug = str(row.get("repo") or row.get("repo_id") or "")
    return {
        "instance_id": instance_id,
        "repo_id": repo_slug,
        "repo": repo_slug,
        "problem_statement": str(row.get("problem_statement") or row.get("prompt") or ""),
        "repo_path": str(repo_path.resolve()),
        "base_commit": str(row.get("base_commit") or "HEAD"),
        "test_command": build_test_command(row),
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def materialize_row(
    *,
    index: int,
    total: int,
    row: Dict[str, Any],
    repos_dir: Path,
    force_reclone: bool,
) -> Tuple[int, Dict[str, Any]]:
    repo_slug = str(row.get("repo") or row.get("repo_id") or "").strip()
    base_commit = str(row.get("base_commit") or "").strip()
    instance_id = str(row.get("instance_id") or row.get("id") or f"instance-{index}")
    if not repo_slug:
        raise RuntimeError(f"{instance_id}: missing repo slug in downloaded row")
    if not base_commit:
        raise RuntimeError(f"{instance_id}: missing base_commit in downloaded row")

    target_dir = repos_dir / instance_id
    log("checkout", index=f"{index}/{total}", instance_id=instance_id, repo=repo_slug, commit=base_commit[:12])
    ensure_checkout(repo_slug, base_commit, target_dir, force_reclone)
    return index, normalize_row(row, target_dir)


def main() -> int:
    args = parse_args()
    root = repo_root()
    defaults = DATASET_OPTIONS[args.dataset]

    dataset_repos_dir = env_value(defaults["repos_env"], "CFG_SWEBENCH_REPOS_DIR", default=defaults["repos_dir"])
    repos_dir = resolve_under_root(args.repos_dir or dataset_repos_dir, root)
    output_path = repos_dir / defaults["output_name"]
    hf_dataset = args.hf_dataset or defaults["hf_dataset"]
    limit = int(args.limit if args.limit is not None else env_value("CFG_SWEBENCH_LIMIT", default=defaults["limit"]))
    workers = max(1, int(args.workers))

    log("start", dataset=args.dataset, hf_dataset=hf_dataset, split=args.split, offset=args.offset, limit=limit, workers=workers)
    log("paths", output=output_path, repos_dir=repos_dir)

    rows = load_dataset_rows(hf_dataset, args.split, args.offset, limit)
    materialized_rows: List[Dict[str, Any] | None] = [None] * len(rows)
    if workers == 1:
        for index, row in enumerate(rows, start=1):
            _, record = materialize_row(
                index=index,
                total=len(rows),
                row=row,
                repos_dir=repos_dir,
                force_reclone=args.force_reclone,
            )
            materialized_rows[index - 1] = record
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    materialize_row,
                    index=index,
                    total=len(rows),
                    row=row,
                    repos_dir=repos_dir,
                    force_reclone=args.force_reclone,
                )
                for index, row in enumerate(rows, start=1)
            ]
            for future in as_completed(futures):
                index, record = future.result()
                materialized_rows[index - 1] = record

    final_rows = [row for row in materialized_rows if row is not None]
    write_jsonl(output_path, final_rows)
    log("wrote-jsonl", rows=len(final_rows), path=output_path)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted")
        raise SystemExit(130)
