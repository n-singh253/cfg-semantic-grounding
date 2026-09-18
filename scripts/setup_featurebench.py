#!/usr/bin/env python3
"""Prepare FeatureBench rows as local git repositories for defense scans.

FeatureBench's HF rows are not plain SWE-Bench rows:

* Level 1 rows use ``base_commit`` after the feature exists. The dataset
  ``patch`` is a corruption/removal patch. We apply it to create the actual
  pre-feature worktree used as the scan/apply base.
* Level 2 rows have an empty ``patch`` and describe a small package to build
  from a minimal/empty repository.

This script materializes those states and writes a local manifest for auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List


SOURCE_REPOS_ROOT = Path(
    os.environ.get("FEATUREBENCH_SOURCE_REPOS", str(Path.home() / "featurebench_repos"))
)
INSTANCE_REPOS_ROOT = Path(
    os.environ.get("FEATUREBENCH_INSTANCE_REPOS", str(Path.home() / "featurebench_instance_repos"))
)
MATERIALIZATION_VERSION = "featurebench_row_defense_ready_v1"
_SOURCE_REPO_CACHE: Dict[str, Path] = {}
_SOURCE_REPO_LOCKS: Dict[str, threading.RLock] = {}
_SOURCE_REPO_LOCKS_GUARD = threading.Lock()


def _run(
    command: List[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed rc={result.returncode}: {' '.join(command)}\n"
            f"cwd={cwd}\nstdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}"
        )
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _repo_path(root: Path, repo_id: str) -> Path:
    org, name = repo_id.split("/", 1)
    return root / org / name


def _source_repo_lock(repo_id: str) -> threading.RLock:
    with _SOURCE_REPO_LOCKS_GUARD:
        lock = _SOURCE_REPO_LOCKS.get(repo_id)
        if lock is None:
            lock = threading.RLock()
            _SOURCE_REPO_LOCKS[repo_id] = lock
        return lock


def _ensure_source_repo(repo_id: str) -> Path:
    with _source_repo_lock(repo_id):
        if repo_id in _SOURCE_REPO_CACHE:
            return _SOURCE_REPO_CACHE[repo_id]

        source = _repo_path(SOURCE_REPOS_ROOT, repo_id)
        if not (source / ".git").exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            print(f"  [clone] {repo_id} -> {source}")
            _run(["git", "clone", "--quiet", f"https://github.com/{repo_id}.git", str(source)], timeout=1200)
        else:
            print(f"  [source] {repo_id} -> {source}")
        _run(["git", "fetch", "--quiet", "origin"], cwd=source, timeout=1200)
        _SOURCE_REPO_CACHE[repo_id] = source
        return source


def _remove_existing_instance(source_repo: Path | None, path: Path) -> None:
    if source_repo and path.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=str(source_repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    if path.exists():
        shutil.rmtree(path)


def _configure_commit_identity(repo: Path) -> None:
    _run(["git", "config", "user.email", "featurebench@example.invalid"], cwd=repo)
    _run(["git", "config", "user.name", "FeatureBench Prep"], cwd=repo)


def _commit_all(repo: Path, message: str) -> str:
    _configure_commit_identity(repo)
    _run(["git", "add", "-A"], cwd=repo)
    status = _run(["git", "status", "--porcelain"], cwd=repo).stdout.strip()
    if status:
        _run(["git", "commit", "-m", message], cwd=repo)
    return _run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()


def _materialize_level1(row: Dict[str, Any], variant: str, force: bool) -> Dict[str, Any]:
    repo_id = str(row["repo"])
    source = _ensure_source_repo(repo_id)
    instance_id = str(row["instance_id"])
    dest = INSTANCE_REPOS_ROOT / variant / instance_id
    patch = str(row.get("patch") or "")
    if not patch.strip():
        raise RuntimeError(f"{instance_id}: Level 1 materialization requires a nonempty patch")

    if force or not (dest / ".git").exists():
        print(f"  [lv1] {instance_id}")
        with _source_repo_lock(repo_id):
            _remove_existing_instance(source, dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "worktree", "add", "--detach", "--force", str(dest), str(row["base_commit"])], cwd=source)
        _run(["git", "apply", "--whitespace=nowarn", "-p1"], cwd=dest, input_text=patch, timeout=300)
        prepared_commit = _commit_all(dest, f"Prepare FeatureBench corrupted state for {instance_id}")
    else:
        prepared_commit = _run(["git", "rev-parse", "HEAD"], cwd=dest).stdout.strip()

    record = _base_record(row, variant, dest, prepared_commit, level="lv1")
    record.update(
        {
            "hf_base_commit": row["base_commit"],
            "featurebench_patch_direction": "applied_as_corruption_patch",
            "featurebench_patch_hash": _sha256_text(patch),
            "source_repo_path": str(source),
        }
    )
    return record


def _materialize_level2(row: Dict[str, Any], variant: str, force: bool) -> Dict[str, Any]:
    instance_id = str(row["instance_id"])
    dest = INSTANCE_REPOS_ROOT / variant / instance_id
    if force or not (dest / ".git").exists():
        print(f"  [lv2] {instance_id}")
        _remove_existing_instance(None, dest)
        dest.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "--quiet"], cwd=dest)
        (dest / "README.md").write_text(str(row.get("problem_statement") or "") + "\n", encoding="utf-8")
        prepared_commit = _commit_all(dest, f"Prepare FeatureBench empty state for {instance_id}")
    else:
        prepared_commit = _run(["git", "rev-parse", "HEAD"], cwd=dest).stdout.strip()

    record = _base_record(row, variant, dest, prepared_commit, level="lv2")
    record.update(
        {
            "hf_base_commit": row["base_commit"],
            "featurebench_patch_direction": "empty_level2_repo",
            "featurebench_patch_hash": "",
            "source_repo_path": "",
        }
    )
    # Level 2 official tests live in FeatureBench's evaluation image. Locally we
    # expose a basic sanity command for auditing the materialized repo.
    record["original_test_command"] = record["test_command"]
    record["test_command"] = ["python3", "-m", "compileall", "."]
    return record


def _base_record(row: Dict[str, Any], variant: str, repo_path: Path, prepared_commit: str, level: str) -> Dict[str, Any]:
    fail_to_pass = row.get("FAIL_TO_PASS") or []
    test_cmd = ["python3", "-m", "pytest", "-xvs"] + (fail_to_pass if isinstance(fail_to_pass, list) else [])
    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "repo_id": row["repo"],
        "repo_path": str(repo_path),
        "base_commit": prepared_commit,
        "problem_statement": row["problem_statement"],
        "test_command": test_cmd,
        "patch": row.get("patch", ""),
        "test_patch": row.get("test_patch", ""),
        "FAIL_TO_PASS": fail_to_pass,
        "PASS_TO_PASS": row.get("PASS_TO_PASS") or [],
        "image_name": row.get("image_name", ""),
        "repo_settings": row.get("repo_settings", ""),
        "featurebench_level": level,
        "featurebench_materialized": True,
        "featurebench_materialization_version": MATERIALIZATION_VERSION,
    }


def _filter_rows(
    ds: Iterable[Dict[str, Any]],
    *,
    instance_ids: List[str],
    limit: int | None,
) -> List[Dict[str, Any]]:
    wanted = set(instance_ids)
    rows: List[Dict[str, Any]] = []
    for raw in ds:
        row = dict(raw)
        if wanted and str(row.get("instance_id", "")) not in wanted:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= max(0, limit):
            break
    return rows


def _materialize_row(row: Dict[str, Any], variant: str, force: bool) -> Dict[str, Any]:
    level = "lv2" if not str(row.get("patch") or "").strip() else "lv1"
    return (
        _materialize_level2(row, variant, force)
        if level == "lv2"
        else _materialize_level1(row, variant, force)
    )


def build_jsonl(rows: Iterable[Dict[str, Any]], variant: str, out_path: Path, force: bool, workers: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    records: List[Dict[str, Any] | None] = [None] * len(rows_list)
    workers = max(1, int(workers))

    if workers == 1:
        for index, row in enumerate(rows_list):
            records[index] = _materialize_row(row, variant, force)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_materialize_row, row, variant, force): index
                for index, row in enumerate(rows_list)
            }
            for future in as_completed(futures):
                records[futures[future]] = future.result()

    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record is None:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return sum(1 for record in records if record is not None)


def main() -> int:
    global SOURCE_REPOS_ROOT, INSTANCE_REPOS_ROOT

    parser = argparse.ArgumentParser(description="Prepare FeatureBench repositories for defense scans.")
    parser.add_argument("--force", action="store_true", help="Recreate all per-instance repositories.")
    parser.add_argument("--variant", choices=["lite", "full", "all"], default="all")
    parser.add_argument(
        "--source-repos-dir",
        default=None,
        help="Directory for cloned upstream GitHub repositories. Defaults to FEATUREBENCH_SOURCE_REPOS or ~/featurebench_repos.",
    )
    parser.add_argument(
        "--instance-repos-dir",
        "--repos-dir",
        dest="instance_repos_dir",
        default=None,
        help="Directory for materialized per-instance repositories. Defaults to FEATUREBENCH_INSTANCE_REPOS or ~/featurebench_instance_repos.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Materialize only this instance id. Can be passed more than once.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Materialize at most N rows after filtering.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel materialization workers.")
    args = parser.parse_args()

    if args.source_repos_dir:
        SOURCE_REPOS_ROOT = Path(args.source_repos_dir).expanduser().resolve()
    if args.instance_repos_dir:
        INSTANCE_REPOS_ROOT = Path(args.instance_repos_dir).expanduser().resolve()

    from datasets import load_dataset

    print("[1/2] Loading FeatureBench from HuggingFace...")
    ds_all = load_dataset("LiberCoders/FeatureBench")
    variants = ["lite", "full"] if args.variant == "all" else [args.variant]

    for variant in variants:
        if variant not in ds_all:
            print(f"  [skip] split '{variant}' not found")
            continue
        rows = _filter_rows(ds_all[variant], instance_ids=args.instance_id, limit=args.limit)
        filtered_note = ""
        if args.instance_id or args.limit is not None:
            filtered_note = " (filtered sample; rerun without filters before full experiments)"
        out_path = INSTANCE_REPOS_ROOT / f"featurebench_{variant}_local.jsonl"
        print(f"[2/2] Materializing {variant} -> {out_path} ({len(rows)} instances){filtered_note}")
        n = build_jsonl(rows, variant, out_path, args.force, args.workers)
        print(f"       Wrote {n} rows")

    print(f"       Source repos: {SOURCE_REPOS_ROOT}")
    print(f"       Instance repos: {INSTANCE_REPOS_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
