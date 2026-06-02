#!/usr/bin/env python3
"""Validate that SWE-Bench clean rows apply at their base commits."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from src.common.git_utils import reset_git_checkout
from src.common.subprocess import command_exists, run_command
from src.eval.patch_eval import apply_patch_with_details
from src.eval.report import load_jsonl_rows


def _repo_key(repo_path: Path) -> str:
    return hashlib.sha256(str(repo_path.resolve()).encode("utf-8")).hexdigest()[:16]


def _repo_label(repo_path: Path) -> str:
    parts = [part for part in repo_path.parts if part not in {"/", ""}]
    label = "__".join(parts[-2:]) if len(parts) >= 2 else repo_path.name or "repo"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in label)


def _materialize_repo(src_repo: Path, clone_root: Path, clones: dict[str, Path]) -> Path:
    key = _repo_key(src_repo)
    existing = clones.get(key)
    if existing is not None:
        return existing

    dst_repo = clone_root / f"{_repo_label(src_repo)}__{key}"
    if (src_repo / ".git").exists() and command_exists("git"):
        clone = run_command(
            ["git", "clone", "--quiet", "--shared", str(src_repo), str(dst_repo)],
            timeout_sec=300,
        )
        if clone.returncode != 0:
            msg = (clone.stderr or clone.stdout or "").strip()
            raise RuntimeError(f"git clone failed for {src_repo}: {msg}")
    else:
        shutil.copytree(src_repo, dst_repo, symlinks=True)

    clones[key] = dst_repo
    return dst_repo


def _patch_path(row: dict[str, Any], root: Path) -> Path:
    artifacts = row.get("patch_artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    raw = str(
        artifacts.get("final_patch_path")
        or artifacts.get("adv_patch_path")
        or artifacts.get("ori_patch_path")
        or ""
    )
    path = Path(raw)
    return path if path.is_absolute() else root / path


def validate_rows(
    *,
    rows: list[dict[str, Any]],
    root: Path,
    clone_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clones: dict[str, Path] = {}
    failures: list[dict[str, Any]] = []
    applied = 0
    skipped = 0

    for row in rows:
        instance_id = str(row.get("instance_id", "unknown"))
        attack_name = str(row.get("attack_name", ""))
        if attack_name and attack_name != "none":
            skipped += 1
            continue

        src_repo = Path(str(row.get("repo_path", "")))
        base_commit = str(row.get("base_commit", "")).strip()
        patch_path = _patch_path(row, root)

        if not src_repo.exists():
            failures.append(
                {
                    "instance_id": instance_id,
                    "reason_code": "missing_repo",
                    "message": str(src_repo),
                }
            )
            continue
        if not patch_path.exists():
            failures.append(
                {
                    "instance_id": instance_id,
                    "reason_code": "missing_patch",
                    "message": str(patch_path),
                }
            )
            continue

        repo = _materialize_repo(src_repo, clone_root, clones)
        reset_ok, reset_res, clean_res, _ = reset_git_checkout(repo, base_commit or "HEAD")
        if not reset_ok:
            failures.append(
                {
                    "instance_id": instance_id,
                    "reason_code": "reset_failed",
                    "message": "\n".join(
                        [
                            (reset_res.stdout or "").strip(),
                            (reset_res.stderr or "").strip(),
                            (clean_res.stdout or "").strip(),
                            (clean_res.stderr or "").strip(),
                        ]
                    ).strip(),
                }
            )
            continue

        patch_text = patch_path.read_text(encoding="utf-8")
        details = apply_patch_with_details(repo, patch_text)
        if details.get("applied"):
            applied += 1
            continue

        failures.append(
            {
                "instance_id": instance_id,
                "base_commit": base_commit,
                "repo_path": str(src_repo),
                "patch_path": str(patch_path),
                "reason_code": str(details.get("reason_code", "apply_failed")),
                "message": str(details.get("raw_output", ""))[:1000],
            }
        )

    summary = {
        "total_rows": len(rows),
        "skipped_non_clean_rows": skipped,
        "validated_clean_rows": len(rows) - skipped,
        "applied": applied,
        "failed": len(failures),
        "apply_ok_rate": applied / max(1, len(rows) - skipped),
    }
    return summary, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-results", required=True, help="Path to a SWE-Bench none attack_dataset.jsonl.")
    parser.add_argument("--out", default="", help="Optional JSON report path.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    attack_results = Path(args.attack_results)
    rows = load_jsonl_rows(attack_results)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    with tempfile.TemporaryDirectory(prefix="swebench-clean-validate-") as tmpdir:
        summary, failures = validate_rows(rows=rows, root=root, clone_root=Path(tmpdir))

    report = {"summary": summary, "failures": failures}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        print(f"first_failure={json.dumps(failures[0], sort_keys=True)}")
    return 0 if args.allow_failures or not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
