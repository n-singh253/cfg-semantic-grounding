"""Attack finalization and validation for downstream evaluation/training."""

from __future__ import annotations

import ast
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.baseline.structural_misalignment.cfg.build import SKIP_DIRS
from src.baseline.structural_misalignment.cfg.diff import touched_files_from_patch
from src.common.artifact_store import atomic_write_json, atomic_write_text
from src.common.diff import apply_unified_diff_detailed, sanitize_unified_diff
from src.common.hashing import sha256_text
from src.common.subprocess import run_command
from src.eval.report import load_jsonl_rows

_FORBIDDEN_PATH_PARTS = {"test", "tests", "test-data"}
_FORBIDDEN_FILE_NAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
}


def normalize_patch_text(diff_text: str) -> str:
    """Return a stable normalized patch text for hashing/comparison."""
    sanitized = sanitize_unified_diff(diff_text)
    patch_text = str(sanitized.get("sanitized_diff", "") or "")
    return patch_text.strip()


def normalized_patch_hash(diff_text: str) -> str:
    normalized = normalize_patch_text(diff_text)
    return sha256_text(normalized) if normalized else ""


def _load_patch_text(path_value: str) -> str:
    path = Path(str(path_value))
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _compile_python_repo(repo_dir: Path, patch_text: str) -> Tuple[bool, Dict[str, Any]]:
    touched_python = [path for path in touched_files_from_patch(patch_text) if path.endswith(".py")]
    repo_python = [
        path
        for path in repo_dir.rglob("*.py")
        if not any(part in SKIP_DIRS for part in path.parts)
    ]
    repo_python_rel = sorted(str(path.relative_to(repo_dir)) for path in repo_python)
    details: Dict[str, Any] = {
        "touched_python_files": touched_python,
        "repo_python_files": repo_python_rel,
        "checked_files": [],
        "failures": [],
    }
    if not repo_python:
        details["skipped"] = True
        return True, details

    for file_path in repo_python:
        rel_path = str(file_path.relative_to(repo_dir))
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            ast.parse(source, filename=str(file_path))
        except SyntaxError as exc:
            details["failures"].append(
                {
                    "file": rel_path,
                    "reason": "syntax_error",
                    "message": str(exc),
                }
            )
            continue

        cmd = [sys.executable, "-m", "py_compile", str(file_path)]
        result = run_command(cmd, cwd=repo_dir)
        details["checked_files"].append(rel_path)
        if result.returncode != 0:
            details["failures"].append(
                {
                    "file": rel_path,
                    "reason": "py_compile_failed",
                    "stdout": result.stdout[:500],
                    "stderr": result.stderr[:500],
                }
            )

    return not details["failures"], details


def _forbidden_touched_files(patch_text: str) -> List[str]:
    forbidden: List[str] = []
    for raw_path in touched_files_from_patch(patch_text):
        path = raw_path.replace("\\", "/").strip("/")
        parts = set(Path(path).parts)
        name = Path(path).name
        if parts & _FORBIDDEN_PATH_PARTS:
            forbidden.append(raw_path)
            continue
        if name.startswith("test_") and name.endswith(".py"):
            forbidden.append(raw_path)
            continue
        if name.endswith("_test.py"):
            forbidden.append(raw_path)
            continue
        if name in _FORBIDDEN_FILE_NAMES:
            forbidden.append(raw_path)
            continue
    return sorted(set(forbidden))


def _validate_attack_row(row: Dict[str, Any]) -> Dict[str, Any]:
    patch_artifacts = row.get("patch_artifacts", {}) if isinstance(row.get("patch_artifacts"), dict) else {}
    ori_patch = _load_patch_text(str(patch_artifacts.get("ori_patch_path", "")))
    adv_patch = _load_patch_text(str(patch_artifacts.get("adv_patch_path", "")))
    attack_name = str(row.get("attack_name", "")).strip().lower()
    repo_path = Path(str(row.get("repo_path", "") or ""))

    normalized_ori_hash = normalized_patch_hash(ori_patch)
    normalized_adv_hash = normalized_patch_hash(adv_patch)
    validation: Dict[str, Any] = {
        "normalized_ori_patch_hash": normalized_ori_hash,
        "normalized_adv_patch_hash": normalized_adv_hash,
        "discard_reason": "",
        "kept": False,
        "apply_details": {},
        "compile": {},
    }

    normalized_patch = normalize_patch_text(adv_patch)
    if not normalized_patch:
        validation["discard_reason"] = "empty_patch"
        return validation

    if attack_name != "none" and normalized_ori_hash and normalized_adv_hash == normalized_ori_hash:
        validation["discard_reason"] = "unchanged_behavior"
        return validation

    forbidden_files = _forbidden_touched_files(normalized_patch)
    if forbidden_files:
        validation["discard_reason"] = "forbidden_file_modification"
        validation["forbidden_touched_files"] = forbidden_files
        return validation

    if not repo_path.exists():
        validation["discard_reason"] = "missing_repo_snapshot"
        return validation

    work_dir = Path(tempfile.mkdtemp(prefix="attack_finalize_"))
    patched_repo = work_dir / "repo"
    try:
        shutil.copytree(repo_path, patched_repo)
        apply_details = apply_unified_diff_detailed(patched_repo, normalized_patch)
        validation["apply_details"] = apply_details
        if not bool(apply_details.get("applied", False)):
            validation["discard_reason"] = str(apply_details.get("reason_code", "patch_apply_failed"))
            return validation

        compile_ok, compile_details = _compile_python_repo(patched_repo, normalized_patch)
        validation["compile"] = compile_details
        if not compile_ok:
            validation["discard_reason"] = "compile_failure"
            return validation
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    validation["kept"] = True
    return validation


def _selected_rows(rows: Iterable[Dict[str, Any]], *, dataset_hash: str, agent_hash: str, attack_hash: str) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("dataset_config_hash", "")) != dataset_hash:
            continue
        if str(row.get("agent_config_hash", "")) != agent_hash:
            continue
        if str(row.get("attack_config_hash", "")) != attack_hash:
            continue
        selected.append(row)
    return selected


def finalize_attack_dataset(
    *,
    out_dir: Path,
    dataset_hash: str,
    agent_hash: str,
    attack_hash: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Validate raw attack rows and produce the finalized dataset artifact."""
    raw_path = out_dir / "attack_results.jsonl"
    rows = _selected_rows(
        load_jsonl_rows(raw_path),
        dataset_hash=dataset_hash,
        agent_hash=agent_hash,
        attack_hash=attack_hash,
    )

    finalized_rows: List[Dict[str, Any]] = []
    discard_counts: Counter[str] = Counter()
    running_rows: List[str] = []

    for idx, row in enumerate(rows, start=1):
        validation = _validate_attack_row(row)
        kept = bool(validation.get("kept", False))
        discard_reason = str(validation.get("discard_reason", ""))
        if not kept and discard_reason:
            discard_counts[discard_reason] += 1

        finalized = dict(row)
        finalized.update(
            {
                "attack_dataset_finalized": True,
                "attack_kept": kept,
                "attack_discard_reason": discard_reason,
                "attack_validation": validation,
                "graph_label": 0 if str(row.get("attack_name", "")).strip().lower() == "none" else 1,
            }
        )
        if kept:
            finalized_rows.append(finalized)

        running_rows.append(
            json.dumps(
                {
                    "processed": idx,
                    "kept": len(finalized_rows),
                    "discarded": sum(discard_counts.values()),
                    "instance_id": str(row.get("instance_id", "unknown")),
                    "discard_reason": discard_reason,
                },
                sort_keys=True,
            )
        )

    dataset_path = out_dir / "attack_dataset.jsonl"
    dataset_text = "\n".join(
        json.dumps(row, sort_keys=True, ensure_ascii=True) for row in finalized_rows
    )
    atomic_write_text(dataset_path, dataset_text + ("\n" if dataset_text else ""))

    log_path = out_dir / "attack_preprocessing_log.jsonl"
    atomic_write_text(log_path, "\n".join(running_rows) + ("\n" if running_rows else ""))

    summary = {
        "attack_dataset_finalized": True,
        "raw_attack_results_path": str(raw_path),
        "attack_dataset_path": str(dataset_path),
        "preprocessing_log_path": str(log_path),
        "total_attacks_generated": len(rows),
        "total_failed_attacks_discarded": sum(discard_counts.values()),
        "final_dataset_size": len(finalized_rows),
        "discard_reasons": dict(sorted(discard_counts.items())),
    }
    atomic_write_json(out_dir / "attack_preprocessing_summary.json", summary)
    return finalized_rows, summary


def require_finalized_attack_rows(rows: List[Dict[str, Any]], source_path: Path) -> None:
    if not rows:
        return
    if all(bool(row.get("attack_dataset_finalized", False)) for row in rows):
        return
    raise ValueError(
        "Baseline evaluation requires a finalized attack dataset. "
        f"Path does not contain finalized rows: {source_path}"
    )
