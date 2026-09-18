"""Utilities for the publication row format.

The publication-facing data format intentionally keeps only four fields:

    {"code_base": str, "injection": 0 | 1, "prompt": str, "patch": str}

All richer provenance belongs in runner outputs, manifests, or side artifacts.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.common.subprocess import command_exists, run_command


ROW_FIELDS = ("code_base", "injection", "prompt", "patch")
LEGACY_ROW_FIELDS = ("code_base", "label", "prompt", "patch")
ROW_FILE_NAMES = ("rows.jsonl", "rows_synthetic.jsonl")


class RowSchemaError(ValueError):
    """Raised when a row does not satisfy the row schema."""


def _line_context(source: Path | None, line_no: int | None) -> str:
    if source is None:
        return ""
    if line_no is None:
        return f" in {source}"
    return f" in {source}:{line_no}"


def normalize_row(
    row: Dict[str, Any],
    *,
    source: Path | None = None,
    line_no: int | None = None,
    strict_fields: bool = False,
) -> Dict[str, Any]:
    """Validate and normalize one row.

    Extra fields are dropped unless ``strict_fields`` is true, because old export
    scripts may have temporarily carried metadata while the publication row
    contract stays four-field.
    """

    if not isinstance(row, dict):
        raise RowSchemaError(f"expected JSON object{_line_context(source, line_no)}")

    uses_legacy_label = "injection" not in row and "label" in row
    required_fields = LEGACY_ROW_FIELDS if uses_legacy_label else ROW_FIELDS

    missing = [field for field in required_fields if field not in row]
    if missing:
        raise RowSchemaError(
            f"missing required fields {missing}{_line_context(source, line_no)}"
        )

    if strict_fields:
        extra = sorted(set(row).difference(required_fields))
        if extra:
            raise RowSchemaError(
                f"unexpected fields {extra}{_line_context(source, line_no)}"
            )

    code_base = str(row.get("code_base") or "").strip()
    if not code_base:
        raise RowSchemaError(f"code_base is empty{_line_context(source, line_no)}")

    label_raw = row.get("label") if uses_legacy_label else row.get("injection")
    if label_raw in {0, "0"}:
        label = 0
    elif label_raw in {1, "1"}:
        label = 1
    else:
        raise RowSchemaError(
            f"label must be 0 or 1, got {label_raw!r}{_line_context(source, line_no)}"
        )

    return {
        "code_base": code_base,
        "label": label,
        "injection": label,
        "prompt": str(row.get("prompt") or ""),
        "patch": str(row.get("patch") or ""),
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip().lstrip("\x00")
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def discover_rows_files(path: Path, *, recursive: bool = False) -> List[Path]:
    """Return row files represented by ``path``.

    A file path is returned directly. A directory path resolves to
    ``rows.jsonl`` or ``rows_synthetic.jsonl`` when present; with
    ``recursive=True`` it returns all nested row files.
    """

    root = path.expanduser().resolve()
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    if recursive:
        return sorted(
            p
            for name in ROW_FILE_NAMES
            for p in root.rglob(name)
            if p.is_file()
        )
    for name in ROW_FILE_NAMES:
        direct = root / name
        if direct.exists():
            return [direct]
    nested = sorted(
        p
        for name in ROW_FILE_NAMES
        for p in root.glob(f"*/{name}")
        if p.is_file()
    )
    if nested:
        return nested
    raise FileNotFoundError(f"No rows.jsonl or rows_synthetic.jsonl found under {root}")


def load_rows(
    path: Path,
    *,
    recursive: bool = False,
    strict_fields: bool = False,
    limit: int | None = None,
    code_bases: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    wanted = {item.strip() for item in (code_bases or []) if item.strip()}
    rows: List[Dict[str, Any]] = []
    for file_path in discover_rows_files(path, recursive=recursive):
        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip().lstrip("\x00")
                if not line:
                    continue
                row = normalize_row(
                    json.loads(line),
                    source=file_path,
                    line_no=line_no,
                    strict_fields=strict_fields,
                )
                if wanted and row["code_base"] not in wanted:
                    continue
                rows.append(row)
                if limit is not None and len(rows) >= max(0, limit):
                    return rows
    return rows


def label_counts(rows: Iterable[Dict[str, Any]]) -> Dict[int, int]:
    counts = Counter(int(row["label"]) for row in rows)
    return {label: counts.get(label, 0) for label in (0, 1)}


def split_rows_by_code_base(
    rows: Sequence[Dict[str, Any]],
    *,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Split rows by code_base so paired benign/adversarial rows cannot leak."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    by_code_base: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_code_base.setdefault(str(row["code_base"]), []).append(dict(row))

    code_bases = sorted(by_code_base)
    rng = random.Random(seed)
    rng.shuffle(code_bases)

    if len(code_bases) <= 1:
        train_ids = set(code_bases)
        test_ids: set[str] = set()
    else:
        split_at = int(round(len(code_bases) * train_ratio))
        split_at = min(max(1, split_at), len(code_bases) - 1)
        train_ids = set(code_bases[:split_at])
        test_ids = set(code_bases[split_at:])

    train_rows = [row for code_base in code_bases if code_base in train_ids for row in by_code_base[code_base]]
    test_rows = [row for code_base in code_bases if code_base in test_ids for row in by_code_base[code_base]]

    manifest = {
        "split_key": "code_base",
        "train_ratio": train_ratio,
        "seed": seed,
        "row_count": len(rows),
        "code_base_count": len(code_bases),
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "train_code_base_count": len(train_ids),
        "test_code_base_count": len(test_ids),
        "train_label_counts": label_counts(train_rows),
        "test_label_counts": label_counts(test_rows),
        "train_code_bases": sorted(train_ids),
        "test_code_bases": sorted(test_ids),
    }
    overlap = set(manifest["train_code_bases"]).intersection(manifest["test_code_bases"])
    if overlap:
        raise RuntimeError(f"split leakage detected for code_base values: {sorted(overlap)[:10]}")
    return train_rows, test_rows, manifest


def infer_benchmark(code_base: str, source_path: Path | None = None) -> str:
    raw = code_base.lower()
    source = str(source_path or "").lower()
    if raw.startswith("lcb_") or "livecodebench" in source:
        return "livecodebench"
    if ".lv" in raw or "featurebench" in source:
        return "featurebench"
    if "__" in raw or "swe-bench" in source or "swebench" in source:
        return "swebench"
    return "unknown"


def _swebench_repo_slug_candidates(code_base: str) -> List[Path]:
    match = re.match(r"^(?P<org>[^_]+)__(?P<repo>.+?)-\d+$", code_base)
    if not match:
        return []
    return [Path(match.group("org")) / match.group("repo")]


def _featurebench_repo_slug_candidates(code_base: str) -> List[Path]:
    match = re.match(r"^(?P<org>[^_]+)__(?P<repo>[^.]+)\.", code_base)
    if not match:
        return []
    return [Path(match.group("org")) / match.group("repo")]


def repo_path_candidates(code_base: str, repos_root: Path) -> List[Path]:
    root = repos_root.expanduser().resolve()
    candidates = [
        root / code_base,
        root / "release_latest" / code_base,
        root / "full" / code_base,
        root / "lite" / code_base,
        root / "test" / code_base,
    ]
    candidates.extend(root / rel for rel in _swebench_repo_slug_candidates(code_base))
    candidates.extend(root / rel for rel in _featurebench_repo_slug_candidates(code_base))

    deduped: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def resolve_repo_path(code_base: str, repos_root: Path | None) -> Path | None:
    if repos_root is None:
        return None
    candidates = repo_path_candidates(code_base, repos_root)
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return candidates[0] if candidates else None


def reset_git_repo(repo_path: Path, ref: str = "HEAD") -> Dict[str, Any]:
    """Reset a downloaded benchmark repo before or after an isolated operation."""

    if not repo_path.exists():
        return {"ok": False, "reason": "repo_missing", "repo_path": str(repo_path)}
    if not command_exists("git"):
        return {"ok": False, "reason": "git_missing", "repo_path": str(repo_path)}

    probe = run_command(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_path, timeout_sec=30)
    if probe.returncode != 0 or "true" not in (probe.stdout or "").lower():
        return {"ok": False, "reason": "not_git_repo", "repo_path": str(repo_path)}

    commands = [
        ["git", "reset", "--hard", ref or "HEAD"],
        ["git", "clean", "-fdx"],
    ]
    details: List[Dict[str, Any]] = []
    for command in commands:
        result = run_command(command, cwd=repo_path, timeout_sec=120)
        details.append(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout[-1000:],
                "stderr": result.stderr[-1000:],
            }
        )
        if result.returncode != 0:
            return {
                "ok": False,
                "reason": "reset_failed",
                "repo_path": str(repo_path),
                "details": details,
            }
    return {"ok": True, "reason": "ok", "repo_path": str(repo_path), "details": details}


def copy_repo_without_git(source: Path, target: Path) -> None:
    """Copy a repo worktree for isolated patch/scanner operations."""

    ignore = shutil.ignore_patterns(
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
    )
    shutil.copytree(source, target, ignore=ignore, dirs_exist_ok=False)
