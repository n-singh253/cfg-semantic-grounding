"""Git checkout helpers used by runners and CLI agent wrappers."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Tuple

from src.common.subprocess import CommandResult, run_command


_FAILED_ENCODING_RE = re.compile(r"failed to encode '([^']+)'")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _remove_failed_encoding_paths(repo_path: Path, stderr: str) -> List[str]:
    repo_root = repo_path.resolve()
    removed: List[str] = []
    for match in _FAILED_ENCODING_RE.finditer(stderr or ""):
        rel = Path(match.group(1))
        if rel.is_absolute() or ".." in rel.parts:
            continue
        target = (repo_root / rel).resolve()
        if not _is_relative_to(target, repo_root):
            continue
        try:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            removed.append(str(rel))
        except OSError:
            continue
    return removed


def _override_working_tree_encoding(repo_path: Path, paths: List[str]) -> None:
    if not paths:
        return
    info_attributes = repo_path / ".git" / "info" / "attributes"
    if not info_attributes.parent.exists():
        return
    existing = info_attributes.read_text(encoding="utf-8", errors="replace") if info_attributes.exists() else ""
    lines = []
    for path in paths:
        line = f"{path} -working-tree-encoding"
        if line not in existing:
            lines.append(line)
    if lines:
        with info_attributes.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write("\n".join(lines) + "\n")


def reset_git_checkout(
    repo_path: Path,
    target: str = "HEAD",
) -> Tuple[bool, CommandResult, CommandResult, List[str]]:
    """Reset and clean a checkout, recovering from Git working-tree encoding errors."""

    reset_result = run_command(["git", "reset", "--hard", target], cwd=repo_path)
    recovered_paths: List[str] = []
    if "failed to encode" in (reset_result.stderr or ""):
        recovered_paths.extend(_remove_failed_encoding_paths(repo_path, reset_result.stderr))
        if recovered_paths:
            _override_working_tree_encoding(repo_path, recovered_paths)
            reset_result = run_command(["git", "reset", "--hard", target], cwd=repo_path)

    clean_result = run_command(["git", "clean", "-fd"], cwd=repo_path)
    status_result = run_command(["git", "status", "--short"], cwd=repo_path)
    if "failed to encode" in (status_result.stderr or ""):
        new_recovered = _remove_failed_encoding_paths(repo_path, status_result.stderr)
        if new_recovered:
            recovered_paths.extend(path for path in new_recovered if path not in recovered_paths)
            _override_working_tree_encoding(repo_path, recovered_paths)
            reset_result = run_command(["git", "reset", "--hard", target], cwd=repo_path)
            clean_result = run_command(["git", "clean", "-fd"], cwd=repo_path)

    ok = reset_result.returncode == 0 and clean_result.returncode == 0
    return ok, reset_result, clean_result, recovered_paths
