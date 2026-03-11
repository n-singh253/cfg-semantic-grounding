"""Unified diff helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from src.common.subprocess import run_command


def parse_unified_diff(diff_text: str) -> List[Dict]:
    lines = diff_text.splitlines() if diff_text else []
    chunks: List[Dict] = []
    current_file = ""
    current_hunk: List[str] = []
    hunk_header = ""
    counter = 0

    def flush() -> None:
        nonlocal counter, current_hunk, hunk_header
        if not current_hunk:
            return
        counter += 1
        chunks.append(
            {
                "chunk_id": f"chunk_{counter}",
                "file_path": current_file,
                "hunk_header": hunk_header,
                "raw_hunk": "\n".join(current_hunk),
            }
        )
        current_hunk = []
        hunk_header = ""

    for line in lines:
        if line.startswith("+++ "):
            raw = line[4:].strip()
            current_file = raw[2:] if raw.startswith("b/") else raw
            continue
        if line.startswith("@@"):
            flush()
            hunk_header = line
            current_hunk = [line]
            continue
        if current_hunk:
            current_hunk.append(line)

    flush()
    return chunks


def looks_like_unified_diff(text: str) -> bool:
    raw = text or ""
    return "diff --git " in raw or ("--- " in raw and "+++ " in raw and "@@" in raw)


def _prefers_strip_one(diff_text: str) -> bool:
    for line in diff_text.splitlines():
        if line.startswith("diff --git a/"):
            return True
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            return True
    return False


def _patch_output(stdout: str, stderr: str) -> str:
    out = (stdout or "").strip()
    err = (stderr or "").strip()
    if out and err:
        return f"{out}\n{err}".strip()
    return (out or err).strip()


def apply_unified_diff(repo_dir: Path, diff_text: str) -> Tuple[bool, str]:
    if not diff_text.strip():
        return True, "No patch to apply"

    # GNU patch is sensitive to missing terminal newlines in patch files.
    if not diff_text.endswith("\n"):
        diff_text = diff_text + "\n"

    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8") as fh:
        fh.write(diff_text)
        patch_path = Path(fh.name)
    strip_levels = [1, 0] if _prefers_strip_one(diff_text) else [0, 1]
    attempts: List[Tuple[int, str]] = []
    tried: set[int] = set()

    try:
        for strip in strip_levels:
            if strip in tried:
                continue
            tried.add(strip)

            dry_cmd = ["patch", f"-p{strip}", "--forward", "--batch", "--dry-run", "--input", str(patch_path)]
            dry = run_command(dry_cmd, cwd=repo_dir)
            dry_out = _patch_output(dry.stdout, dry.stderr)
            attempts.append((strip, dry_out))
            if dry.returncode != 0:
                continue

            apply_cmd = ["patch", f"-p{strip}", "--forward", "--batch", "--input", str(patch_path)]
            apply = run_command(apply_cmd, cwd=repo_dir)
            apply_out = _patch_output(apply.stdout, apply.stderr)
            if apply.returncode == 0:
                prefix = f"Applied with strip level -p{strip}"
                return True, f"{prefix}\n{apply_out}".strip()
            attempts.append((strip, f"apply failed after dry-run success\n{apply_out}".strip()))

        lines = ["Patch apply failed for all attempted strip levels."]
        for strip, out in attempts:
            snippet = out.strip() or "(no output)"
            lines.append(f"- -p{strip}: {snippet}")
        return False, "\n".join(lines).strip()
    finally:
        try:
            patch_path.unlink(missing_ok=True)
        except Exception:
            pass
