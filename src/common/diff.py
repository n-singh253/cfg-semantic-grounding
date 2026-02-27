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


def apply_unified_diff(repo_dir: Path, diff_text: str) -> Tuple[bool, str]:
    if not diff_text.strip():
        return True, "No patch to apply"

    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8") as fh:
        fh.write(diff_text)
        patch_path = Path(fh.name)

    result = run_command(["patch", "-p1", "--forward", "--input", str(patch_path)], cwd=repo_dir)
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return result.returncode == 0, output.strip()
