"""Line-level patch parser for unified diffs.

Ported from the cfge project. While the existing ``diff.py`` focuses on structural
CFG diffing (before/after node comparison), this module extracts the raw set of
changed line numbers per file from a unified diff. This is used by the risk analysis
module to correlate patch impact with CFG nodes.
"""

from __future__ import annotations

import re
from typing import Dict, Set


def parse_patch_lines(patch_content: str) -> Dict[str, Set[int]]:
    """Parse a unified diff and return changed (added) line numbers per file.

    Args:
        patch_content: Unified diff text (e.g., from ``git diff``).

    Returns:
        Mapping of file path to set of line numbers that were added/modified
        in the new version.
    """
    file_changes: Dict[str, Set[int]] = {}
    current_file: str | None = None
    current_lines: Set[int] = set()
    current_line_idx = 0

    file_header = re.compile(r"^\+\+\+ b/(.*)")
    hunk_header = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

    for line in patch_content.splitlines():
        fmatch = file_header.match(line)
        if fmatch:
            if current_file:
                file_changes[current_file] = current_lines
            current_file = fmatch.group(1)
            current_lines = set()
            continue

        if not current_file:
            continue

        hmatch = hunk_header.match(line)
        if hmatch:
            current_line_idx = int(hmatch.group(1)) - 1
            continue

        if line.startswith("+++") or line.startswith("---"):
            continue

        if line.startswith("+"):
            current_line_idx += 1
            current_lines.add(current_line_idx)
        elif line.startswith(" "):
            current_line_idx += 1
        # Lines starting with '-' don't count in new-file numbering.

    if current_file:
        file_changes[current_file] = current_lines

    return file_changes
