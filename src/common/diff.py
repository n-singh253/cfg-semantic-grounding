"""Unified diff helpers."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from src.common.subprocess import run_command


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_FILE_HEADER_PREFIXES = (
    "diff --git ",
    "index ",
    "--- ",
    "+++ ",
    "new file mode ",
    "deleted file mode ",
    "similarity index ",
    "rename from ",
    "rename to ",
    "old mode ",
    "new mode ",
    "Binary files ",
)


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


def _extract_diff_lines(raw: str) -> List[str]:
    lines = raw.splitlines()
    start = -1
    for idx, line in enumerate(lines):
        if line.startswith("diff --git ") or line.startswith("--- "):
            start = idx
            break
    if start < 0:
        return []

    out: List[str] = []
    started = False
    for line in lines[start:]:
        if line.startswith("```"):
            break
        if (
            line.startswith(_FILE_HEADER_PREFIXES)
            or line.startswith("@@ ")
            or line.startswith("+")
            or line.startswith("-")
            or line.startswith(" ")
            or line == r"\ No newline at end of file"
            or line == ""
        ):
            out.append(line)
            started = True
            continue
        if started:
            break
    return out


def _normalize_file_header(line: str) -> str:
    if not (line.startswith("--- ") or line.startswith("+++ ")):
        return line
    marker = line[:4]
    payload = line[4:].strip()
    if not payload:
        return line

    if "\t" in payload:
        path, tail = payload.split("\t", 1)
        suffix = f"\t{tail}"
    else:
        path = payload
        suffix = ""
    path = path.strip()

    if path == "/dev/null":
        return f"{marker}{path}{suffix}"
    if marker == "--- " and not path.startswith("a/"):
        path = f"a/{path.lstrip('/')}"
    if marker == "+++ " and not path.startswith("b/"):
        path = f"b/{path.lstrip('/')}"
    return f"{marker}{path}{suffix}"


def sanitize_unified_diff(diff_text: str) -> Dict[str, object]:
    raw = diff_text or ""
    if not raw.strip():
        return {
            "sanitized_diff": "",
            "validation": {
                "valid": False,
                "reason_code": "empty_patch",
                "details": {"message": "Patch text is empty."},
            },
        }

    extracted = _extract_diff_lines(raw)
    if not extracted:
        return {
            "sanitized_diff": "",
            "validation": {
                "valid": False,
                "reason_code": "no_diff_headers",
                "details": {"message": "No unified diff headers found."},
            },
        }

    sanitized: List[str] = []
    rewritten_hunks = 0
    recovered_blank_lines = 0
    recovered_missing_prefix_lines = 0

    i = 0
    while i < len(extracted):
        line = extracted[i]

        if line.startswith("--- "):
            sanitized.append(_normalize_file_header(line))
            i += 1
            continue
        if line.startswith("+++ "):
            sanitized.append(_normalize_file_header(line))
            i += 1
            continue
        if line.startswith(_FILE_HEADER_PREFIXES):
            sanitized.append(line)
            i += 1
            continue

        if line.startswith("@@ "):
            match = _HUNK_HEADER_RE.match(line)
            if not match:
                partial = "\n".join(sanitized)
                if partial and not partial.endswith("\n"):
                    partial = f"{partial}\n"
                return {
                    "sanitized_diff": partial,
                    "validation": {
                        "valid": False,
                        "reason_code": "malformed_hunk_header",
                        "details": {"line": line},
                    },
                }
            old_start = int(match.group(1))
            old_count_raw = int(match.group(2)) if match.group(2) else 1
            new_start = int(match.group(3))
            new_count_raw = int(match.group(4)) if match.group(4) else 1
            tail = match.group(5) or ""
            header_index = len(sanitized)
            sanitized.append(line)
            i += 1

            old_count = 0
            new_count = 0
            while i < len(extracted):
                body_line = extracted[i]
                if body_line.startswith("@@ "):
                    break
                if body_line.startswith("diff --git ") or body_line.startswith("--- "):
                    break

                if body_line == r"\ No newline at end of file":
                    sanitized.append(body_line)
                    i += 1
                    continue

                if body_line == "":
                    body_line = " "
                    recovered_blank_lines += 1
                elif body_line[0] not in {" ", "+", "-"}:
                    body_line = f" {body_line}"
                    recovered_missing_prefix_lines += 1

                if body_line[0] in {" ", "-"}:
                    old_count += 1
                if body_line[0] in {" ", "+"}:
                    new_count += 1
                sanitized.append(body_line)
                i += 1

            rewritten = old_count != old_count_raw or new_count != new_count_raw
            if rewritten:
                rewritten_hunks += 1
            sanitized[header_index] = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{tail}"
            continue

        i += 1

    sanitized_text = "\n".join(sanitized)
    if sanitized_text and not sanitized_text.endswith("\n"):
        sanitized_text = f"{sanitized_text}\n"

    if not looks_like_unified_diff(sanitized_text):
        return {
            "sanitized_diff": sanitized_text,
            "validation": {
                "valid": False,
                "reason_code": "sanitized_not_unified_diff",
                "details": {"message": "Sanitized patch is not a valid unified diff."},
            },
        }

    return {
        "sanitized_diff": sanitized_text,
        "validation": {
            "valid": True,
            "reason_code": "ok",
            "details": {
                "rewritten_hunks": rewritten_hunks,
                "recovered_blank_lines": recovered_blank_lines,
                "recovered_missing_prefix_lines": recovered_missing_prefix_lines,
            },
        },
    }


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


def apply_unified_diff_detailed(repo_dir: Path, diff_text: str) -> Dict[str, object]:
    sanitized = sanitize_unified_diff(diff_text)
    validation = sanitized.get("validation", {})
    patch_text = str(sanitized.get("sanitized_diff", ""))

    if not bool(validation.get("valid")):
        return {
            "applied": False,
            "method_used": "none",
            "reason_code": str(validation.get("reason_code", "invalid_patch")),
            "raw_output": str(validation.get("details", {})),
            "validation": validation,
            "sanitized_diff": patch_text,
        }

    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8") as fh:
        fh.write(patch_text)
        patch_path = Path(fh.name)

    strip_levels = [1, 0] if _prefers_strip_one(patch_text) else [0, 1]
    attempts: List[Tuple[int, str]] = []
    tried: set[int] = set()

    try:
        for strip in strip_levels:
            if strip in tried:
                continue
            tried.add(strip)

            check_cmd = [
                "git",
                "apply",
                "--check",
                f"-p{strip}",
                "--recount",
                "--whitespace=nowarn",
                str(patch_path),
            ]
            checked = run_command(check_cmd, cwd=repo_dir)
            checked_out = _patch_output(checked.stdout, checked.stderr)
            attempts.append((strip, f"git_apply_check: {checked_out}".strip()))
            if checked.returncode != 0:
                continue

            apply_cmd = [
                "git",
                "apply",
                f"-p{strip}",
                "--recount",
                "--whitespace=nowarn",
                str(patch_path),
            ]
            apply = run_command(apply_cmd, cwd=repo_dir)
            apply_out = _patch_output(apply.stdout, apply.stderr)
            if apply.returncode == 0:
                return {
                    "applied": True,
                    "method_used": f"git_apply_p{strip}",
                    "reason_code": "applied",
                    "raw_output": apply_out,
                    "validation": validation,
                    "sanitized_diff": patch_text,
                }
            attempts.append((strip, f"git_apply: {apply_out}".strip()))

        for strip in strip_levels:
            if strip not in tried:
                tried.add(strip)
            apply_3way_cmd = [
                "git",
                "apply",
                "--3way",
                f"-p{strip}",
                "--recount",
                "--whitespace=nowarn",
                str(patch_path),
            ]
            apply_3way = run_command(apply_3way_cmd, cwd=repo_dir)
            apply_3way_out = _patch_output(apply_3way.stdout, apply_3way.stderr)
            if apply_3way.returncode == 0:
                return {
                    "applied": True,
                    "method_used": f"git_apply_3way_p{strip}",
                    "reason_code": "applied",
                    "raw_output": apply_3way_out,
                    "validation": validation,
                    "sanitized_diff": patch_text,
                }
            attempts.append((strip, f"git_apply_3way: {apply_3way_out}".strip()))

        patch_variants: List[Tuple[str, List[str]]] = [
            ("patch", []),
            ("patch_ws", ["-l"]),
            ("patch_fuzz3", ["-F", "3"]),
            ("patch_fuzz3_ws", ["-F", "3", "-l"]),
            ("patch_fuzz10_ws", ["-F", "10", "-l"]),
        ]
        for strip in strip_levels:
            for variant_name, extra_flags in patch_variants:
                dry_cmd = [
                    "patch",
                    f"-p{strip}",
                    "--forward",
                    "--batch",
                    "--dry-run",
                    *extra_flags,
                    "--input",
                    str(patch_path),
                ]
                dry_res = run_command(dry_cmd, cwd=repo_dir)
                dry_out = _patch_output(dry_res.stdout, dry_res.stderr)
                attempts.append((strip, f"{variant_name}_dry_run: {dry_out}".strip()))
                if dry_res.returncode != 0:
                    continue

                patch_cmd = [
                    "patch",
                    f"-p{strip}",
                    "--forward",
                    "--batch",
                    *extra_flags,
                    "--input",
                    str(patch_path),
                ]
                patch_res = run_command(patch_cmd, cwd=repo_dir)
                patch_out = _patch_output(patch_res.stdout, patch_res.stderr)
                if patch_res.returncode == 0:
                    return {
                        "applied": True,
                        "method_used": f"{variant_name}_p{strip}",
                        "reason_code": "applied",
                        "raw_output": patch_out,
                        "validation": validation,
                        "sanitized_diff": patch_text,
                    }
                attempts.append((strip, f"{variant_name}: {patch_out}".strip()))

        lines = ["Patch apply failed for all attempted strip levels."]
        for strip, out in attempts:
            snippet = out.strip() or "(no output)"
            lines.append(f"- -p{strip}: {snippet}")
        return {
            "applied": False,
            "method_used": "none",
            "reason_code": "apply_failed_all_methods",
            "raw_output": "\n".join(lines).strip(),
            "validation": validation,
            "sanitized_diff": patch_text,
        }
    finally:
        try:
            patch_path.unlink(missing_ok=True)
        except Exception:
            pass


def apply_unified_diff(repo_dir: Path, diff_text: str) -> Tuple[bool, str]:
    detail = apply_unified_diff_detailed(repo_dir, diff_text)
    return bool(detail.get("applied", False)), str(detail.get("raw_output", ""))
