"""CFG diff computation and patch-grounded candidate node extraction."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.baseline.structural_misalignment.cfg.build import build_cfg_for_files
from src.common.diff import apply_unified_diff


def _node_signature(node: Dict[str, Any], file_path: Optional[str] = None) -> str:
    node_id = str(node.get("node_id", ""))
    func_name = "unknown"
    if "::" in node_id:
        parts = node_id.split("::")
        if len(parts) >= 2:
            if not file_path:
                file_path = parts[0]
            func_name = parts[1]
    start_line = int(node.get("start_line", 0) or 0)
    end_line = int(node.get("end_line", 0) or 0)
    code_hash = str(node.get("code_hash", ""))
    if not code_hash:
        snippet = str(node.get("code_snippet", ""))
        code_hash = hashlib.md5(snippet.encode("utf-8")).hexdigest()[:8]
    return f"{file_path}::{func_name}::{start_line}::{end_line}::{code_hash}"


def _edge_signature(edge: Dict[str, Any]) -> str:
    return f"{edge.get('src', '')}--{edge.get('kind', 'fallthrough')}-->{edge.get('dst', '')}"


def _extract_files(cfg_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "files" in cfg_data:
        return cfg_data["files"]
    if "functions" in cfg_data:
        return {str(cfg_data.get("file_path", "unknown")): cfg_data}
    return {}


def _diff_function(
    func_before: Optional[Dict[str, Any]],
    func_after: Optional[Dict[str, Any]],
    file_path: str,
    func_name: str,
    out: Dict[str, Any],
) -> None:
    nodes_before = (func_before or {}).get("nodes", [])
    nodes_after = (func_after or {}).get("nodes", [])
    edges_before = (func_before or {}).get("edges", [])
    edges_after = (func_after or {}).get("edges", [])

    out["summary"]["total_nodes_before"] += len(nodes_before)
    out["summary"]["total_nodes_after"] += len(nodes_after)
    out["summary"]["total_edges_before"] += len(edges_before)
    out["summary"]["total_edges_after"] += len(edges_after)

    def loc_key(node: Dict[str, Any]) -> Tuple[int, int, str]:
        return (
            int(node.get("start_line", 0) or 0),
            int(node.get("end_line", 0) or 0),
            str(node.get("type", "")),
        )

    before_by_loc = {loc_key(n): n for n in nodes_before}
    after_by_loc = {loc_key(n): n for n in nodes_after}
    before_by_hash = {str(n.get("code_hash", "")): n for n in nodes_before if n.get("code_hash")}
    after_by_hash = {str(n.get("code_hash", "")): n for n in nodes_after if n.get("code_hash")}

    for loc, node in after_by_loc.items():
        if loc in before_by_loc:
            continue
        code_hash = str(node.get("code_hash", ""))
        if code_hash and code_hash in before_by_hash:
            out["nodes_changed"].append(
                {
                    "change_type": "moved",
                    "file": file_path,
                    "function": func_name,
                    "before": before_by_hash[code_hash],
                    "after": node,
                }
            )
        else:
            out["nodes_added"].append({"file": file_path, "function": func_name, "node": node})

    for loc, node in before_by_loc.items():
        if loc in after_by_loc:
            continue
        code_hash = str(node.get("code_hash", ""))
        if code_hash and code_hash in after_by_hash:
            continue
        out["nodes_removed"].append({"file": file_path, "function": func_name, "node": node})

    for loc in set(before_by_loc.keys()) & set(after_by_loc.keys()):
        left = before_by_loc[loc]
        right = after_by_loc[loc]
        if left.get("code_hash") == right.get("code_hash"):
            continue
        out["nodes_changed"].append(
            {
                "change_type": "modified",
                "file": file_path,
                "function": func_name,
                "before": left,
                "after": right,
            }
        )

    before_edge_sigs = {_edge_signature(e) for e in edges_before}
    after_edge_sigs = {_edge_signature(e) for e in edges_after}

    for edge in edges_after:
        sig = _edge_signature(edge)
        if sig not in before_edge_sigs:
            out["edges_added"].append({"file": file_path, "function": func_name, "edge": edge})

    for edge in edges_before:
        sig = _edge_signature(edge)
        if sig not in after_edge_sigs:
            out["edges_removed"].append({"file": file_path, "function": func_name, "edge": edge})


def diff_cfg(cfg_before: Dict[str, Any], cfg_after: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "nodes_added": [],
        "nodes_removed": [],
        "nodes_changed": [],
        "edges_added": [],
        "edges_removed": [],
        "summary": {
            "total_nodes_before": 0,
            "total_nodes_after": 0,
            "total_edges_before": 0,
            "total_edges_after": 0,
            "files_compared": 0,
            "functions_compared": 0,
        },
    }

    files_before = _extract_files(cfg_before)
    files_after = _extract_files(cfg_after)
    all_files = set(files_before.keys()) | set(files_after.keys())
    out["summary"]["files_compared"] = len(all_files)

    for file_path in all_files:
        funcs_before = files_before.get(file_path, {"functions": {}}).get("functions", {})
        funcs_after = files_after.get(file_path, {"functions": {}}).get("functions", {})
        all_funcs = set(funcs_before.keys()) | set(funcs_after.keys())
        out["summary"]["functions_compared"] += len(all_funcs)
        for func_name in all_funcs:
            _diff_function(funcs_before.get(func_name), funcs_after.get(func_name), file_path, func_name, out)

    return out


def get_diff_candidate_nodes(cfg_diff: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for item in cfg_diff.get("nodes_added", []):
        node = item.get("node", {})
        cand = {
            "node_id": node.get("node_id", _node_signature(node, item.get("file"))),
            "change_type": "added",
            "file": item.get("file", ""),
            "function": item.get("function", ""),
            "start_line": node.get("start_line", 0),
            "end_line": node.get("end_line", 0),
            "node_type": node.get("type", "basic_block"),
            "code_snippet": node.get("code_snippet", ""),
            "contains_calls": node.get("contains_calls", []),
        }
        node_id = str(cand["node_id"])
        if node_id in seen:
            continue
        seen.add(node_id)
        nodes.append(cand)

    for item in cfg_diff.get("nodes_changed", []):
        node = item.get("after", {})
        cand = {
            "node_id": node.get("node_id", _node_signature(node, item.get("file"))),
            "change_type": item.get("change_type", "modified"),
            "file": item.get("file", ""),
            "function": item.get("function", ""),
            "start_line": node.get("start_line", 0),
            "end_line": node.get("end_line", 0),
            "node_type": node.get("type", "basic_block"),
            "code_snippet": node.get("code_snippet", ""),
            "contains_calls": node.get("contains_calls", []),
            "before_snippet": item.get("before", {}).get("code_snippet", ""),
        }
        node_id = str(cand["node_id"])
        if node_id in seen:
            continue
        seen.add(node_id)
        nodes.append(cand)

    return nodes


def touched_files_from_patch(patch_text: str) -> List[str]:
    files: Set[str] = set()
    for line in patch_text.splitlines():
        if line.startswith("--- a/"):
            path = line[6:].strip()
            if path and path != "/dev/null":
                files.add(path)
        elif line.startswith("+++ b/"):
            path = line[6:].strip()
            if path and path != "/dev/null":
                files.add(path)
        elif line.startswith("--- "):
            path = line[4:].split("\t", 1)[0].strip()
            if path and path != "/dev/null":
                files.add(path[2:] if path.startswith("a/") else path)
        elif line.startswith("+++ "):
            path = line[4:].split("\t", 1)[0].strip()
            if path and path != "/dev/null":
                files.add(path[2:] if path.startswith("b/") else path)
    return sorted(files)


def _patch_chunks(patch_text: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    current_file = ""
    current_header = ""
    current_lines: List[str] = []
    chunk_id = 0

    def flush() -> None:
        nonlocal chunk_id, current_lines, current_header
        if not current_header:
            return
        chunk_id += 1
        added_lines = [line[1:] for line in current_lines if line.startswith("+") and not line.startswith("+++")]
        chunks.append(
            {
                "chunk_id": f"chunk_{chunk_id}",
                "file_path": current_file,
                "hunk_header": current_header,
                "added_lines": added_lines,
                "raw_hunk": "\n".join(current_lines),
            }
        )
        current_header = ""
        current_lines = []

    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            raw = line[4:].strip()
            current_file = raw[2:] if raw.startswith("b/") else raw
            continue
        if line.startswith("@@"):
            flush()
            current_header = line.strip()
            current_lines = [line]
            continue
        if current_header:
            current_lines.append(line)

    flush()
    return chunks


def create_nodes_from_patch_hunks(patch_text: str) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for chunk in _patch_chunks(patch_text):
        file_path = str(chunk.get("file_path", ""))
        if not file_path.endswith(".py"):
            continue
        added_lines = chunk.get("added_lines", [])
        snippet = "\n".join(added_lines)
        hunk_header = str(chunk.get("hunk_header", ""))
        func_name = "unknown"
        if "def " in hunk_header:
            func_name = hunk_header.split("def ", 1)[-1].split("(", 1)[0].strip() or "unknown"

        start_line = 0
        if hunk_header:
            import re

            match = re.search(r"\+(\d+)", hunk_header)
            if match:
                start_line = int(match.group(1))

        node_id = f"{file_path}::{func_name}::{chunk.get('chunk_id', '')}"
        nodes.append(
            {
                "node_id": node_id,
                "change_type": "added",
                "file": file_path,
                "function": func_name,
                "start_line": start_line,
                "end_line": start_line + len(added_lines),
                "node_type": "basic_block",
                "code_snippet": snippet,
                "contains_calls": [],
            }
        )
    return nodes


def compute_cfg_diff_for_patch(
    patch_text: str,
    *,
    base_repo: Optional[Path],
    allow_hunk_fallback: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Compute CFG diff/candidate nodes for a patch.

    Returns: (cfg_diff, candidate_nodes, diagnostics)
    """
    touched = [path for path in touched_files_from_patch(patch_text) if path.endswith(".py")]
    diagnostics: Dict[str, Any] = {
        "touched_python_files": touched,
        "fallback_used": False,
        "fallback_reason": "",
        "apply_success": None,
        "apply_message": "",
    }

    if not touched:
        return (
            {
                "nodes_added": [],
                "nodes_removed": [],
                "nodes_changed": [],
                "edges_added": [],
                "edges_removed": [],
                "summary": {},
            },
            [],
            diagnostics,
        )

    if base_repo and base_repo.exists():
        work_dir = Path(tempfile.mkdtemp(prefix="cfg_grounding_"))
        patched_repo = work_dir / "patched_repo"
        try:
            shutil.copytree(base_repo, patched_repo)
            apply_ok, apply_msg = apply_unified_diff(patched_repo, patch_text)
            diagnostics["apply_success"] = bool(apply_ok)
            diagnostics["apply_message"] = apply_msg
            if apply_ok:
                cfg_before = build_cfg_for_files(touched, base_path=str(base_repo))
                cfg_after = build_cfg_for_files(touched, base_path=str(patched_repo))
                cfg_diff = diff_cfg(cfg_before, cfg_after)
                candidates = get_diff_candidate_nodes(cfg_diff)
                return cfg_diff, candidates, diagnostics
            diagnostics["fallback_reason"] = f"patch_apply_failed: {apply_msg}"
        except Exception as exc:
            diagnostics["fallback_reason"] = f"cfg_diff_exception: {type(exc).__name__}: {exc}"
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
    else:
        diagnostics["fallback_reason"] = "missing_base_repo"

    if not allow_hunk_fallback:
        return (
            {
                "nodes_added": [],
                "nodes_removed": [],
                "nodes_changed": [],
                "edges_added": [],
                "edges_removed": [],
                "summary": {},
            },
            [],
            diagnostics,
        )

    candidates = create_nodes_from_patch_hunks(patch_text)
    diagnostics["fallback_used"] = True
    cfg_diff = {
        "nodes_added": [{"file": n.get("file", ""), "function": n.get("function", ""), "node": n} for n in candidates],
        "nodes_removed": [],
        "nodes_changed": [],
        "edges_added": [],
        "edges_removed": [],
        "summary": {
            "files_compared": len(touched),
            "functions_compared": 0,
        },
        "candidate_nodes": candidates,
    }
    return cfg_diff, candidates, diagnostics
