"""CFG diff computation and patch-grounded candidate node extraction."""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.baseline.structural_misalignment.cfg.build import build_cfg_for_files
from src.common.diff import apply_unified_diff
from src.common.subprocess import run_command


RangeMap = Dict[str, List[Tuple[int, int]]]


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


def _node_range(node: Dict[str, Any]) -> Tuple[int, int]:
    start = int(node.get("start_line", 0) or 0)
    end = int(node.get("end_line", start) or start)
    if end < start:
        end = start
    return start, end


def _ranges_overlap(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    left_start, left_end = left
    right_start, right_end = right
    if left_start <= 0 or right_start <= 0:
        return False
    return left_start <= right_end and right_start <= left_end


def _node_overlaps_changed_ranges(node: Dict[str, Any], file_path: str, changed_ranges: RangeMap) -> bool:
    ranges = changed_ranges.get(file_path, [])
    if not ranges:
        return False
    node_range = _node_range(node)
    return any(_ranges_overlap(node_range, changed_range) for changed_range in ranges)


def _filter_cfg_diff_to_changed_ranges(cfg_diff: Dict[str, Any], changed_ranges: RangeMap) -> Dict[str, Any]:
    filtered = dict(cfg_diff)
    filtered["nodes_added"] = [
        item
        for item in cfg_diff.get("nodes_added", [])
        if _node_overlaps_changed_ranges(item.get("node", {}), str(item.get("file", "")), changed_ranges)
    ]
    filtered["nodes_changed"] = [
        item
        for item in cfg_diff.get("nodes_changed", [])
        if _node_overlaps_changed_ranges(item.get("after", {}), str(item.get("file", "")), changed_ranges)
    ]
    return filtered


def get_candidate_code_edges(cfg_after: Dict[str, Any], candidate_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidate_ids = {str(node.get("node_id", "")) for node in candidate_nodes if node.get("node_id")}
    if not candidate_ids:
        return []

    edges: List[Dict[str, Any]] = []
    for file_cfg in _extract_files(cfg_after).values():
        for func_cfg in file_cfg.get("functions", {}).values():
            for edge in func_cfg.get("edges", []):
                src = str(edge.get("src", ""))
                dst = str(edge.get("dst", ""))
                if src not in candidate_ids or dst not in candidate_ids:
                    continue
                edges.append(
                    {
                        "src": src,
                        "dst": dst,
                        "kind": str(edge.get("kind", "fallthrough")),
                    }
                )
    return edges


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


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))?")


def _parse_changed_ranges_from_diff(diff_text: str) -> RangeMap:
    ranges: RangeMap = {}
    current_file = ""
    new_line = 0
    current_start: Optional[int] = None
    current_end = 0

    def flush_run() -> None:
        nonlocal current_start, current_end
        if current_file and current_start is not None:
            ranges.setdefault(current_file, []).append((current_start, current_end))
        current_start = None
        current_end = 0

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            flush_run()
            raw = line[4:].split("\t", 1)[0].strip()
            current_file = raw[2:] if raw.startswith("b/") else raw
            if current_file == "/dev/null":
                current_file = ""
            continue
        if line.startswith("@@"):
            flush_run()
            match = _HUNK_RE.match(line)
            if not match:
                new_line = 0
                continue
            new_line = int(match.group(3))
            continue
        if not current_file or not line:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            if current_start is None:
                current_start = new_line
            current_end = new_line
            new_line += 1
            continue
        flush_run()
        if line.startswith(" ") or line.startswith("-"):
            if not line.startswith("-"):
                new_line += 1

    flush_run()
    return ranges


def _normalized_changed_ranges(patched_repo: Path, touched: List[str]) -> Tuple[RangeMap, str]:
    command = ["git", "diff", "--unified=0", "--", *touched]
    result = run_command(command, cwd=patched_repo)
    if result.returncode != 0:
        return {}, (result.stderr or result.stdout or "").strip()
    return _parse_changed_ranges_from_diff(result.stdout or ""), ""


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


def _fallback_runs_from_chunk(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    hunk_header = str(chunk.get("hunk_header", ""))
    match = _HUNK_RE.match(hunk_header)
    if not match:
        return []
    new_line = int(match.group(3))
    current_start: Optional[int] = None
    current_lines: List[str] = []
    runs: List[Dict[str, Any]] = []

    def flush_run() -> None:
        nonlocal current_start, current_lines
        if current_start is None:
            return
        runs.append(
            {
                "start_line": current_start,
                "end_line": current_start + max(0, len(current_lines) - 1),
                "added_lines": list(current_lines),
            }
        )
        current_start = None
        current_lines = []

    for line in str(chunk.get("raw_hunk", "")).splitlines()[1:]:
        if line.startswith("+") and not line.startswith("+++"):
            if current_start is None:
                current_start = new_line
            current_lines.append(line[1:])
            new_line += 1
            continue
        flush_run()
        if line.startswith(" ") or line == r"\ No newline at end of file":
            if line.startswith(" "):
                new_line += 1
            continue
        if line.startswith("-") and not line.startswith("---"):
            continue

    flush_run()
    return runs


def create_nodes_from_patch_hunks(patch_text: str) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    run_id = 0
    for chunk in _patch_chunks(patch_text):
        file_path = str(chunk.get("file_path", ""))
        if not file_path.endswith(".py"):
            continue
        hunk_header = str(chunk.get("hunk_header", ""))
        func_name = "unknown"
        if "def " in hunk_header:
            func_name = hunk_header.split("def ", 1)[-1].split("(", 1)[0].strip() or "unknown"

        for run in _fallback_runs_from_chunk(chunk):
            added_lines = run.get("added_lines", [])
            snippet = "\n".join(added_lines)
            if not snippet.strip():
                continue
            run_id += 1
            node_id = f"{file_path}::{func_name}::{chunk.get('chunk_id', '')}_run_{run_id}"
            nodes.append(
                {
                    "node_id": node_id,
                    "change_type": "added",
                    "file": file_path,
                    "function": func_name,
                    "start_line": run.get("start_line", 0),
                    "end_line": run.get("end_line", run.get("start_line", 0)),
                    "node_type": "basic_block",
                    "code_snippet": snippet,
                    "contains_calls": [],
                }
            )
    return nodes


def _empty_cfg_diff(summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "nodes_added": [],
        "nodes_removed": [],
        "nodes_changed": [],
        "edges_added": [],
        "edges_removed": [],
        "summary": summary or {},
    }


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
        "hunk_count": len(_patch_chunks(patch_text)),
        "changed_range_count": 0,
        "raw_candidate_node_count": 0,
        "filtered_candidate_node_count": 0,
    }

    if not touched:
        return (_empty_cfg_diff(), [], diagnostics)

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
                raw_candidates = get_diff_candidate_nodes(cfg_diff)
                diagnostics["raw_candidate_node_count"] = len(raw_candidates)
                changed_ranges, range_error = _normalized_changed_ranges(patched_repo, touched)
                diagnostics["changed_ranges"] = {
                    file_path: [[start, end] for start, end in ranges]
                    for file_path, ranges in changed_ranges.items()
                }
                diagnostics["changed_range_count"] = sum(len(ranges) for ranges in changed_ranges.values())
                if range_error:
                    diagnostics["changed_range_error"] = range_error
                filtered_cfg_diff = _filter_cfg_diff_to_changed_ranges(cfg_diff, changed_ranges)
                candidates = get_diff_candidate_nodes(filtered_cfg_diff)
                diagnostics["filtered_candidate_node_count"] = len(candidates)
                filtered_cfg_diff.setdefault("summary", {})
                filtered_cfg_diff["summary"].update(
                    {
                        "raw_candidate_node_count": diagnostics["raw_candidate_node_count"],
                        "filtered_candidate_node_count": diagnostics["filtered_candidate_node_count"],
                        "hunk_count": diagnostics["hunk_count"],
                        "changed_range_count": diagnostics["changed_range_count"],
                    }
                )
                candidate_edges = get_candidate_code_edges(cfg_after, candidates)
                filtered_cfg_diff["candidate_edges"] = candidate_edges
                diagnostics["candidate_code_edges"] = candidate_edges
                return filtered_cfg_diff, candidates, diagnostics
            diagnostics["fallback_reason"] = f"patch_apply_failed: {apply_msg}"
        except Exception as exc:
            diagnostics["fallback_reason"] = f"cfg_diff_exception: {type(exc).__name__}: {exc}"
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
    else:
        diagnostics["fallback_reason"] = "missing_base_repo"

    if not allow_hunk_fallback:
        return (_empty_cfg_diff(), [], diagnostics)

    candidates = create_nodes_from_patch_hunks(patch_text)
    diagnostics["fallback_used"] = True
    diagnostics["raw_candidate_node_count"] = len(candidates)
    diagnostics["filtered_candidate_node_count"] = len(candidates)
    fallback_ranges = _parse_changed_ranges_from_diff(patch_text)
    diagnostics["changed_ranges"] = {
        file_path: [[start, end] for start, end in ranges]
        for file_path, ranges in fallback_ranges.items()
    }
    diagnostics["changed_range_count"] = sum(len(ranges) for ranges in fallback_ranges.values())
    cfg_diff = {
        "nodes_added": [{"file": n.get("file", ""), "function": n.get("function", ""), "node": n} for n in candidates],
        "nodes_removed": [],
        "nodes_changed": [],
        "edges_added": [],
        "edges_removed": [],
        "candidate_edges": [],
        "summary": {
            "files_compared": len(touched),
            "functions_compared": 0,
            "raw_candidate_node_count": diagnostics["raw_candidate_node_count"],
            "filtered_candidate_node_count": diagnostics["filtered_candidate_node_count"],
            "hunk_count": diagnostics["hunk_count"],
            "changed_range_count": diagnostics["changed_range_count"],
        },
        "candidate_nodes": candidates,
    }
    diagnostics["candidate_code_edges"] = []
    return cfg_diff, candidates, diagnostics
