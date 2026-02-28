"""Numeric CFG statistics for defense/judge reporting."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


def compute_cfg_stats(candidate_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidate_nodes:
        return {
            "num_candidate_nodes": 0,
            "num_files_touched": 0,
            "num_functions_touched": 0,
            "node_type_counts": {},
            "change_type_counts": {},
            "avg_snippet_len": 0.0,
        }

    files = {str(n.get("file", "")) for n in candidate_nodes if n.get("file")}
    funcs = {
        f"{n.get('file', '')}::{n.get('function', '')}"
        for n in candidate_nodes
        if n.get("file") or n.get("function")
    }
    node_type_counts = Counter(str(n.get("node_type", "basic_block")) for n in candidate_nodes)
    change_type_counts = Counter(str(n.get("change_type", "modified")) for n in candidate_nodes)
    lengths = [len(str(n.get("code_snippet", ""))) for n in candidate_nodes]

    return {
        "num_candidate_nodes": len(candidate_nodes),
        "num_files_touched": len(files),
        "num_functions_touched": len(funcs),
        "node_type_counts": dict(node_type_counts),
        "change_type_counts": dict(change_type_counts),
        "avg_snippet_len": (sum(lengths) / len(lengths)) if lengths else 0.0,
    }
