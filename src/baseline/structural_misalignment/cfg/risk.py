"""Risk analysis for CFG nodes based on patch impact and test coverage.

Ported from the cfge project. This module identifies which candidate CFG nodes
are *patched but not covered by tests* (high-risk) vs. *patched and covered*
(safe). This signal can be used as additional features for the structural
misalignment classifier or for defense prioritization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple


def analyze_node_risk(
    candidate_nodes: List[Dict[str, Any]],
    patched_lines_by_file: Dict[str, Set[int]],
    covered_lines_by_file: Dict[str, Set[int]] | None = None,
) -> Dict[str, Any]:
    """Score each candidate node's risk based on patch + coverage overlap.

    Args:
        candidate_nodes: Candidate nodes from CFG diff (standard schema with
            ``file``, ``start_line``, ``end_line``, ``node_id``, etc.).
        patched_lines_by_file: Output of ``parse_patch_lines`` -- maps file
            path to set of added/modified line numbers.
        covered_lines_by_file: Optional mapping of file path to set of line
            numbers covered by tests (e.g., from pytest-cov JSON). If *None*,
            all patched lines are treated as uncovered (worst-case).

    Returns:
        Dict with:
        - ``high_risk``: list of nodes that are patched but NOT covered
        - ``safe``: list of nodes that are patched AND covered
        - ``unpatched``: list of nodes with no patch overlap
        - ``risk_scores``: dict mapping node_id to a float risk score (0-1)
        - ``summary``: aggregate statistics
    """
    if covered_lines_by_file is None:
        covered_lines_by_file = {}

    high_risk: List[Dict[str, Any]] = []
    safe: List[Dict[str, Any]] = []
    unpatched: List[Dict[str, Any]] = []
    risk_scores: Dict[str, float] = {}

    for node in candidate_nodes:
        file_path = str(node.get("file", ""))
        start = int(node.get("start_line", 0) or 0)
        end = int(node.get("end_line", 0) or 0)
        node_id = str(node.get("node_id", ""))

        if start <= 0 or end <= 0:
            unpatched.append(node)
            risk_scores[node_id] = 0.0
            continue

        node_lines = set(range(start, end + 1))
        patched = patched_lines_by_file.get(file_path, set())
        covered = covered_lines_by_file.get(file_path, set())

        lines_in_patch = node_lines & patched
        lines_covered = node_lines & covered

        if not lines_in_patch:
            unpatched.append(node)
            risk_scores[node_id] = 0.0
            continue

        uncovered_patch_lines = lines_in_patch - covered
        total_node_lines = max(len(node_lines), 1)

        # Risk score: fraction of node lines that are patched but uncovered
        risk = len(uncovered_patch_lines) / total_node_lines
        risk_scores[node_id] = float(risk)

        info = {
            **node,
            "patched_line_count": len(lines_in_patch),
            "covered_line_count": len(lines_covered),
            "uncovered_patch_line_count": len(uncovered_patch_lines),
            "risk_score": float(risk),
        }

        if uncovered_patch_lines:
            high_risk.append(info)
        else:
            safe.append(info)

    total = len(candidate_nodes)
    return {
        "high_risk": high_risk,
        "safe": safe,
        "unpatched": unpatched,
        "risk_scores": risk_scores,
        "summary": {
            "total_nodes": total,
            "high_risk_count": len(high_risk),
            "safe_count": len(safe),
            "unpatched_count": len(unpatched),
            "high_risk_ratio": len(high_risk) / max(total, 1),
            "coverage_available": bool(covered_lines_by_file),
        },
    }


def compute_risk_features(risk_result: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric features from risk analysis for ML pipeline integration.

    Returns feature dict compatible with the structural features schema.
    """
    summary = risk_result.get("summary", {})
    total = max(int(summary.get("total_nodes", 0)), 1)
    high_risk = risk_result.get("high_risk", [])
    safe = risk_result.get("safe", [])

    # Aggregate risk scores
    all_scores = list(risk_result.get("risk_scores", {}).values())
    avg_risk = sum(all_scores) / max(len(all_scores), 1) if all_scores else 0.0
    max_risk = max(all_scores) if all_scores else 0.0

    # Patch density across nodes
    patched_nodes = len(high_risk) + len(safe)
    patch_density = patched_nodes / total

    return {
        "risk_high_risk_ratio": float(summary.get("high_risk_ratio", 0.0)),
        "risk_safe_ratio": float(len(safe)) / total,
        "risk_avg_score": float(avg_risk),
        "risk_max_score": float(max_risk),
        "risk_patch_density": float(patch_density),
        "risk_uncovered_patch_lines": float(
            sum(n.get("uncovered_patch_line_count", 0) for n in high_risk)
        ),
    }
