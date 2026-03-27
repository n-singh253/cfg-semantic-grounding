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
            coverage is treated as unavailable and all patched nodes get
            neutral risk scores (not worst-case).

    Returns:
        Dict with:
        - ``high_risk``: list of nodes that are patched but NOT covered
        - ``safe``: list of nodes that are patched AND covered
        - ``coverage_unknown``: list of patched nodes when coverage is unavailable
        - ``unpatched``: list of nodes with no patch overlap
        - ``risk_scores``: dict mapping node_id to a float risk score (0-1)
        - ``summary``: aggregate statistics
    """
    coverage_available = covered_lines_by_file is not None
    if not coverage_available:
        covered_lines_by_file = {}

    high_risk: List[Dict[str, Any]] = []
    safe: List[Dict[str, Any]] = []
    coverage_unknown: List[Dict[str, Any]] = []
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

        lines_in_patch = node_lines & patched

        if not lines_in_patch:
            unpatched.append(node)
            risk_scores[node_id] = 0.0
            continue

        # When coverage data is not available, classify patched nodes as
        # "coverage_unknown" with neutral scores instead of assuming worst-case.
        if not coverage_available:
            risk_scores[node_id] = 0.0
            coverage_unknown.append({
                **node,
                "patched_line_count": len(lines_in_patch),
                "covered_line_count": 0,
                "uncovered_patch_line_count": 0,
                "risk_score": 0.0,
            })
            continue

        covered = covered_lines_by_file.get(file_path, set())
        lines_covered = node_lines & covered
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
        "coverage_unknown": coverage_unknown,
        "unpatched": unpatched,
        "risk_scores": risk_scores,
        "summary": {
            "total_nodes": total,
            "high_risk_count": len(high_risk),
            "safe_count": len(safe),
            "coverage_unknown_count": len(coverage_unknown),
            "unpatched_count": len(unpatched),
            "high_risk_ratio": len(high_risk) / max(total, 1),
            "coverage_available": coverage_available,
        },
    }


def compute_risk_features(risk_result: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric features from risk analysis for ML pipeline integration.

    When coverage data was unavailable, all risk features are 0.0 (neutral)
    and ``risk_coverage_available`` is 0.0 so downstream consumers can
    distinguish "no risk detected" from "risk unknown".

    Returns feature dict compatible with the structural features schema.
    """
    summary = risk_result.get("summary", {})
    coverage_available = bool(summary.get("coverage_available", False))
    total = max(int(summary.get("total_nodes", 0)), 1)
    high_risk = risk_result.get("high_risk", [])
    safe = risk_result.get("safe", [])
    coverage_unknown = risk_result.get("coverage_unknown", [])

    if not coverage_available:
        # Neutral features — coverage was not provided, so risk is unknown.
        patched_nodes = len(coverage_unknown) + len(high_risk) + len(safe)
        return {
            "risk_high_risk_ratio": 0.0,
            "risk_safe_ratio": 0.0,
            "risk_avg_score": 0.0,
            "risk_max_score": 0.0,
            "risk_patch_density": float(patched_nodes) / total,
            "risk_uncovered_patch_lines": 0.0,
            "risk_coverage_available": 0.0,
        }

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
        "risk_coverage_available": 1.0,
    }
