"""Scope-aware AST-based CFG patch parser.

Extends the default ``cfg_ast`` parser with module-level and class-level scope
analysis from the cfge project. Candidate nodes now include ``__module__`` and
``Class_*`` scopes, and each node carries a ``scope_depth`` field.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.baseline.structural_misalignment.cfg.diff import (
    diff_cfg,
    get_diff_candidate_nodes,
    create_nodes_from_patch_hunks,
    touched_files_from_patch,
)
from src.baseline.structural_misalignment.cfg.patch_lines import parse_patch_lines
from src.baseline.structural_misalignment.cfg.risk import analyze_node_risk, compute_risk_features
from src.baseline.structural_misalignment.cfg.scope_builder import build_scoped_cfg_for_files
from src.baseline.structural_misalignment.parsers.registry import register_patch_parser
from src.common.diff import apply_unified_diff


def _restore_scope_depth(
    candidates: List[Dict[str, Any]],
    cfg_data: Dict[str, Any],
) -> None:
    """Re-attach ``scope_depth`` to candidates from the scoped CFG data.

    ``get_diff_candidate_nodes`` copies only a fixed set of fields, so
    ``scope_depth`` is lost.  This rebuilds the mapping from node_id to
    scope_depth in the after-CFG and patches it back onto candidates.
    """
    node_id_to_depth: Dict[str, int] = {}
    for file_cfg in cfg_data.get("files", {}).values():
        for func_cfg in file_cfg.get("functions", {}).values():
            depth = func_cfg.get("scope_depth", 1)
            for node in func_cfg.get("nodes", []):
                nid = str(node.get("node_id", ""))
                if nid:
                    node_id_to_depth[nid] = int(node.get("scope_depth", depth))

    for cand in candidates:
        nid = str(cand.get("node_id", ""))
        cand["scope_depth"] = node_id_to_depth.get(nid, -1)


def cfg_ast_scoped_parser(
    patch_text: str,
    *,
    base_repo: Optional[Path],
    allow_hunk_fallback: bool,
    **kwargs: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Parse patch into CFG nodes using scope-aware AST analysis.

    Like ``cfg_ast`` but uses the scoped builder that captures module-level
    and class-level control flow in addition to function-level CFGs.

    The diagnostics dict includes a ``risk_analysis`` key with coverage-based
    risk scoring when coverage data is available (passed via kwargs).

    Args:
        patch_text: Unified diff format patch.
        base_repo: Base repository path for AST analysis.
        allow_hunk_fallback: Whether to use hunk-based fallback on failure.
        **kwargs: May include ``coverage_data`` (Dict[str, Set[int]]) for
            risk analysis.

    Returns:
        Tuple of (cfg_diff, candidate_nodes, diagnostics).
    """
    touched = [p for p in touched_files_from_patch(patch_text) if p.endswith(".py")]
    diagnostics: Dict[str, Any] = {
        "touched_python_files": touched,
        "fallback_used": False,
        "fallback_reason": "",
        "apply_success": None,
        "apply_message": "",
        "scoped_builder": True,
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

    candidates: List[Dict[str, Any]] = []
    cfg_diff: Dict[str, Any] = {}

    if base_repo and base_repo.exists():
        work_dir = Path(tempfile.mkdtemp(prefix="cfg_scoped_"))
        patched_repo = work_dir / "patched_repo"
        try:
            shutil.copytree(base_repo, patched_repo)
            apply_ok, apply_msg = apply_unified_diff(patched_repo, patch_text)
            diagnostics["apply_success"] = bool(apply_ok)
            diagnostics["apply_message"] = apply_msg

            if apply_ok:
                # Use scoped builder for richer CFG extraction
                cfg_before = build_scoped_cfg_for_files(touched, base_path=str(base_repo))
                cfg_after = build_scoped_cfg_for_files(touched, base_path=str(patched_repo))
                cfg_diff = diff_cfg(cfg_before, cfg_after)
                candidates = get_diff_candidate_nodes(cfg_diff)

                # get_diff_candidate_nodes copies a fixed set of fields,
                # dropping scope_depth. Re-attach it from the after-CFG nodes.
                _restore_scope_depth(candidates, cfg_after)

                # Attach scope info from scoped builder
                diagnostics["scoped_stats"] = {
                    "before": cfg_before.get("stats", {}),
                    "after": cfg_after.get("stats", {}),
                }

                # Risk analysis using line-level patch data
                patched_lines = parse_patch_lines(patch_text)
                coverage_data = kwargs.get("coverage_data")
                risk_result = analyze_node_risk(candidates, patched_lines, coverage_data)
                diagnostics["risk_analysis"] = risk_result.get("summary", {})
                diagnostics["risk_features"] = compute_risk_features(risk_result)

                # Annotate candidate nodes with risk scores
                risk_scores = risk_result.get("risk_scores", {})
                for node in candidates:
                    node_id = str(node.get("node_id", ""))
                    node["risk_score"] = risk_scores.get(node_id, 0.0)

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

    # Fallback to hunk-based parsing
    candidates = create_nodes_from_patch_hunks(patch_text)
    diagnostics["fallback_used"] = True

    # Still do risk analysis on fallback nodes
    patched_lines = parse_patch_lines(patch_text)
    coverage_data = kwargs.get("coverage_data")
    risk_result = analyze_node_risk(candidates, patched_lines, coverage_data)
    diagnostics["risk_analysis"] = risk_result.get("summary", {})
    diagnostics["risk_features"] = compute_risk_features(risk_result)

    for node in candidates:
        node_id = str(node.get("node_id", ""))
        node["risk_score"] = risk_result.get("risk_scores", {}).get(node_id, 0.0)

    cfg_diff = {
        "nodes_added": [
            {"file": n.get("file", ""), "function": n.get("function", ""), "node": n}
            for n in candidates
        ],
        "nodes_removed": [],
        "nodes_changed": [],
        "edges_added": [],
        "edges_removed": [],
        "summary": {
            "files_compared": len(touched),
            "functions_compared": 0,
        },
    }
    return cfg_diff, candidates, diagnostics


register_patch_parser("cfg_ast_scoped")(cfg_ast_scoped_parser)
