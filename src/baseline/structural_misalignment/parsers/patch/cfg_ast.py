"""AST-based CFG extraction (default patch parser)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.baseline.structural_misalignment.cfg.diff import compute_cfg_diff_for_patch as _compute_cfg_diff_impl
from src.baseline.structural_misalignment.parsers.registry import register_patch_parser


def cfg_ast_parser(
    patch_text: str,
    *,
    base_repo: Optional[Path],
    allow_hunk_fallback: bool,
    **kwargs: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Parse patch into CFG nodes using AST analysis.
    
    This is the default patch parser that:
    1. Applies patch to repository
    2. Builds CFG from AST for before/after versions
    3. Diffs CFGs to find changed nodes
    4. Falls back to hunk-based parsing if AST fails
    
    Args:
        patch_text: Unified diff format patch
        base_repo: Base repository path for AST analysis
        allow_hunk_fallback: Whether to use hunk-based fallback on failure
        **kwargs: Additional arguments (ignored for extensibility)
    
    Returns:
        Tuple of (cfg_diff, candidate_nodes, diagnostics)
        - cfg_diff: Dict with nodes_added, nodes_removed, nodes_changed, edges_*, summary
        - candidate_nodes: List of dicts with node_id, change_type, file, function, etc.
        - diagnostics: Dict with touched_python_files, fallback_used, etc.
    """
    return _compute_cfg_diff_impl(
        patch_text,
        base_repo=base_repo,
        allow_hunk_fallback=allow_hunk_fallback,
    )


# Register as default patch parser
register_patch_parser("cfg_ast")(cfg_ast_parser)
