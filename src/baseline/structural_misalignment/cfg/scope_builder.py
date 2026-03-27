"""Scope-aware CFG builder that extends the base builder with module/class-level analysis.

Ported from the cfge project (https://github.com/river/cfge). The base CFGBuilder only
extracts function-level CFGs. This module adds:
  - Module-level control flow (top-level if/for/while/try blocks)
  - Class-level scope tracking (class body statements, decorators)
  - Scope depth metadata on each node for downstream filtering

The ``build_scoped_cfg_for_file`` entry point returns the same schema as
``build_cfg_for_file`` but with additional ``__module__`` and ``Class_*`` entries
in the ``functions`` dict, plus a ``scope_depth`` field on each node.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.baseline.structural_misalignment.cfg.build import (
    CFGBuilder,
    _called_functions,
    _hash_code,
    _source_segment,
)


class ScopedCFGBuilder(CFGBuilder):
    """CFGBuilder subclass that tracks scope depth and handles module/class scopes."""

    def __init__(self, source_lines: List[str], function_name: str, file_path: str) -> None:
        super().__init__(source_lines, function_name, file_path)
        self.scope_depth: int = 0

    def _create_node(
        self,
        node_type: str,
        start_line: int,
        end_line: int,
        stmts: Optional[List[ast.stmt]] = None,
    ) -> Dict[str, Any]:
        node = super()._create_node(node_type, start_line, end_line, stmts)
        node["scope_depth"] = self.scope_depth
        return node


def _build_scope_cfg(
    source_lines: List[str],
    scope_name: str,
    file_path: str,
    body: List[ast.stmt],
    scope_depth: int,
    start_line: int,
    end_line: int,
) -> Dict[str, Any]:
    """Build a CFG for a scope (module body, class body, or function body)."""
    builder = ScopedCFGBuilder(source_lines, scope_name, file_path)
    builder.scope_depth = scope_depth

    entry = builder._create_node("entry", start_line, start_line)
    tail_nodes = builder._process_body(body, [entry])
    exit_node = builder._create_node("exit", end_line, end_line)
    for node in tail_nodes:
        builder._add_edge(node["node_id"], exit_node["node_id"])

    return {
        "function_name": scope_name,
        "start_line": start_line,
        "end_line": end_line,
        "scope_depth": scope_depth,
        "nodes": builder.nodes,
        "edges": builder.edges,
    }


def build_scoped_cfg_for_file(py_path: str) -> Dict[str, Any]:
    """Build CFGs for all scopes in a Python file.

    Returns the same schema as ``build_cfg_for_file`` but with additional
    ``__module__`` and ``Class_*`` entries in the ``functions`` dict.
    Each node includes a ``scope_depth`` field (0 = module, 1+ = nested).
    """
    result: Dict[str, Any] = {
        "file_path": py_path,
        "functions": {},
        "parse_error": None,
    }

    try:
        source = Path(py_path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        result["parse_error"] = f"File not found: {py_path}"
        return result
    except Exception as exc:
        result["parse_error"] = f"Error: {type(exc).__name__}: {exc}"
        return result

    source_lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=py_path)
    except SyntaxError as exc:
        result["parse_error"] = f"SyntaxError: {exc}"
        return result

    # --- Module-level scope (depth 0) ---
    module_body = [
        stmt for stmt in tree.body
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if module_body:
        end_line = max(getattr(s, "end_lineno", s.lineno) for s in module_body)
        result["functions"]["__module__"] = _build_scope_cfg(
            source_lines, "__module__", py_path, module_body,
            scope_depth=0, start_line=1, end_line=end_line,
        )

    # --- Class-level scopes (depth 1) + their methods ---
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = f"Class_{node.name}"
            class_end = getattr(node, "end_lineno", node.lineno)
            # Class body excluding methods (class-level statements)
            class_body = [
                stmt for stmt in node.body
                if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if class_body:
                result["functions"][class_name] = _build_scope_cfg(
                    source_lines, class_name, py_path, class_body,
                    scope_depth=1, start_line=node.lineno, end_line=class_end,
                )

    # --- Function-level scopes (depth 1+) ---
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        func_name = node.name
        unique_name = func_name
        if unique_name in result["functions"]:
            unique_name = f"{func_name}_L{node.lineno}"

        end_line = getattr(node, "end_lineno", node.lineno)
        builder = ScopedCFGBuilder(source_lines, unique_name, py_path)
        builder.scope_depth = 1  # functions are at least depth 1
        cfg = builder.build(node)
        # Add scope_depth to all nodes
        for n in cfg.get("nodes", []):
            if "scope_depth" not in n:
                n["scope_depth"] = 1
        cfg["scope_depth"] = 1
        result["functions"][unique_name] = cfg

    return result


def build_scoped_cfg_for_files(
    file_paths: List[str],
    base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build scope-aware CFGs for multiple files."""
    result: Dict[str, Any] = {
        "base_path": base_path,
        "files": {},
        "stats": {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_functions": 0,
            "total_scopes": 0,
        },
    }

    for file_path in file_paths:
        full_path = (
            Path(base_path) / file_path
            if base_path and not Path(file_path).is_absolute()
            else Path(file_path)
        )
        if not full_path.exists():
            result["files"][file_path] = {
                "file_path": file_path,
                "functions": {},
                "parse_error": f"File not found: {full_path}",
            }
            result["stats"]["total_files"] += 1
            result["stats"]["failed_files"] += 1
            continue

        file_cfg = build_scoped_cfg_for_file(str(full_path))
        # Normalize file paths in node IDs
        if not file_cfg.get("parse_error"):
            for func_cfg in file_cfg.get("functions", {}).values():
                for node in func_cfg.get("nodes", []):
                    node_id = str(node.get("node_id", ""))
                    if node_id.startswith(str(full_path)):
                        node["node_id"] = node_id.replace(str(full_path), file_path, 1)
            file_cfg["file_path"] = file_path

        result["files"][file_path] = file_cfg
        result["stats"]["total_files"] += 1
        if file_cfg["parse_error"]:
            result["stats"]["failed_files"] += 1
        else:
            result["stats"]["successful_files"] += 1
            funcs = file_cfg.get("functions", {})
            result["stats"]["total_functions"] += sum(
                1 for k in funcs if not k.startswith("__module__") and not k.startswith("Class_")
            )
            result["stats"]["total_scopes"] += len(funcs)

    return result
