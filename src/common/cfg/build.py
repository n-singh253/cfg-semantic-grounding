"""Approximate CFG extraction for Python files using AST."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional


SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".tox",
    ".eggs",
    "build",
    "dist",
    ".venv",
    "venv",
    "node_modules",
}


def _hash_code(code: str) -> str:
    return hashlib.md5(code.encode("utf-8")).hexdigest()[:8]


def _source_segment(lines: List[str], start_line: int, end_line: int, max_chars: int = 500) -> str:
    if start_line < 1 or end_line < start_line:
        return ""
    code = "\n".join(lines[start_line - 1 : end_line])
    if len(code) > max_chars:
        return code[:max_chars] + "..."
    return code


def _called_functions(node: ast.AST) -> List[str]:
    calls: List[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            calls.append(child.func.id)
            continue
        if isinstance(child.func, ast.Attribute):
            parts: List[str] = []
            cur = child.func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            calls.append(".".join(reversed(parts)))
    return calls


class CFGBuilder:
    """Build a lightweight CFG for one function."""

    def __init__(self, source_lines: List[str], function_name: str, file_path: str) -> None:
        self.source_lines = source_lines
        self.function_name = function_name
        self.file_path = file_path
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, str]] = []
        self.node_counter = 0

    def _node_id(self) -> str:
        node_id = f"{self.file_path}::{self.function_name}::n{self.node_counter}"
        self.node_counter += 1
        return node_id

    def _add_edge(self, src_id: str, dst_id: str, kind: str = "fallthrough") -> None:
        self.edges.append({"src": src_id, "dst": dst_id, "kind": kind})

    def _create_node(
        self,
        node_type: str,
        start_line: int,
        end_line: int,
        stmts: Optional[List[ast.stmt]] = None,
    ) -> Dict[str, Any]:
        snippet = _source_segment(self.source_lines, start_line, end_line)
        calls: List[str] = []
        if stmts:
            for stmt in stmts:
                calls.extend(_called_functions(stmt))
        node = {
            "node_id": self._node_id(),
            "type": node_type,
            "start_line": start_line,
            "end_line": end_line,
            "code_snippet": snippet,
            "code_hash": _hash_code(snippet),
            "contains_calls": sorted(set(calls)),
        }
        self.nodes.append(node)
        return node

    @staticmethod
    def _is_control(stmt: ast.stmt) -> bool:
        return isinstance(
            stmt,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.Try,
                ast.Return,
                ast.Raise,
                ast.Break,
                ast.Continue,
                ast.With,
                ast.AsyncFor,
                ast.AsyncWith,
            ),
        )

    def _create_block(self, stmts: List[ast.stmt]) -> Dict[str, Any]:
        start = stmts[0].lineno
        end = max(getattr(s, "end_lineno", s.lineno) for s in stmts)
        return self._create_node("basic_block", start, end, stmts)

    def _process_body(self, stmts: List[ast.stmt], entry_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not stmts:
            return entry_nodes

        current_entries = entry_nodes
        pending: List[ast.stmt] = []

        for stmt in stmts:
            if self._is_control(stmt):
                if pending:
                    block = self._create_block(pending)
                    for entry in current_entries:
                        self._add_edge(entry["node_id"], block["node_id"])
                    current_entries = [block]
                    pending = []
                current_entries = self._process_control(stmt, current_entries)
            else:
                pending.append(stmt)

        if pending:
            block = self._create_block(pending)
            for entry in current_entries:
                self._add_edge(entry["node_id"], block["node_id"])
            current_entries = [block]

        return current_entries

    def _process_control(self, stmt: ast.stmt, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if isinstance(stmt, ast.If):
            return self._process_if(stmt, entries)
        if isinstance(stmt, (ast.For, ast.While)):
            return self._process_loop(stmt, entries)
        if isinstance(stmt, ast.Try):
            return self._process_try(stmt, entries)
        if isinstance(stmt, ast.Return):
            return self._process_terminal(stmt, entries, node_type="return")
        if isinstance(stmt, ast.Raise):
            return self._process_terminal(stmt, entries, node_type="raise")
        if isinstance(stmt, (ast.With, ast.AsyncWith, ast.AsyncFor)):
            return self._process_with(stmt, entries)

        # break/continue fallback block
        block = self._create_node(
            "basic_block",
            stmt.lineno,
            getattr(stmt, "end_lineno", stmt.lineno),
            [stmt],
        )
        for entry in entries:
            self._add_edge(entry["node_id"], block["node_id"])
        return [block]

    def _process_if(self, stmt: ast.If, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cond = self._create_node("condition", stmt.lineno, stmt.lineno, [stmt])
        for entry in entries:
            self._add_edge(entry["node_id"], cond["node_id"])

        exit_nodes: List[Dict[str, Any]] = []
        if stmt.body:
            true_exits = self._process_body(stmt.body, [cond])
            for edge in reversed(self.edges):
                if edge["src"] == cond["node_id"]:
                    edge["kind"] = "branch_true"
                    break
            exit_nodes.extend(true_exits)

        if stmt.orelse:
            false_exits = self._process_body(stmt.orelse, [cond])
            for edge in reversed(self.edges):
                if edge["src"] == cond["node_id"] and edge["kind"] == "fallthrough":
                    edge["kind"] = "branch_false"
                    break
            exit_nodes.extend(false_exits)
        else:
            exit_nodes.append(cond)

        return exit_nodes

    def _process_loop(self, stmt: ast.stmt, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        header = self._create_node("loop_header", stmt.lineno, stmt.lineno, [stmt])
        for entry in entries:
            self._add_edge(entry["node_id"], header["node_id"])

        if getattr(stmt, "body", None):
            body_exits = self._process_body(stmt.body, [header])
            for exit_node in body_exits:
                self._add_edge(exit_node["node_id"], header["node_id"], "back_edge")

        exits: List[Dict[str, Any]] = [header]
        if getattr(stmt, "orelse", None):
            exits.extend(self._process_body(stmt.orelse, [header]))
        return exits

    def _process_try(self, stmt: ast.Try, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        header = self._create_node("try_header", stmt.lineno, stmt.lineno)
        for entry in entries:
            self._add_edge(entry["node_id"], header["node_id"])

        exits: List[Dict[str, Any]] = []
        try_exits: List[Dict[str, Any]] = []
        if stmt.body:
            try_exits = self._process_body(stmt.body, [header])
            exits.extend(try_exits)

        for handler in stmt.handlers:
            handler_node = self._create_node(
                "except_handler",
                handler.lineno,
                getattr(handler, "end_lineno", handler.lineno),
            )
            self._add_edge(header["node_id"], handler_node["node_id"], "exception")
            if handler.body:
                exits.extend(self._process_body(handler.body, [handler_node]))

        if stmt.orelse:
            else_exits = self._process_body(stmt.orelse, try_exits if try_exits else [header])
            exits = [node for node in exits if node not in try_exits]
            exits.extend(else_exits)

        if stmt.finalbody:
            finally_header = self._create_node(
                "finally",
                stmt.finalbody[0].lineno,
                getattr(stmt.finalbody[-1], "end_lineno", stmt.finalbody[-1].lineno),
            )
            for exit_node in exits:
                self._add_edge(exit_node["node_id"], finally_header["node_id"], "finally")
            exits = self._process_body(stmt.finalbody, [finally_header])

        return exits

    def _process_terminal(
        self,
        stmt: ast.stmt,
        entries: List[Dict[str, Any]],
        *,
        node_type: str,
    ) -> List[Dict[str, Any]]:
        node = self._create_node(node_type, stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno), [stmt])
        for entry in entries:
            self._add_edge(entry["node_id"], node["node_id"])
        return []

    def _process_with(self, stmt: ast.stmt, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        header = self._create_node("with_header", stmt.lineno, stmt.lineno, [stmt])
        for entry in entries:
            self._add_edge(entry["node_id"], header["node_id"])
        if getattr(stmt, "body", None):
            return self._process_body(stmt.body, [header])
        return [header]

    def build(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        entry = self._create_node("entry", func_node.lineno, func_node.lineno)
        tail_nodes = self._process_body(func_node.body, [entry])
        end_line = getattr(func_node, "end_lineno", func_node.lineno)
        exit_node = self._create_node("exit", end_line, end_line)
        for node in tail_nodes:
            self._add_edge(node["node_id"], exit_node["node_id"])
        return {
            "function_name": self.function_name,
            "start_line": func_node.lineno,
            "end_line": end_line,
            "nodes": self.nodes,
            "edges": self.edges,
        }


def build_cfg_for_file(py_path: str) -> Dict[str, Any]:
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

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        func_name = node.name
        unique_name = func_name
        if unique_name in result["functions"]:
            unique_name = f"{func_name}_L{node.lineno}"
        builder = CFGBuilder(source_lines, unique_name, py_path)
        result["functions"][unique_name] = builder.build(node)

    return result


def build_cfg_for_files(file_paths: List[str], base_path: Optional[str] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "base_path": base_path,
        "files": {},
        "stats": {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_functions": 0,
        },
    }

    for file_path in file_paths:
        full_path = Path(base_path) / file_path if base_path and not Path(file_path).is_absolute() else Path(file_path)
        if not full_path.exists():
            result["files"][file_path] = {
                "file_path": file_path,
                "functions": {},
                "parse_error": f"File not found: {full_path}",
            }
            result["stats"]["total_files"] += 1
            result["stats"]["failed_files"] += 1
            continue

        file_cfg = build_cfg_for_file(str(full_path))
        # Normalize file paths in node IDs to the requested relative path.
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
            result["stats"]["total_functions"] += len(file_cfg["functions"])

    return result


def build_cfg_for_repo(repo_root: str) -> Dict[str, Any]:
    root = Path(repo_root).resolve()
    files = [
        path
        for path in root.rglob("*.py")
        if not any(segment in SKIP_DIRS for segment in path.parts)
    ]
    rel_paths = [str(path.relative_to(root)) for path in files]
    cfg = build_cfg_for_files(rel_paths, base_path=str(root))
    cfg["repo_root"] = str(root)
    return cfg
