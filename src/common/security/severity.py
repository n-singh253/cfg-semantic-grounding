"""Security/severity scoring helpers for CFG nodes."""

from __future__ import annotations

import io
import re
import tokenize
from typing import Any, Dict, List, Optional

from src.common.security.patterns import BENIGN_PATTERNS, select_patterns


def _regex_strip_comments(text: str) -> str:
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _strip_python_comments_and_strings(text: str) -> str:
    out_tokens: List[str] = []
    try:
        token_stream = tokenize.generate_tokens(io.StringIO(text).readline)
        for tok_type, tok_string, *_ in token_stream:
            if tok_type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out_tokens.append(tok_string)
    except (tokenize.TokenError, IndentationError):
        return text
    return " ".join(out_tokens)


def _sanitize_code(code: str, comment_stripping_mode: str = "python_tokenize") -> str:
    mode = comment_stripping_mode.strip().lower()
    if mode == "none":
        return code
    if mode == "python_tokenize":
        return _strip_python_comments_and_strings(code)
    if mode == "regex_conservative":
        return _regex_strip_comments(code)
    return code


def score_node_security(
    node: Dict[str, Any],
    *,
    severity_mode: str = "universal",
    comment_stripping_mode: str = "python_tokenize",
) -> Dict[str, Any]:
    snippet = str(node.get("code_snippet", "") or "")
    if not snippet:
        return {
            "is_suspicious": False,
            "matched_patterns": [],
            "severity_score": 0.0,
            "pattern_details": [],
            "is_benign": False,
            "benign_patterns": [],
            "severity_mode": severity_mode,
            "comment_stripping_mode": comment_stripping_mode,
        }

    scanned = _sanitize_code(snippet, comment_stripping_mode=comment_stripping_mode).lower()

    suspicious_matches: List[tuple[str, str, float]] = []
    max_severity = 0.0
    for needle, category, score in select_patterns(severity_mode):
        if needle in scanned:
            suspicious_matches.append((needle, category, score))
            max_severity = max(max_severity, score)

    benign_matches: List[tuple[str, str, float]] = []
    for needle, category, score in BENIGN_PATTERNS:
        if needle in scanned:
            benign_matches.append((needle, category, score))

    suspicious_score = sum(score for _, _, score in suspicious_matches)
    benign_score = sum(score for _, _, score in benign_matches)
    is_suspicious = bool(suspicious_matches) and max_severity >= 0.5
    is_benign = benign_score > suspicious_score and not is_suspicious

    return {
        "is_suspicious": is_suspicious,
        "matched_patterns": [needle for needle, _, _ in suspicious_matches],
        "severity_score": max_severity,
        "pattern_details": suspicious_matches,
        "is_benign": is_benign,
        "benign_patterns": [needle for needle, _, _ in benign_matches],
        "severity_mode": severity_mode,
        "comment_stripping_mode": comment_stripping_mode,
    }


def is_test_code(file_path: str, function_name: Optional[str] = None) -> bool:
    lower_path = (file_path or "").lower()
    path_markers = ["/test/", "/tests/", "test_", "_test.py", "conftest.py"]
    if any(marker in lower_path for marker in path_markers):
        return True
    if function_name:
        lower_func = function_name.lower()
        if lower_func.startswith("test"):
            return True
    return False


def is_validation_code(node: Dict[str, Any]) -> bool:
    code = str(node.get("code_snippet", "")).lower()
    markers = [
        "validate",
        "check",
        "assert",
        "raise valueerror",
        "raise typeerror",
        "if not ",
        "is none",
        "is not none",
        "isinstance(",
        "hasattr(",
        "verify",
        "ensure",
        "require",
        "expect",
    ]
    return any(marker in code for marker in markers)


def classify_node(
    node: Dict[str, Any],
    *,
    severity_mode: str = "universal",
    comment_stripping_mode: str = "python_tokenize",
) -> Dict[str, Any]:
    node_id = str(node.get("node_id", ""))
    file_path = str(node.get("file", ""))
    function_name = str(node.get("function", ""))

    if "::" in node_id:
        parts = node_id.split("::")
        if len(parts) >= 2:
            file_path = file_path or parts[0]
            function_name = function_name or parts[1]

    sec = score_node_security(
        node,
        severity_mode=severity_mode,
        comment_stripping_mode=comment_stripping_mode,
    )
    test_flag = is_test_code(file_path, function_name)
    val_flag = is_validation_code(node)

    label = "neutral"
    if sec["is_suspicious"] and sec["severity_score"] >= 0.8:
        label = "highly_suspicious"
    elif sec["is_suspicious"]:
        label = "suspicious"
    elif test_flag:
        label = "test_code"
    elif val_flag:
        label = "validation"
    elif sec["is_benign"]:
        label = "benign"

    return {
        "node_id": node.get("node_id", ""),
        "file": file_path,
        "function": function_name,
        "is_suspicious": sec["is_suspicious"],
        "severity_score": sec["severity_score"],
        "suspicious_patterns": sec["matched_patterns"],
        "is_benign": sec["is_benign"],
        "benign_patterns": sec["benign_patterns"],
        "severity_mode": sec["severity_mode"],
        "comment_stripping_mode": sec["comment_stripping_mode"],
        "is_test_code": test_flag,
        "is_validation": val_flag,
        "classification": label,
    }


def analyze_cfg_diff_nodes(
    candidate_nodes: List[Dict[str, Any]],
    *,
    severity_mode: str = "universal",
    comment_stripping_mode: str = "python_tokenize",
) -> Dict[str, Any]:
    analyzed: List[Dict[str, Any]] = []
    suspicious = 0
    benign = 0
    tests = 0
    validation = 0

    for node in candidate_nodes:
        cls = classify_node(
            node,
            severity_mode=severity_mode,
            comment_stripping_mode=comment_stripping_mode,
        )
        combined = {**node, **cls}
        analyzed.append(combined)
        suspicious += int(bool(cls["is_suspicious"]))
        benign += int(bool(cls["is_benign"]))
        tests += int(bool(cls["is_test_code"]))
        validation += int(bool(cls["is_validation"]))

    return {
        "nodes": analyzed,
        "summary": {
            "total_nodes": len(candidate_nodes),
            "suspicious_nodes": suspicious,
            "benign_nodes": benign,
            "test_nodes": tests,
            "validation_nodes": validation,
            "severity_mode": severity_mode,
            "comment_stripping_mode": comment_stripping_mode,
        },
    }
