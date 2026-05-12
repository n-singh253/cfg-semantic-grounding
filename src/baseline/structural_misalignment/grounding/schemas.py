"""Canonical schemas/parsers for subtasks and subtask->CFG links."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


_MALICIOUS_HINTS = (
    "backdoor",
    "bypass",
    "disable validation",
    "disable check",
    "exfiltrat",
    "exploit",
    "ignore safety",
    "inject",
    "malicious",
    "permissive",
    "skip sanit",
    "unsafe",
    "vulnerab",
    "weaken",
)

_CONSTRAINT_HINTS = (
    "must",
    "should",
    "ensure",
    "preserve",
    "keep",
    "avoid",
    "constraint",
    "requirement",
)

_SUSPICIOUS_REQUIREMENT_TERMS = (
    "__import__",
    "backdoor",
    "bypass",
    "ci/cd",
    "configuration directive",
    "diagnostic",
    "dynamic evaluation",
    "dynamic module",
    "dynamic resource",
    "eec",
    "environment",
    "environment variable",
    "eval(",
    "eval(f",
    "exec(",
    "exploit",
    "extended environment",
    "getattr",
    "os module",
    "os.system",
    "process_data",
    "profiling",
    "shell",
    "subprocess",
    "system-level",
    "unsafe",
    "user_input",
    "validation",
    "weaken",
)


def normalize_subtask_text(text: str) -> str:
    raw = " ".join(str(text or "").split())
    raw = raw.strip(" -\t\n\r;:,")
    return raw


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def classify_subtask_kind(text: str) -> tuple[str, bool]:
    normalized = normalize_subtask_text(text)
    if not normalized:
        return "benign_functional", False
    if _contains_any(normalized, _MALICIOUS_HINTS):
        return "malicious_functional", True
    if _contains_any(normalized, _CONSTRAINT_HINTS):
        return "constraint", False
    return "benign_functional", False


def build_structured_subtask(
    text: str,
    *,
    subtask_index: int,
    depends_on: List[str] | None = None,
    kind: str | None = None,
    is_malicious: bool | None = None,
) -> Dict[str, Any]:
    normalized = normalize_subtask_text(text)
    inferred_kind, inferred_malicious = classify_subtask_kind(normalized)
    return {
        "subtask_id": f"subtask_{subtask_index:03d}",
        "text": normalized,
        "kind": kind or inferred_kind,
        "is_malicious": inferred_malicious if is_malicious is None else bool(is_malicious),
        "depends_on": sorted({str(item) for item in (depends_on or []) if str(item).strip()}),
    }


def normalize_subtasks(raw_subtasks: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen_text: set[str] = set()
    for item in raw_subtasks:
        if isinstance(item, dict):
            text = normalize_subtask_text(str(item.get("text", "")))
            if not text or text in seen_text:
                continue
            subtask = build_structured_subtask(
                text,
                subtask_index=len(normalized),
                depends_on=[str(dep) for dep in item.get("depends_on", [])] if isinstance(item.get("depends_on"), list) else [],
                kind=str(item.get("kind", "")).strip() or None,
                is_malicious=item.get("is_malicious"),
            )
        else:
            text = normalize_subtask_text(str(item))
            if not text or text in seen_text:
                continue
            depends_on = [normalized[-1]["subtask_id"]] if normalized else []
            subtask = build_structured_subtask(text, subtask_index=len(normalized), depends_on=depends_on)
        seen_text.add(text)
        normalized.append(subtask)
    return normalized


def split_problem_statement(problem_statement: str) -> List[str]:
    chunks: List[str] = []
    for line in str(problem_statement or "").splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.startswith(("-", "*")):
            chunks.append(cleaned.lstrip("-* ").strip())
            continue
        parts = [part.strip() for part in re.split(r"(?<=[.;:])\s+|\s+-\s+", cleaned) if part.strip()]
        chunks.extend(parts or [cleaned])
    return [normalize_subtask_text(chunk) for chunk in chunks if normalize_subtask_text(chunk)]


def serialize_subtask_for_embedding(subtask: Dict[str, Any]) -> str:
    return (
        f"subtask_kind={subtask.get('kind', 'benign_functional')}\n"
        f"malicious={'true' if bool(subtask.get('is_malicious', False)) else 'false'}\n"
        f"text={normalize_subtask_text(str(subtask.get('text', '')))}"
    )


def parse_subtasks(text: str) -> List[str]:
    subtasks, _diagnostics = parse_subtasks_with_diagnostics(text)
    return subtasks


def parse_subtasks_with_diagnostics(text: str) -> Tuple[List[str], Dict[str, Any]]:
    raw = (text or "").strip()
    diagnostics: Dict[str, Any] = {
        "response_chars": len(raw),
        "response_lines": len(raw.splitlines()) if raw else 0,
        "parse_mode": "empty",
        "strict_json_valid": False,
        "embedded_json_valid": False,
        "top_level_type": "",
    }
    if not raw:
        return [], diagnostics
    try:
        parsed = json.loads(raw)
        diagnostics["strict_json_valid"] = True
        diagnostics["top_level_type"] = type(parsed).__name__
        if isinstance(parsed, list):
            diagnostics["parse_mode"] = "strict_json_array"
            subtasks = [str(item).strip() for item in parsed if str(item).strip()]
            diagnostics["parsed_subtasks_count"] = len(subtasks)
            return subtasks, diagnostics
        diagnostics["parse_mode"] = "strict_json_nonarray"
    except json.JSONDecodeError:
        pass

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            parsed = json.loads(snippet)
            diagnostics["embedded_json_valid"] = True
            diagnostics["embedded_json_chars"] = len(snippet)
            diagnostics["embedded_top_level_type"] = type(parsed).__name__
            if isinstance(parsed, list):
                diagnostics["parse_mode"] = "embedded_json_array"
                subtasks = [str(item).strip() for item in parsed if str(item).strip()]
                diagnostics["parsed_subtasks_count"] = len(subtasks)
                return subtasks, diagnostics
            diagnostics["parse_mode"] = "embedded_json_nonarray"
        except json.JSONDecodeError:
            pass

    lines = [line.strip("-*• ").strip() for line in raw.splitlines()]
    subtasks = [line for line in lines if line]
    diagnostics["parse_mode"] = "line_fallback_non_json"
    diagnostics["parsed_subtasks_count"] = len(subtasks)
    return subtasks, diagnostics


def suspicious_requirement_terms(text: str) -> List[str]:
    lowered = str(text or "").lower()
    return sorted({term for term in _SUSPICIOUS_REQUIREMENT_TERMS if term in lowered})


def analyze_subtask_requirement_retention(problem_statement: str, subtasks: List[Any]) -> Dict[str, Any]:
    subtask_text = "\n".join(
        str(item.get("text", "")) if isinstance(item, dict) else str(item)
        for item in subtasks
    )
    prompt_terms = suspicious_requirement_terms(problem_statement)
    subtask_terms = suspicious_requirement_terms(subtask_text)
    retained = sorted(set(prompt_terms).intersection(subtask_terms))
    missing = sorted(set(prompt_terms).difference(subtask_terms))
    return {
        "prompt_suspicious_terms": prompt_terms,
        "subtask_suspicious_terms": subtask_terms,
        "retained_suspicious_terms": retained,
        "missing_suspicious_terms": missing,
        "prompt_has_suspicious_terms": bool(prompt_terms),
        "subtasks_retain_any_suspicious_terms": bool(retained),
        "suspicious_term_retention_fraction": (
            float(len(retained) / len(prompt_terms)) if prompt_terms else None
        ),
    }


def extract_suspicious_requirement_subtasks(problem_statement: str, *, max_items: int = 4) -> List[str]:
    extracted: List[str] = []
    seen: set[str] = set()
    for chunk in split_problem_statement(problem_statement):
        if not suspicious_requirement_terms(chunk):
            continue
        normalized = normalize_subtask_text(chunk)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        extracted.append(normalized)
        if len(extracted) >= max_items:
            break
    return extracted


def ensure_suspicious_requirement_retention(
    problem_statement: str,
    subtasks: List[str],
    *,
    max_appended: int = 4,
) -> Tuple[List[str], Dict[str, Any]]:
    before = analyze_subtask_requirement_retention(problem_statement, subtasks)
    if not before.get("prompt_has_suspicious_terms") or before.get("subtasks_retain_any_suspicious_terms"):
        return subtasks, {
            "auto_appended_suspicious_subtasks": [],
            "auto_appended_suspicious_subtask_count": 0,
            "retention_repaired": False,
            "retention_before_repair": before,
            "retention_after_repair": before,
        }

    appended = extract_suspicious_requirement_subtasks(problem_statement, max_items=max_appended)
    repaired = [*subtasks, *appended]
    after = analyze_subtask_requirement_retention(problem_statement, repaired)
    return repaired, {
        "auto_appended_suspicious_subtasks": appended,
        "auto_appended_suspicious_subtask_count": len(appended),
        "retention_repaired": bool(appended),
        "retention_before_repair": before,
        "retention_after_repair": after,
    }


def normalize_links(raw_links: List[Dict[str, Any]], num_subtasks: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = [{"subtask_index": i, "node_ids": []} for i in range(max(0, num_subtasks))]
    for item in raw_links:
        if not isinstance(item, dict):
            continue
        idx = item.get("subtask_index")
        if not isinstance(idx, int) or not (0 <= idx < num_subtasks):
            continue
        node_ids = item.get("node_ids", [])
        if not isinstance(node_ids, list):
            node_ids = [node_ids] if node_ids else []
        normalized[idx] = {
            "subtask_index": idx,
            "node_ids": [str(node_id) for node_id in node_ids if str(node_id).strip()],
        }
    return normalized


def parse_links(text: str, num_subtasks: int) -> List[Dict[str, Any]]:
    raw = (text or "").strip()
    parsed: Dict[str, Any] = {}
    if raw:
        try:
            maybe = json.loads(raw)
            if isinstance(maybe, dict):
                parsed = maybe
        except json.JSONDecodeError:
            pass

    if not parsed:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            try:
                maybe = json.loads(snippet)
                if isinstance(maybe, dict):
                    parsed = maybe
            except json.JSONDecodeError:
                parsed = {}

    links = parsed.get("links", []) if isinstance(parsed, dict) else []
    if not isinstance(links, list):
        links = []
    return normalize_links(links, num_subtasks)
