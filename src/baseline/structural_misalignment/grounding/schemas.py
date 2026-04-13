"""Canonical schemas/parsers for subtasks and subtask->CFG links."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


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
    raw = (text or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass

    lines = [line.strip("-*• ").strip() for line in raw.splitlines()]
    return [line for line in lines if line]


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
