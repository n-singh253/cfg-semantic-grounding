"""Canonical schemas/parsers for subtasks and subtask->CFG links."""

from __future__ import annotations

import json
from typing import Any, Dict, List


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
