"""Artifact helpers with hash + config references for auditability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.common.artifact_store import atomic_write_json
from src.common.hashing import sha256_json


def _normalize_refs(refs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(refs, dict):
        return {}
    return refs


def write_hashed_json_artifact(
    path: Path,
    payload: Dict[str, Any],
    *,
    config_hash: str,
    refs: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write an artifact envelope containing payload + hash metadata."""
    envelope = {
        "config_hash": config_hash,
        "hashes": {
            "payload_sha256": sha256_json(payload),
        },
        "refs": _normalize_refs(refs),
        "payload": payload,
    }
    return atomic_write_json(path, envelope)


def read_hashed_json_artifact(path: Path) -> Dict[str, Any]:
    """Read a hashed artifact envelope."""
    return json.loads(path.read_text(encoding="utf-8"))
