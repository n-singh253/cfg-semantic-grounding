"""Helpers for stable structural graph cache keys."""

from __future__ import annotations

import re
from typing import Any, Dict

from src.common.hashing import sha256_text


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(value: str) -> str:
    return _SAFE.sub("_", value).strip("._") or "unknown"


def structural_graph_key(row: Dict[str, Any]) -> str:
    """Return a stable key for a row's structural graph artifact.

    Code-base values alone are not unique because each benchmark task normally
    has both original and adversarial rows. Include label, row index, and hashes
    when present so variants cannot clobber each other.
    """
    instance_id = _safe(str(row.get("artifact_instance_id") or row.get("instance_id") or row.get("code_base") or "unknown"))
    label = _safe(str(row.get("label", row.get("graph_label", "unknown"))))
    row_index = _safe(str(row.get("row_index", "unknown")))
    patch_hash = str(row.get("patch_hash", "") or row.get("adv_patch_hash", "")).strip()
    if not patch_hash:
        patch_hash = sha256_text(str(row.get("patch", "") or row.get("patch_text", "")))
    prompt_hash = str(row.get("prompt_hash", "")).strip()
    suffix = patch_hash[:12]
    if prompt_hash:
        suffix = f"{prompt_hash[:8]}_{suffix}"
    return f"{instance_id}__label_{label}__row_{row_index}__{suffix}"
