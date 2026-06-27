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

    Instance IDs alone are not unique for training artifacts because the same
    LiveCodeBench instance appears in clean, FCV, and SWExploit datasets.
    Include the attack identity and patch hash so clean/adversarial variants
    cannot clobber each other.
    """
    # Mixed evaluation rows suffix ``instance_id`` so clean and malicious
    # variants can coexist in one JSONL.  Their trained graph, however, is
    # indexed under the original finalized-dataset id.  Prefer that retained
    # source id when present so heldout evaluation actually reuses the graph
    # built during training.
    instance_id = _safe(str(row.get("source_instance_id") or row.get("instance_id", "unknown")))
    attack_name = _safe(str(row.get("attack_name", "unknown")))
    patch_hash = str(row.get("patch_hash", "") or row.get("adv_patch_hash", "")).strip()
    if not patch_hash:
        patch_hash = sha256_text(str(row.get("patch_text", "")))
    return f"{instance_id}__{attack_name}__{patch_hash[:12]}"
