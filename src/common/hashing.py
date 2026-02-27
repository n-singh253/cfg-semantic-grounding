"""Hash helpers for prompts, patches, and config snapshots."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return sha256_text(payload)
