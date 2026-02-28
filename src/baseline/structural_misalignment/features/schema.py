"""Feature schema/version contracts for structural misalignment defense."""

from __future__ import annotations

from typing import Dict, List

FEATURE_SCHEMA_VERSION = "v1"

CANONICAL_MODE_NAMES: Dict[str, str] = {
    "structural_only": "structural_only",
    "similarity_only": "similarity_only",
    "structural_combined": "structural_combined",
    "full_universal": "full_universal",
    "severity_only_universal": "severity_only_universal",
    "no_security": "no_security",
}

STRUCTURAL_FAMILY_MODES = {
    "structural_only",
    "similarity_only",
    "structural_combined",
}

UNIVERSAL_FAMILY_MODES = {
    "full_universal",
    "severity_only_universal",
    "no_security",
}

ALL_SUPPORTED_MODES = set(CANONICAL_MODE_NAMES.keys())


def normalize_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise ValueError(f"Unsupported structural_misalignment mode type: {type(mode)}")
    raw = mode.strip()
    if raw in CANONICAL_MODE_NAMES:
        return raw
    raise ValueError(f"Unsupported structural_misalignment mode: {mode}")


def feature_set_name(mode: str) -> str:
    canonical = normalize_mode(mode)
    return CANONICAL_MODE_NAMES[canonical]


def ensure_feature_order(feature_row: Dict[str, float], expected_order: List[str]) -> List[float]:
    """Return values in expected order, with zero-fill for missing columns."""
    return [float(feature_row.get(col, 0.0) or 0.0) for col in expected_order]
