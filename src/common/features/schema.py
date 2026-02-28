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

LEGACY_MODE_ALIASES: Dict[str, str] = {
    "task8_structure_only": "structural_only",
    "task8_similarity_only": "similarity_only",
    "task8_combined": "structural_combined",
    "task7_full_universal": "full_universal",
    "task7_severity_only_universal": "severity_only_universal",
    "task7_no_security": "no_security",
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

ALL_SUPPORTED_MODES = set(CANONICAL_MODE_NAMES.keys()).union(LEGACY_MODE_ALIASES.keys())


def normalize_structural_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise ValueError(f"Unsupported structural_misalignment mode type: {type(mode)}")
    raw = mode.strip()
    if raw in LEGACY_MODE_ALIASES:
        return LEGACY_MODE_ALIASES[raw]
    if raw in CANONICAL_MODE_NAMES:
        return raw
    raise ValueError(f"Unsupported structural_misalignment mode: {mode}")


def is_legacy_mode(mode: str) -> bool:
    return isinstance(mode, str) and mode.strip() in LEGACY_MODE_ALIASES


def feature_set_name(mode: str) -> str:
    canonical = normalize_structural_mode(mode)
    return CANONICAL_MODE_NAMES[canonical]


def ensure_feature_order(feature_row: Dict[str, float], expected_order: List[str]) -> List[float]:
    """Return values in expected order, with zero-fill for missing columns."""
    return [float(feature_row.get(col, 0.0) or 0.0) for col in expected_order]
