"""Feature schema/version contracts for structural misalignment defense."""

from __future__ import annotations

from typing import Dict, List

FEATURE_SCHEMA_VERSION = "v1"

FEATURE_SET_NAMES: Dict[str, str] = {
    "task8_structure_only": "task8_structure_only",
    "task8_similarity_only": "task8_similarity_only",
    "task8_combined": "task8_combined",
    "task7_full_universal": "task7_full_universal",
    "task7_severity_only_universal": "task7_severity_only_universal",
    "task7_no_security": "task7_no_security",
}


def feature_set_name(mode: str) -> str:
    if mode not in FEATURE_SET_NAMES:
        raise ValueError(f"Unsupported structural_misalignment mode: {mode}")
    return FEATURE_SET_NAMES[mode]


def ensure_feature_order(feature_row: Dict[str, float], expected_order: List[str]) -> List[float]:
    """Return values in expected order, with zero-fill for missing columns."""
    return [float(feature_row.get(col, 0.0) or 0.0) for col in expected_order]
