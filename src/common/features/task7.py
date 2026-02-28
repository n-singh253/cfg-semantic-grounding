"""Legacy compatibility shim for universal feature extraction."""

from src.common.features.universal_features import (  # noqa: F401
    FEATURE_COLUMNS,
    SECURITY_FEATURE_NAMES,
    SECURITY_FEATURE_PREFIXES,
    SEVERITY_ONLY_FEATURES_UNIVERSAL,
    columns_for_task7_mode,
    columns_for_universal_mode,
    compute_link_entropy,
    compute_suspicious_misassignment,
    count_suspicious_categories,
    extract_task7_features,
    extract_universal_features,
    is_security_feature,
    select_feature_subset,
)
