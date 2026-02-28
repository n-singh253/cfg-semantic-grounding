"""Legacy compatibility shim for structural feature extraction."""

from src.common.features.structural_features import (  # noqa: F401
    SIMILARITY_FEATURES,
    STRUCTURAL_FEATURES,
    SIMILARITY_ONLY_FEATURES,
    STRUCTURAL_COMBINED_FEATURES,
    STRUCTURAL_ONLY_FEATURES,
    TASK8_COMBINED_FEATURES,
    TASK8_SIMILARITY_ONLY_FEATURES,
    TASK8_STRUCTURE_ONLY_FEATURES,
    compute_structural_feature_row,
    compute_task8_feature_row,
    compute_similarity_features,
    compute_structural_features,
    deserialize_from_csv,
    fit_tfidf_vectorizer,
    mask_code_tokens,
    parse_json_column,
    serialize_for_csv,
    validate_links_schema,
)
