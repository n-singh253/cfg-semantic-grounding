"""Model bundle loading for structural misalignment inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


@dataclass
class ModelBundle:
    model: Any
    imputer: Any
    scaler: Any
    feature_list: list[str]
    metadata: Dict[str, Any]
    model_dir: Path
    vectorizer: Optional[Any] = None


def _resolve_model_dir(model_path: Path) -> Path:
    if model_path.is_dir():
        return model_path
    return model_path.parent


def load_model_bundle(model_path: str, *, require_vectorizer: bool = False) -> ModelBundle:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")

    model_dir = _resolve_model_dir(path)
    model_file = path if path.is_file() else model_dir / "model.joblib"
    imputer_file = model_dir / "imputer.joblib"
    scaler_file = model_dir / "scaler.joblib"
    metadata_file = model_dir / "metadata.json"

    missing = [
        str(p)
        for p in [model_file, imputer_file, scaler_file, metadata_file]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Model bundle incomplete. Missing required artifact(s): " + ", ".join(missing)
        )

    model = joblib.load(model_file)
    imputer = joblib.load(imputer_file)
    scaler = joblib.load(scaler_file)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    feature_list = metadata.get("feature_list")
    if not isinstance(feature_list, list) or not all(isinstance(c, str) for c in feature_list):
        raise ValueError(f"Invalid model metadata feature_list in {metadata_file}")

    vectorizer = None
    vectorizer_path = metadata.get("vectorizer_path")
    candidate = None
    if isinstance(vectorizer_path, str) and vectorizer_path.strip():
        candidate = Path(vectorizer_path)
        if not candidate.is_absolute():
            candidate = (model_dir / candidate).resolve()
    else:
        maybe = model_dir / "tfidf_vectorizer.pkl"
        if maybe.exists():
            candidate = maybe

    if candidate is not None and candidate.exists():
        vectorizer = joblib.load(candidate)

    if require_vectorizer and vectorizer is None:
        raise FileNotFoundError(
            f"Vectorizer artifact required but missing for model bundle at {model_dir}. "
            "Expected metadata.vectorizer_path or tfidf_vectorizer.pkl"
        )

    return ModelBundle(
        model=model,
        imputer=imputer,
        scaler=scaler,
        feature_list=feature_list,
        metadata=metadata,
        model_dir=model_dir,
        vectorizer=vectorizer,
    )
