"""Inference helpers for structural misalignment model bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.common.models.load import ModelBundle


@dataclass
class InferenceResult:
    score: float
    prediction: int
    missing_columns_filled_zero: List[str]


def _as_2d_array(values: List[float]) -> np.ndarray:
    return np.asarray([values], dtype=float)


def _score_with_model(model, x_scaled: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x_scaled)
        if prob.ndim == 2 and prob.shape[1] >= 2:
            return float(prob[0, 1])
        return float(prob.ravel()[0])
    if hasattr(model, "decision_function"):
        value = float(model.decision_function(x_scaled)[0])
        return float(1.0 / (1.0 + np.exp(-value)))
    pred = float(model.predict(x_scaled)[0])
    return min(1.0, max(0.0, pred))


def predict_reject_score(bundle: ModelBundle, feature_row: Dict[str, float]) -> InferenceResult:
    # Eval-only guard: inference path must never call fit().
    if not hasattr(bundle.model, "predict"):
        raise ValueError("Loaded model object does not support predict()")

    ordered: List[float] = []
    missing: List[str] = []
    for col in bundle.feature_list:
        if col not in feature_row:
            ordered.append(0.0)
            missing.append(col)
            continue
        try:
            ordered.append(float(feature_row[col]))
        except (TypeError, ValueError):
            ordered.append(0.0)
            missing.append(col)

    x = _as_2d_array(ordered)
    x_imp = bundle.imputer.transform(x)
    x_scaled = bundle.scaler.transform(x_imp)
    score = _score_with_model(bundle.model, x_scaled)
    prediction = int(score >= 0.5)

    return InferenceResult(
        score=float(score),
        prediction=prediction,
        missing_columns_filled_zero=missing,
    )


def decide_from_policy(score: float, threshold: float, decision_policy: str) -> bool:
    policy = decision_policy.strip().lower()
    if policy == "reject_if_score_ge_threshold":
        return score < threshold
    if policy == "accept_if_score_ge_threshold":
        return score >= threshold
    raise ValueError(f"Unsupported decision_policy: {decision_policy}")
