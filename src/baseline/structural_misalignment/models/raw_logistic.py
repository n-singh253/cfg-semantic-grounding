"""Training and inference helpers for the standardized raw graph head."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def raw_graph_vector(graph: Any, input_dim: int = 768) -> np.ndarray:
    parts: List[np.ndarray] = []
    for node_type in ("subtask", "code"):
        features = graph[node_type].x
        if features.numel():
            parts.append(features.detach().cpu().mean(dim=0).numpy().astype(np.float64, copy=False))
        else:
            parts.append(np.zeros(input_dim, dtype=np.float64))
    return np.concatenate(parts)


def fit_raw_logistic_head(
    graphs: List[Any],
    *,
    c_value: float = 0.1,
    seed: int = 42,
    input_dim: int = 768,
) -> Dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - training dependency guard.
        raise ImportError("Raw logistic graph-head training requires scikit-learn.") from exc

    features = np.asarray([raw_graph_vector(graph, input_dim) for graph in graphs])
    labels = np.asarray([int(graph.y.view(-1)[0].item()) for graph in graphs])
    scaler = StandardScaler().fit(features)
    normalized = scaler.transform(features)
    classifier = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=3000,
        random_state=int(seed),
    ).fit(normalized, labels)
    return {
        "schema_version": 1,
        "input_dim": int(input_dim),
        "c_value": float(c_value),
        "seed": int(seed),
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "coef": classifier.coef_[0].astype(float).tolist(),
        "intercept": float(classifier.intercept_[0]),
    }


def raw_logistic_decision(graph: Any, payload: Dict[str, Any]) -> float:
    input_dim = int(payload.get("input_dim", 768))
    vector = raw_graph_vector(graph, input_dim)
    mean = np.asarray(payload["mean"], dtype=np.float64)
    scale = np.asarray(payload["scale"], dtype=np.float64)
    coef = np.asarray(payload["coef"], dtype=np.float64)
    if not (vector.shape == mean.shape == scale.shape == coef.shape):
        raise ValueError(
            "Raw logistic graph-head shape mismatch: "
            f"vector={vector.shape} mean={mean.shape} scale={scale.shape} coef={coef.shape}"
        )
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    normalized = (vector - mean) / safe_scale
    return float(np.dot(coef, normalized) + float(payload["intercept"]))
