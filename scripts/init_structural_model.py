#!/usr/bin/env python3
"""Create a local demo model bundle for structural_misalignment canary runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.common.features.task8 import TASK8_COMBINED_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize demo structural model bundle")
    parser.add_argument(
        "--out",
        default="data/models/structural_misalignment/task8_combined",
        help="Output model bundle directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parents[1] / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_rows = 80
    x = rng.normal(0, 1, size=(n_rows, len(TASK8_COMBINED_FEATURES)))
    score = x[:, 0] + 0.8 * x[:, 2] + 0.5 * x[:, 6] + rng.normal(0, 0.5, size=n_rows)
    y = (score > np.median(score)).astype(int)

    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()
    x_imp = imputer.fit_transform(x)
    x_scaled = scaler.fit_transform(x_imp)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
    model.fit(x_scaled, y)

    vectorizer = TfidfVectorizer(max_features=256, ngram_range=(1, 2), sublinear_tf=True)
    vectorizer.fit(
        [
            "validate user input and parse request payload",
            "add fallback handling for edge case parsing",
            "update tests for regression scenario",
            "ensure compatibility while preserving sanitization",
        ]
    )

    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(imputer, out_dir / "imputer.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.pkl")

    metadata = {
        "feature_list": TASK8_COMBINED_FEATURES,
        "feature_schema_version": "v1",
        "feature_set_name": "task8_combined",
        "decision_policy_default": "reject_if_score_ge_threshold",
        "missing_value_policy": "fill_zero",
        "tfidf_fitted": True,
        "mask_tokens": True,
        "vectorizer_path": "tfidf_vectorizer.pkl",
        "trained_at": "2026-02-27T00:00:00Z",
        "task": "task8",
        "notes": "Demo local model bundle for canary harness runs.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[init-structural-model] wrote bundle to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
