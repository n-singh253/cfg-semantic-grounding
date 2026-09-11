"""Tabular classifiers for structural misalignment feature rows."""

from __future__ import annotations

import math
import os
import warnings
from typing import Any, Dict, Iterable, List, Sequence


def numeric(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        try:
            parsed = float(value)
        except Exception:
            return 0.0
    if math.isnan(parsed) or math.isinf(parsed):
        return 0.0
    return parsed


def feature_columns(samples: Iterable[Dict[str, Any]]) -> List[str]:
    columns = sorted({name for sample in samples for name in sample["features"]})
    if not columns:
        raise ValueError("No feature columns found.")
    return columns


def matrix(samples: Sequence[Dict[str, Any]], columns: Sequence[str]):
    import numpy as np

    x = np.array(
        [[numeric(sample["features"].get(col, 0.0)) for col in columns] for sample in samples],
        dtype=float,
    )
    y = np.array([int(sample["label"]) for sample in samples], dtype=int)
    return x, y


def evaluate_predictions(y_true, y_pred, y_score=None) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_score is not None and len(set(y_true.tolist())) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["roc_auc"] = None
    return out


def train_feature_models(
    train: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (
        AdaBoostClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    x_train, y_train = matrix(train, columns)
    x_test, y_test = matrix(test, columns)
    n_jobs = int(os.environ.get("FEATURE_CLASSIFIER_N_JOBS", "-1"))
    knn_neighbors = max(1, min(5, len(train)))
    specs = {
        "Dummy Most Frequent": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression L2": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)),
            ]
        ),
        "Logistic Regression L1": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Ridge Classifier": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", RidgeClassifier(class_weight="balanced", random_state=seed)),
            ]
        ),
        "Linear SVM": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="linear", class_weight="balanced", random_state=seed)),
            ]
        ),
        "RBF SVM": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        ),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=seed),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, random_state=seed),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=seed),
        "KNN": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=knn_neighbors))]),
        "Gaussian Naive Bayes": Pipeline([("scale", StandardScaler()), ("clf", GaussianNB())]),
    }

    metrics: Dict[str, Any] = {}
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for name, model in specs.items():
        try:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = None
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(x_test)[:, 1]
            elif hasattr(model, "decision_function"):
                score = model.decision_function(x_test)
            metrics[name] = evaluate_predictions(y_test, pred, score)
        except Exception as exc:
            metrics[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return metrics
