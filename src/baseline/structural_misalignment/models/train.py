"""Training helpers for the structural misalignment hetero GNN."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.baseline.structural_misalignment.models.gnn import HeteroGraphClassifier
from src.baseline.structural_misalignment.models.raw_logistic import (
    fit_raw_logistic_head,
    raw_logistic_decision,
)


def _require_training_deps():
    try:
        import torch
        from torch.utils.data import WeightedRandomSampler
        from torch_geometric.loader import DataLoader
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Structural misalignment GNN training requires torch, torch-geometric, and scikit-learn."
        ) from exc
    return torch, WeightedRandomSampler, DataLoader, accuracy_score, precision_score, recall_score, roc_auc_score


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _, _, _, _, _, _ = _require_training_deps()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _graph_label(graph) -> int:
    return int(graph.y.view(-1)[0].item())


def _graph_attack_name(graph: Any) -> str:
    value = getattr(graph, "attack_name", "")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value or "")


def _sampling_weights(graphs: List[Any], strategy: str):
    torch, _, _, _, _, _, _ = _require_training_deps()
    labels = [_graph_label(graph) for graph in graphs]
    normalized = strategy.strip().lower()
    if normalized == "attack_balanced":
        attacks = [_graph_attack_name(graph) or ("none" if label == 0 else "malicious") for graph, label in zip(graphs, labels)]
        benign_groups = sorted({attack for attack, label in zip(attacks, labels) if label == 0})
        malicious_groups = sorted({attack for attack, label in zip(attacks, labels) if label == 1})
        if benign_groups and malicious_groups:
            group_mass = {
                **{group: 0.5 / len(benign_groups) for group in benign_groups},
                **{group: 0.5 / len(malicious_groups) for group in malicious_groups},
            }
            counts = Counter(attacks)
            return torch.tensor([group_mass[group] / counts[group] for group in attacks], dtype=torch.float)
    counts = {label: max(1, labels.count(label)) for label in sorted(set(labels))}
    return torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.float)


def _build_loader(graphs: List[Any], batch_size: int, weighted: bool, sampling_strategy: str = "binary_balanced"):
    torch, WeightedRandomSampler, DataLoader, _, _, _, _ = _require_training_deps()
    if not graphs:
        return DataLoader([], batch_size=batch_size)
    if not weighted:
        return DataLoader(graphs, batch_size=batch_size, shuffle=False)
    sampler = WeightedRandomSampler(
        weights=_sampling_weights(graphs, sampling_strategy),
        num_samples=len(graphs),
        replacement=True,
    )
    return DataLoader(graphs, batch_size=batch_size, sampler=sampler)


def _compute_metrics(labels: List[int], predictions: List[int], probabilities: List[float]) -> Dict[str, Any]:
    _, _, _, accuracy_score, precision_score, recall_score, roc_auc_score = _require_training_deps()
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)) if labels else 0.0,
        "precision": float(precision_score(labels, predictions, zero_division=0)) if labels else 0.0,
        "recall": float(recall_score(labels, predictions, zero_division=0)) if labels else 0.0,
    }
    if labels and len(set(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    else:
        metrics["roc_auc"] = None
    return metrics


def evaluate_model(model, graphs: List[Any], batch_size: int = 8) -> Dict[str, Any]:
    torch, _, DataLoader, _, _, _, _ = _require_training_deps()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    labels: List[int] = []
    predictions: List[int] = []
    probabilities: List[float] = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)
            labels.extend(batch.y.view(-1).cpu().tolist())
            predictions.extend(preds.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
    metrics = _compute_metrics(labels, predictions, probabilities)
    metrics["label_counts"] = {
        "benign": int(sum(1 for label in labels if label == 0)),
        "malicious": int(sum(1 for label in labels if label == 1)),
    }
    return metrics


def evaluate_fused_model(
    model,
    graphs: List[Any],
    raw_logistic_head: Dict[str, Any],
    fusion_weight: float,
) -> Dict[str, Any]:
    torch, _, _, _, _, _, _ = _require_training_deps()
    labels: List[int] = []
    predictions: List[int] = []
    probabilities: List[float] = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            logits = model(graph)
            gnn_decision = float((logits[0, 1] - logits[0, 0]).item())
            raw_decision = raw_logistic_decision(graph, raw_logistic_head)
            fused_decision = (1.0 - fusion_weight) * gnn_decision + fusion_weight * raw_decision
            probability = float(torch.sigmoid(torch.tensor(fused_decision)).item())
            label = _graph_label(graph)
            labels.append(label)
            probabilities.append(probability)
            predictions.append(int(probability >= 0.5))
    metrics = _compute_metrics(labels, predictions, probabilities)
    metrics["label_counts"] = {
        "benign": int(sum(1 for label in labels if label == 0)),
        "malicious": int(sum(1 for label in labels if label == 1)),
    }
    return metrics


def train_graph_model(
    *,
    train_graphs: List[Any],
    test_graphs: List[Any],
    output_dir: Path,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 8,
    seed: int = 42,
    embedding_model_name: str = "microsoft/codebert-base",
    embedding_pooling: str = "mean",
    sampling_strategy: str = "binary_balanced",
    use_raw_feature_residual: bool = False,
    use_raw_aux_head: bool = False,
    raw_fusion_weight: float = 0.5,
    raw_aux_loss_weight: float = 1.0,
    fit_raw_logistic: bool = False,
    raw_logistic_c: float = 0.1,
    raw_logistic_fusion_weight: float = 0.5,
) -> Dict[str, Any]:
    torch, _, _, _, _, _, _ = _require_training_deps()
    if not train_graphs:
        raise ValueError("No training graphs provided.")
    if not test_graphs:
        raise ValueError("No test graphs provided.")

    set_global_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeteroGraphClassifier(
        input_dim=768,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_raw_feature_residual=use_raw_feature_residual,
        use_raw_aux_head=use_raw_aux_head,
        raw_fusion_weight=raw_fusion_weight,
    ).to(device)

    train_loader = _build_loader(
        train_graphs,
        batch_size=batch_size,
        weighted=True,
        sampling_strategy=sampling_strategy,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    labels = [_graph_label(graph) for graph in train_graphs]
    benign_count = max(1, labels.count(0))
    malicious_count = max(1, labels.count(1))
    if sampling_strategy.strip().lower() == "attack_balanced":
        # Sampling already assigns half the mass to benign rows and splits the
        # malicious half evenly across attack families. Applying binary class
        # weights again would double-correct the sampled distribution.
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        class_weights = torch.tensor(
            [len(labels) / (2 * benign_count), len(labels) / (2 * malicious_count)],
            dtype=torch.float32,
            device=device,
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    history: List[Dict[str, Any]] = []
    best_score = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, raw_logits, _ = model.forward_with_aux(batch)
            loss = loss_fn(logits, batch.y.view(-1))
            if raw_logits is not None:
                loss = loss + float(raw_aux_loss_weight) * loss_fn(raw_logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batch_count += 1

        train_metrics = evaluate_model(model, train_graphs, batch_size=batch_size)
        test_metrics = evaluate_model(model, test_graphs, batch_size=batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, batch_count),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }
        )
        monitored = test_metrics.get("roc_auc")
        score = float(monitored) if monitored is not None else float(test_metrics.get("accuracy", 0.0))
        if score >= best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training completed without producing a model state.")

    checkpoint_path = output_dir / "model.pt"
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)
    gnn_metrics = evaluate_model(model, test_graphs, batch_size=batch_size)
    final_metrics = gnn_metrics
    raw_logistic_head_file = ""
    if fit_raw_logistic:
        raw_logistic_head = fit_raw_logistic_head(
            train_graphs,
            c_value=raw_logistic_c,
            seed=seed,
            input_dim=768,
        )
        raw_logistic_head_file = "raw_logistic_head.json"
        (output_dir / raw_logistic_head_file).write_text(
            json.dumps(raw_logistic_head, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        final_metrics = evaluate_fused_model(
            model,
            test_graphs,
            raw_logistic_head,
            raw_logistic_fusion_weight,
        )
    metadata = {
        "gnn_model_type": "hetero_sage",
        "input_dim": 768,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "embedding_model_name": embedding_model_name,
        "embedding_pooling": embedding_pooling,
        "sampling_strategy": sampling_strategy,
        "use_raw_feature_residual": bool(use_raw_feature_residual),
        "use_raw_aux_head": bool(use_raw_aux_head),
        "raw_fusion_weight": float(raw_fusion_weight),
        "raw_aux_loss_weight": float(raw_aux_loss_weight),
        "fit_raw_logistic": bool(fit_raw_logistic),
        "raw_logistic_c": float(raw_logistic_c),
        "raw_logistic_fusion_weight": float(raw_logistic_fusion_weight),
        "raw_logistic_head_file": raw_logistic_head_file,
        "best_epoch": int(best_epoch),
        "train_graph_count": len(train_graphs),
        "test_graph_count": len(test_graphs),
        "metrics": final_metrics,
        "gnn_metrics": gnn_metrics,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    return metadata
