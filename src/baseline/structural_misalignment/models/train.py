"""Training helpers for the structural misalignment hetero GNN."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.baseline.structural_misalignment.models.gnn import HeteroGraphClassifier


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


def _build_loader(graphs: List[Any], batch_size: int, weighted: bool):
    torch, WeightedRandomSampler, DataLoader, _, _, _, _ = _require_training_deps()
    if not graphs:
        return DataLoader([], batch_size=batch_size)
    if not weighted:
        return DataLoader(graphs, batch_size=batch_size, shuffle=False)
    labels = [_graph_label(graph) for graph in graphs]
    counts = {label: max(1, labels.count(label)) for label in sorted(set(labels))}
    sample_weights = [1.0 / counts[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
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
) -> Dict[str, Any]:
    torch, _, _, _, _, _, _ = _require_training_deps()
    if not train_graphs:
        raise ValueError("No training graphs provided.")
    if not test_graphs:
        raise ValueError("No test graphs provided.")

    set_global_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeteroGraphClassifier(input_dim=768, hidden_dim=hidden_dim, dropout=dropout).to(device)

    train_loader = _build_loader(train_graphs, batch_size=batch_size, weighted=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    labels = [_graph_label(graph) for graph in train_graphs]
    benign_count = max(1, labels.count(0))
    malicious_count = max(1, labels.count(1))
    class_weights = torch.tensor(
        [len(labels) / (2 * benign_count), len(labels) / (2 * malicious_count)],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    history: List[Dict[str, Any]] = []
    best_score = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y.view(-1))
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
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training completed without producing a model state.")

    checkpoint_path = output_dir / "model.pt"
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)
    final_metrics = evaluate_model(model, test_graphs, batch_size=batch_size)
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
        "train_graph_count": len(train_graphs),
        "test_graph_count": len(test_graphs),
        "metrics": final_metrics,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    return metadata
