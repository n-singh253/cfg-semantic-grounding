"""Load trained structural misalignment GNN bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.baseline.structural_misalignment.models.gnn import HeteroGraphClassifier


@dataclass
class GraphModelBundle:
    model: Any
    metadata: Dict[str, Any]
    model_dir: Path
    checkpoint_path: Path
    raw_logistic_head: Dict[str, Any] | None = None


def load_graph_model_bundle(model_path: str) -> GraphModelBundle:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Loading structural misalignment GNN bundles requires torch.") from exc

    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")

    model_dir = path if path.is_dir() else path.parent
    checkpoint = path if path.is_file() else model_dir / "model.pt"
    metadata_path = model_dir / "metadata.json"
    if not checkpoint.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Model bundle incomplete at {model_dir}. Expected model.pt and metadata.json."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = HeteroGraphClassifier(
        input_dim=int(metadata.get("input_dim", 768)),
        hidden_dim=int(metadata.get("hidden_dim", 128)),
        dropout=float(metadata.get("dropout", 0.1)),
        use_raw_feature_residual=bool(metadata.get("use_raw_feature_residual", False)),
        use_raw_aux_head=bool(metadata.get("use_raw_aux_head", False)),
        raw_fusion_weight=float(metadata.get("raw_fusion_weight", 0.5)),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    raw_logistic_head = None
    raw_head_file = str(metadata.get("raw_logistic_head_file", "") or "").strip()
    if raw_head_file:
        raw_head_path = model_dir / raw_head_file
        if not raw_head_path.exists():
            raise FileNotFoundError(f"Raw logistic graph head not found: {raw_head_path}")
        raw_logistic_head = json.loads(raw_head_path.read_text(encoding="utf-8"))
    return GraphModelBundle(
        model=model,
        metadata=metadata,
        model_dir=model_dir,
        checkpoint_path=checkpoint,
        raw_logistic_head=raw_logistic_head,
    )
