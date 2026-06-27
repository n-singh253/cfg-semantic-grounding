"""PyG heterogeneous GNN model for graph-level injection detection."""

from __future__ import annotations

from typing import Dict


def _require_pyg():
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Structural misalignment GNN requires torch-geometric.") from exc
    return torch, nn, F, HeteroConv, SAGEConv, global_mean_pool


class HeteroGraphClassifier:  # pragma: no cover - thin wrapper around torch module
    def __new__(cls, *args, **kwargs):
        torch, nn, F, HeteroConv, SAGEConv, global_mean_pool = _require_pyg()

        class _Model(nn.Module):
            def __init__(
                self,
                input_dim: int = 768,
                hidden_dim: int = 128,
                dropout: float = 0.1,
                use_raw_feature_residual: bool = False,
                use_raw_aux_head: bool = False,
                raw_fusion_weight: float = 0.5,
            ) -> None:
                super().__init__()
                relations = {
                    ("subtask", "depends_on", "subtask"): SAGEConv((input_dim, input_dim), hidden_dim),
                    ("code", "cfg", "code"): SAGEConv((input_dim, input_dim), hidden_dim),
                    ("subtask", "grounds", "code"): SAGEConv((input_dim, input_dim), hidden_dim),
                }
                self.conv1 = HeteroConv(relations, aggr="sum")
                self.conv2 = HeteroConv(
                    {
                        ("subtask", "depends_on", "subtask"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                        ("code", "cfg", "code"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                        ("subtask", "grounds", "code"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    },
                    aggr="sum",
                )
                self.dropout = float(dropout)
                self.use_raw_feature_residual = bool(use_raw_feature_residual)
                self.use_raw_aux_head = bool(use_raw_aux_head)
                self.raw_fusion_weight = float(raw_fusion_weight)
                if not (0.0 <= self.raw_fusion_weight <= 1.0):
                    raise ValueError("raw_fusion_weight must be between 0 and 1")
                if self.use_raw_feature_residual:
                    self.raw_projection = nn.Sequential(
                        nn.Linear(input_dim * 2, hidden_dim * 2),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                    )
                classifier_input_dim = hidden_dim * (4 if self.use_raw_feature_residual else 2)
                self.classifier = nn.Sequential(
                    nn.Linear(classifier_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(hidden_dim, 2),
                )
                if self.use_raw_aux_head:
                    self.raw_classifier = nn.Sequential(
                        nn.LayerNorm(input_dim * 2),
                        nn.Linear(input_dim * 2, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(hidden_dim, 2),
                    )

            def _pool(
                self,
                x_dict: Dict[str, "torch.Tensor"],
                batch_dict: Dict[str, "torch.Tensor"],
                num_graphs: int,
                feature_dim: int,
            ):
                pooled = []
                device = next(self.parameters()).device
                for node_type in ("subtask", "code"):
                    x = x_dict.get(node_type)
                    if x is None or x.numel() == 0:
                        pooled.append(torch.zeros((num_graphs, feature_dim), device=device))
                        continue
                    if node_type in batch_dict:
                        pooled.append(global_mean_pool(x, batch_dict[node_type], size=num_graphs))
                    else:
                        pooled.append(x.mean(dim=0, keepdim=True))
                return torch.cat(pooled, dim=-1)

            def forward_with_aux(self, data):
                raw_x_dict = {key: value for key, value in data.x_dict.items()}
                x_dict = dict(raw_x_dict)
                x_dict = self.conv1(x_dict, data.edge_index_dict)
                x_dict = {key: F.relu(value) for key, value in x_dict.items()}
                x_dict = self.conv2(x_dict, data.edge_index_dict)
                x_dict = {key: F.relu(value) for key, value in x_dict.items()}
                try:
                    batch_dict = data.batch_dict
                except KeyError:
                    batch_dict = {}
                num_graphs = int(data.y.view(-1).size(0)) if hasattr(data, "y") else 1
                hidden_dim = self.classifier[-1].in_features
                pooled = self._pool(x_dict, batch_dict, num_graphs, hidden_dim)
                raw_pooled = None
                if self.use_raw_feature_residual or self.use_raw_aux_head:
                    if self.use_raw_feature_residual:
                        input_dim = self.raw_projection[0].in_features // 2
                    else:
                        input_dim = self.raw_classifier[0].normalized_shape[0] // 2
                    raw_pooled = self._pool(raw_x_dict, batch_dict, num_graphs, input_dim)
                if self.use_raw_feature_residual:
                    pooled = torch.cat([pooled, self.raw_projection(raw_pooled)], dim=-1)
                gnn_logits = self.classifier(pooled)
                raw_logits = self.raw_classifier(raw_pooled) if self.use_raw_aux_head else None
                if raw_logits is None:
                    return gnn_logits, None, gnn_logits
                final_logits = (
                    (1.0 - self.raw_fusion_weight) * gnn_logits
                    + self.raw_fusion_weight * raw_logits
                )
                return final_logits, raw_logits, gnn_logits

            def forward(self, data):
                final_logits, _, _ = self.forward_with_aux(data)
                return final_logits

        return _Model(*args, **kwargs)
