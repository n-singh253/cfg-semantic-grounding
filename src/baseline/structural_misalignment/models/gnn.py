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
            def __init__(self, input_dim: int = 768, hidden_dim: int = 128, dropout: float = 0.1) -> None:
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
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(hidden_dim, 2),
                )

            def _pool(self, x_dict: Dict[str, "torch.Tensor"], batch_dict: Dict[str, "torch.Tensor"]):
                pooled = []
                for node_type in ("subtask", "code"):
                    x = x_dict.get(node_type)
                    if x is None:
                        pooled.append(torch.zeros((1, self.classifier[0].in_features // 2), device=next(self.parameters()).device))
                        continue
                    if node_type in batch_dict:
                        pooled.append(global_mean_pool(x, batch_dict[node_type]))
                    else:
                        pooled.append(x.mean(dim=0, keepdim=True))
                return torch.cat(pooled, dim=-1)

            def forward(self, data):
                x_dict = {key: value for key, value in data.x_dict.items()}
                x_dict = self.conv1(x_dict, data.edge_index_dict)
                x_dict = {key: F.relu(value) for key, value in x_dict.items()}
                x_dict = self.conv2(x_dict, data.edge_index_dict)
                x_dict = {key: F.relu(value) for key, value in x_dict.items()}
                pooled = self._pool(x_dict, getattr(data, "batch_dict", {}))
                return self.classifier(pooled)

        return _Model(*args, **kwargs)
