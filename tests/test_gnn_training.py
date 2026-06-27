from types import SimpleNamespace

import pytest

from src.baseline.structural_misalignment.models.train import _sampling_weights
from src.baseline.structural_misalignment.models.gnn import HeteroGraphClassifier
from src.baseline.structural_misalignment.graph.build import build_pyg_heterodata
from src.baseline.structural_misalignment.models.raw_logistic import (
    fit_raw_logistic_head,
    raw_logistic_decision,
)


def _graph(torch, label: int, attack_name: str):
    return SimpleNamespace(y=torch.tensor([label]), attack_name=attack_name)


def test_attack_balanced_sampling_splits_mass_between_benign_and_attack_families():
    torch = pytest.importorskip("torch")
    graphs = [
        *[_graph(torch, 0, "none") for _ in range(4)],
        *[_graph(torch, 1, "fcv_cwe78") for _ in range(2)],
        *[_graph(torch, 1, "swexploit_anthropic") for _ in range(6)],
    ]

    weights = _sampling_weights(graphs, "attack_balanced")

    assert float(weights[:4].sum()) == pytest.approx(0.5)
    assert float(weights[4:6].sum()) == pytest.approx(0.25)
    assert float(weights[6:].sum()) == pytest.approx(0.25)


def test_raw_aux_head_returns_fused_and_component_logits():
    torch = pytest.importorskip("torch")
    graph = build_pyg_heterodata(
        {
            "instance_id": "row",
            "graph_label": 1,
            "subtask_features": [[0.1] * 768],
            "code_features": [[0.2] * 768],
            "edges": {
                "subtask_to_subtask": [{"src": 0, "dst": 0}],
                "code_to_code": [{"src": 0, "dst": 0}],
                "subtask_to_code": [{"src": 0, "dst": 0}],
            },
        }
    )
    model = HeteroGraphClassifier(
        use_raw_aux_head=True,
        raw_fusion_weight=0.75,
    )
    model.eval()

    with torch.no_grad():
        fused, raw, gnn = model.forward_with_aux(graph)

    assert fused.shape == raw.shape == gnn.shape == (1, 2)
    assert torch.allclose(fused, 0.25 * gnn + 0.75 * raw)


def test_standardized_raw_logistic_head_separates_training_graphs():
    def graph(value: float, label: int):
        return build_pyg_heterodata(
            {
                "instance_id": f"row-{value}",
                "graph_label": label,
                "subtask_features": [[value] * 768],
                "code_features": [[value] * 768],
                "edges": {
                    "subtask_to_subtask": [{"src": 0, "dst": 0}],
                    "code_to_code": [{"src": 0, "dst": 0}],
                    "subtask_to_code": [{"src": 0, "dst": 0}],
                },
            }
        )

    graphs = [graph(-2.0, 0), graph(-1.0, 0), graph(1.0, 1), graph(2.0, 1)]
    payload = fit_raw_logistic_head(graphs, c_value=1.0)

    assert raw_logistic_decision(graphs[0], payload) < 0
    assert raw_logistic_decision(graphs[-1], payload) > 0
