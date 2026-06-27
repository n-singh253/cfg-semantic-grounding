"""Inference helpers for graph-level structural misalignment detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphInferenceResult:
    injection_probability: float
    prediction: int


def predict_graph_probability(bundle, graph) -> GraphInferenceResult:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Graph inference requires torch.") from exc

    model = bundle.model
    with torch.no_grad():
        logits = model(graph)
        if bundle.raw_logistic_head is not None:
            from src.baseline.structural_misalignment.models.raw_logistic import raw_logistic_decision

            gnn_decision = float((logits[0, 1] - logits[0, 0]).item())
            raw_decision = raw_logistic_decision(graph, bundle.raw_logistic_head)
            fusion_weight = float(bundle.metadata.get("raw_logistic_fusion_weight", 0.5))
            fused_decision = (1.0 - fusion_weight) * gnn_decision + fusion_weight * raw_decision
            malicious_prob = float(torch.sigmoid(torch.tensor(fused_decision)).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            malicious_prob = float(probs[0, 1].item())
    return GraphInferenceResult(
        injection_probability=malicious_prob,
        prediction=int(malicious_prob >= 0.5),
    )


def decide_from_policy(score: float, threshold: float, decision_policy: str) -> bool:
    policy = decision_policy.strip().lower()
    if policy == "reject_if_score_ge_threshold":
        return score < threshold
    if policy == "accept_if_score_ge_threshold":
        return score >= threshold
    raise ValueError(f"Unsupported decision_policy: {decision_policy}")
