"""Canonical graph construction for structural misalignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def build_canonical_graph(
    *,
    instance_id: str,
    graph_label: int,
    subtasks: List[Dict[str, Any]],
    candidate_nodes: List[Dict[str, Any]],
    code_edges: List[Dict[str, Any]],
    links: List[Dict[str, Any]],
    subtask_features: np.ndarray,
    code_features: np.ndarray,
) -> Dict[str, Any]:
    subtask_id_to_index = {subtask["subtask_id"]: idx for idx, subtask in enumerate(subtasks)}
    node_id_to_index = {str(node.get("node_id", "")): idx for idx, node in enumerate(candidate_nodes)}

    dependency_edges: List[Dict[str, Any]] = []
    for subtask in subtasks:
        dst_idx = subtask_id_to_index[subtask["subtask_id"]]
        for dependency in subtask.get("depends_on", []):
            if dependency not in subtask_id_to_index:
                continue
            dependency_edges.append(
                {
                    "src": subtask_id_to_index[dependency],
                    "dst": dst_idx,
                    "kind": "depends_on",
                }
            )

    cfg_edges: List[Dict[str, Any]] = []
    for edge in code_edges:
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        if src not in node_id_to_index or dst not in node_id_to_index:
            continue
        cfg_edges.append(
            {
                "src": node_id_to_index[src],
                "dst": node_id_to_index[dst],
                "kind": str(edge.get("kind", "fallthrough")),
            }
        )

    cross_edges: List[Dict[str, Any]] = []
    for link in links:
        subtask_id = str(link.get("subtask_id", ""))
        if subtask_id not in subtask_id_to_index:
            continue
        src_idx = subtask_id_to_index[subtask_id]
        scores = link.get("scores", {}) if isinstance(link.get("scores"), dict) else {}
        for node_id in link.get("node_ids", []):
            if node_id not in node_id_to_index:
                continue
            cross_edges.append(
                {
                    "src": src_idx,
                    "dst": node_id_to_index[node_id],
                    "score": float(scores.get(node_id, 0.0)),
                    "fallback_used": bool(link.get("fallback_used", False)),
                }
            )

    return {
        "instance_id": instance_id,
        "graph_label": int(graph_label),
        "subtasks": subtasks,
        "code_nodes": candidate_nodes,
        "subtask_features": subtask_features.tolist(),
        "code_features": code_features.tolist(),
        "edges": {
            "subtask_to_subtask": dependency_edges,
            "code_to_code": cfg_edges,
            "subtask_to_code": cross_edges,
        },
    }


def _edge_index(edges: List[Dict[str, Any]]):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Graph serialization requires torch.") from exc
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    pairs = [[int(edge["src"]), int(edge["dst"])] for edge in edges]
    return torch.tensor(pairs, dtype=torch.long).T.contiguous()


def build_pyg_heterodata(graph: Dict[str, Any]):
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("PyG graph construction requires torch-geometric.") from exc

    data = HeteroData()
    subtask_features = np.asarray(graph.get("subtask_features", []), dtype=np.float32)
    code_features = np.asarray(graph.get("code_features", []), dtype=np.float32)
    if subtask_features.size == 0:
        subtask_dim = code_features.shape[1] if code_features.ndim == 2 and code_features.size else 768
        subtask_features = np.zeros((0, subtask_dim), dtype=np.float32)
    if code_features.size == 0:
        code_dim = subtask_features.shape[1] if subtask_features.ndim == 2 and subtask_features.size else 768
        code_features = np.zeros((0, code_dim), dtype=np.float32)

    data["subtask"].x = torch.tensor(subtask_features, dtype=torch.float32)
    data["code"].x = torch.tensor(code_features, dtype=torch.float32)
    data["subtask", "depends_on", "subtask"].edge_index = _edge_index(graph.get("edges", {}).get("subtask_to_subtask", []))
    data["code", "cfg", "code"].edge_index = _edge_index(graph.get("edges", {}).get("code_to_code", []))
    data["subtask", "grounds", "code"].edge_index = _edge_index(graph.get("edges", {}).get("subtask_to_code", []))
    data.y = torch.tensor([int(graph.get("graph_label", 0))], dtype=torch.long)
    data.instance_id = str(graph.get("instance_id", "unknown"))
    return data


def write_graph_artifacts(artifact_dir: Path, graph: Dict[str, Any]) -> Dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    graph_json = artifact_dir / "graph.json"
    graph_json.write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
    paths = {"graph_json": str(graph_json)}
    try:
        import torch
        hetero = build_pyg_heterodata(graph)
        graph_pt = artifact_dir / "graph.pt"
        torch.save(hetero, graph_pt)
        paths["graph_pt"] = str(graph_pt)
    except ImportError:
        pass
    return paths
