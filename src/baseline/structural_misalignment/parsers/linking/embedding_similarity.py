"""Deterministic embedding-similarity linker for structured subtasks and code nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.baseline.structural_misalignment.embeddings import (
    cosine_similarity_matrix,
    encode_texts,
    serialize_code_node_for_embedding,
)
from src.baseline.structural_misalignment.grounding.schemas import (
    normalize_subtasks,
    serialize_subtask_for_embedding,
)
from src.baseline.structural_misalignment.parsers.registry import register_linker


def embedding_similarity_linker(
    *,
    subtasks: List[Dict[str, Any]] | List[str],
    candidate_nodes: List[Dict[str, Any]],
    artifact_dir: Path,
    config: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    del kwargs
    config = config or {}
    model_name = str(config.get("embedding_model_name", "microsoft/codebert-base"))
    pooling = str(config.get("embedding_pooling", "mean"))
    threshold = float(config.get("link_similarity_threshold", 0.35))
    topk_fallback = int(config.get("link_topk_fallback", 1))

    normalized_subtasks = normalize_subtasks(list(subtasks))
    links = [
        {
            "subtask_id": subtask["subtask_id"],
            "subtask_index": idx,
            "node_ids": [],
            "scores": {},
            "fallback_used": False,
        }
        for idx, subtask in enumerate(normalized_subtasks)
    ]

    metadata: Dict[str, Any] = {
        "linker": "embedding_similarity",
        "embedding_model_name": model_name,
        "embedding_pooling": pooling,
        "similarity_threshold": threshold,
        "link_topk_fallback": topk_fallback,
        "subtask_count": len(normalized_subtasks),
        "node_count": len(candidate_nodes),
        "deterministic": True,
    }
    if not normalized_subtasks or not candidate_nodes:
        metadata["warning"] = "No subtasks or candidate nodes available for linking."
        return links, metadata

    subtask_batch = encode_texts(
        [serialize_subtask_for_embedding(subtask) for subtask in normalized_subtasks],
        model_name=model_name,
        pooling=pooling,
    )
    node_batch = encode_texts(
        [serialize_code_node_for_embedding(node) for node in candidate_nodes],
        model_name=model_name,
        pooling=pooling,
    )
    similarity = cosine_similarity_matrix(subtask_batch.vectors, node_batch.vectors)
    node_ids = [str(node.get("node_id", "")) for node in candidate_nodes]

    similarity_artifact = {
        "subtasks": normalized_subtasks,
        "node_ids": node_ids,
        "similarity": similarity.tolist(),
        "threshold": threshold,
        "topk_fallback": topk_fallback,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = artifact_dir / "similarity_matrix.json"
    matrix_path.write_text(json.dumps(similarity_artifact, indent=2, sort_keys=True), encoding="utf-8")

    total_links = 0
    for row_idx, subtask in enumerate(normalized_subtasks):
        ranked = sorted(
            [
                (float(similarity[row_idx, node_idx]), node_ids[node_idx])
                for node_idx in range(len(candidate_nodes))
            ],
            key=lambda item: (-item[0], item[1]),
        )
        selected = [(score, node_id) for score, node_id in ranked if score >= threshold]
        fallback_used = False
        if not selected and ranked and topk_fallback > 0:
            selected = ranked[:topk_fallback]
            fallback_used = True
        links[row_idx] = {
            "subtask_id": subtask["subtask_id"],
            "subtask_index": row_idx,
            "node_ids": [node_id for _, node_id in selected],
            "scores": {node_id: score for score, node_id in selected},
            "fallback_used": fallback_used,
        }
        total_links += len(selected)

    metadata.update(
        {
            "device": subtask_batch.device,
            "similarity_artifact_path": str(matrix_path),
            "link_count": total_links,
            "zero_link_subtasks_before_fallback": sum(
                1
                for row_idx in range(len(normalized_subtasks))
                if not any(float(similarity[row_idx, node_idx]) >= threshold for node_idx in range(len(candidate_nodes)))
            ),
        }
    )
    return links, metadata


register_linker("embedding_similarity")(embedding_similarity_linker)
