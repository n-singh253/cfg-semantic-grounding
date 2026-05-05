"""Shared deterministic embedding helpers for subtasks and code nodes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np

from src.baseline.structural_misalignment.grounding.schemas import normalize_subtask_text


def serialize_code_node_for_embedding(node: Dict[str, Any]) -> str:
    return (
        f"file={str(node.get('file', ''))}\n"
        f"function={str(node.get('function', ''))}\n"
        f"lines={int(node.get('start_line', 0) or 0)}-{int(node.get('end_line', 0) or 0)}\n"
        f"node_type={str(node.get('node_type', 'basic_block'))}\n"
        f"change_type={str(node.get('change_type', 'modified'))}\n"
        "code:\n"
        f"{normalize_subtask_text(str(node.get('code_snippet', '')))}"
    )


@dataclass
class EmbeddingBatch:
    vectors: np.ndarray
    model_name: str
    device: str
    pooling: str


def _require_embedding_deps():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Structural misalignment embedding pipeline requires torch and transformers."
        ) from exc
    return torch, AutoModel, AutoTokenizer


@lru_cache(maxsize=4)
def _load_encoder(model_name: str):
    torch, AutoModel, AutoTokenizer = _require_embedding_deps()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Prefer safetensors so recent Transformers versions do not block loading
    # PyTorch .bin checkpoints on older torch releases.
    model = AutoModel.from_pretrained(model_name, use_safetensors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def encode_texts(
    texts: List[str],
    *,
    model_name: str,
    pooling: str = "mean",
) -> EmbeddingBatch:
    torch, _, _ = _require_embedding_deps()
    tokenizer, model, device = _load_encoder(model_name)
    if not texts:
        hidden_size = int(getattr(model.config, "hidden_size", 768))
        return EmbeddingBatch(
            vectors=np.zeros((0, hidden_size), dtype=np.float32),
            model_name=model_name,
            device=device,
            pooling=pooling,
        )

    with torch.no_grad():
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model(**encoded)
        hidden = output.last_hidden_state
        if pooling != "mean":
            raise ValueError(f"Unsupported embedding pooling: {pooling}")
        attention = encoded["attention_mask"].unsqueeze(-1)
        summed = (hidden * attention).sum(dim=1)
        counts = attention.sum(dim=1).clamp(min=1)
        vectors = (summed / counts).detach().cpu().numpy().astype(np.float32)

    return EmbeddingBatch(vectors=vectors, model_name=model_name, device=device, pooling=pooling)


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
    left_norm = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    right_norm = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    return np.matmul(left_norm, right_norm.T).astype(np.float32)
