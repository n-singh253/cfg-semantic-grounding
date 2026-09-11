"""Shared deterministic embedding helpers for subtasks and code nodes."""

from __future__ import annotations

import os
import hashlib
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np

from src.baseline.structural_misalignment.grounding.schemas import normalize_subtask_text


_ENCODER_LOAD_LOCK = threading.Lock()


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
        _patch_transformers_runtime()
        from transformers import AutoModel, AutoTokenizer
        _patch_transformers_runtime()
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Structural misalignment embedding pipeline requires torch and transformers."
        ) from exc
    return torch, AutoModel, AutoTokenizer


def _patch_transformers_runtime() -> None:
    """Patch optional Transformers features that are irrelevant for text embeddings."""

    try:
        from transformers.utils import import_utils

        import_utils._torchvision_available = False
    except Exception:
        pass

    try:
        import transformers.tokenization_utils_base as tokenization_utils_base
        import transformers.utils.hub as hub

        original = hub.list_repo_templates
        if getattr(original, "_cfg_semantic_grounding_patched", False):
            safe_list_repo_templates = original
        else:
            def safe_list_repo_templates(*args: Any, **kwargs: Any) -> List[str]:
                try:
                    return original(*args, **kwargs)
                except Exception as exc:
                    text = str(exc)
                    exc_name = type(exc).__name__
                    if (
                        "additional_chat_templates" in text
                        or "Entry Not Found" in text
                        or exc_name in {"EntryNotFoundError", "RemoteEntryNotFoundError"}
                    ):
                        return []
                    raise

            safe_list_repo_templates._cfg_semantic_grounding_patched = True  # type: ignore[attr-defined]

        hub.list_repo_templates = safe_list_repo_templates
        tokenization_utils_base.list_repo_templates = safe_list_repo_templates
    except Exception:
        pass


def _embedding_device(torch, device: str | None = None) -> str:
    requested = (os.environ.get("CFG_STRUCTURAL_EMBEDDING_DEVICE") or device or "auto").strip().lower()
    if requested in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported embedding device: {device!r}")
    return requested


@lru_cache(maxsize=8)
def _load_encoder_cached(model_name: str, device: str):
    torch, AutoModel, AutoTokenizer = _require_embedding_deps()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Prefer safetensors so recent Transformers versions do not block loading
    # PyTorch .bin checkpoints on older torch releases.
    model = AutoModel.from_pretrained(model_name, use_safetensors=True)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _load_encoder(model_name: str, device: str):
    with _ENCODER_LOAD_LOCK:
        return _load_encoder_cached(model_name, device)


def clear_embedding_encoder_cache() -> None:
    """Release cached embedding models before later GPU-heavy stages."""
    with _ENCODER_LOAD_LOCK:
        _load_encoder_cached.cache_clear()


def _deterministic_hash_embeddings(texts: List[str], model_name: str, pooling: str) -> EmbeddingBatch:
    dim = int(os.environ.get("CFG_STRUCTURAL_HASH_EMBEDDING_DIM", "768"))
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        seed = hashlib.sha256(str(text).encode("utf-8")).digest()
        for col in range(dim):
            byte = seed[col % len(seed)]
            sign = -1.0 if byte % 2 else 1.0
            magnitude = ((byte % 31) + 1) / 31.0
            vectors[row, col] = sign * magnitude
        norm = float(np.linalg.norm(vectors[row]))
        if norm > 0:
            vectors[row] /= norm
    return EmbeddingBatch(
        vectors=vectors,
        model_name=model_name,
        device="deterministic",
        pooling=pooling,
    )


def encode_texts(
    texts: List[str],
    *,
    model_name: str,
    pooling: str = "mean",
    batch_size: int | None = None,
    device: str | None = None,
) -> EmbeddingBatch:
    if model_name in {"deterministic_hash", "surrogate_debug", "debug"}:
        return _deterministic_hash_embeddings(texts, model_name, pooling)

    torch, _, _ = _require_embedding_deps()
    resolved_device = _embedding_device(torch, device)
    tokenizer, model, device = _load_encoder(model_name, resolved_device)
    if not texts:
        hidden_size = int(getattr(model.config, "hidden_size", 768))
        return EmbeddingBatch(
            vectors=np.zeros((0, hidden_size), dtype=np.float32),
            model_name=model_name,
            device=device,
            pooling=pooling,
        )

    env_batch_size = os.environ.get("CFG_STRUCTURAL_EMBEDDING_BATCH_SIZE", "").strip()
    configured_batch_size = int(env_batch_size) if env_batch_size else batch_size
    if configured_batch_size is None:
        configured_batch_size = len(texts)
    configured_batch_size = max(1, int(configured_batch_size))
    batches: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), configured_batch_size):
            encoded = tokenizer(
                texts[start : start + configured_batch_size],
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
            batches.append((summed / counts).detach().cpu().numpy().astype(np.float32))

    vectors = np.concatenate(batches, axis=0) if batches else np.zeros((0, 768), dtype=np.float32)

    return EmbeddingBatch(vectors=vectors, model_name=model_name, device=device, pooling=pooling)


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
    left_norm = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    right_norm = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    return np.matmul(left_norm, right_norm.T).astype(np.float32)
