
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.baseline.structural_misalignment.parsers.registry import register_linker


def _check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _compute_embeddings(texts: List[str], device: str) -> Any:
    """Compute BERT embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Tensor of embeddings (N x hidden_size)

    Raises:
        RuntimeError: If transformers or torch are not installed
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        raise RuntimeError(
            "embedding_similarity_linker requires 'transformers' and 'torch' packages. "
            "Install with: pip install transformers torch"
        ) from e
    
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings


def _compute_cosine_similarity(embeddings1: Any, embeddings2: Any) -> Any:
    """Compute cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1: Tensor of shape (N x D)
        embeddings2: Tensor of shape (M x D)
    
    Returns:
        Similarity matrix of shape (N x M)

    Raises:
        RuntimeError: If torch is not installed
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("embedding_similarity_linker requires 'torch' package") from e
    
    embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
    embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
    
    similarity = torch.mm(embeddings1, embeddings2.T)
    
    return similarity


def embedding_similarity_linker(
    *,
    subtasks: List[str],
    candidate_nodes: List[Dict[str, Any]],
    artifact_dir: Path,
    similarity_threshold: float = 0.3,
    **kwargs: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Link subtasks to nodes using BERT embeddings and cosine similarity.
    
    This linker:
    1. Computes BERT embeddings for subtasks and node code snippets
    2. Computes cosine similarity between all pairs
    3. Creates links where similarity > threshold (default 0.3)
    4. Requires CUDA to be available, otherwise raises error
    
    Args:
        subtasks: List of subtask strings from prompt parser
        candidate_nodes: List of candidate nodes from patch parser
        artifact_dir: Directory to save artifacts
        similarity_threshold: Minimum similarity to create a link (default 0.3)
        **kwargs: Additional arguments (ignored for extensibility)
    
    Returns:
        Tuple of (links, metadata)
        - links: List of dicts with subtask_index and node_ids
        - metadata: Dict with embedding_model, device, similarity_stats, etc.
    """
    cuda_available = _check_cuda_available()
    if not cuda_available:
        raise RuntimeError(
            "embedding_similarity_linker requires CUDA to be available. "
            "CUDA not detected. Please ensure you have a GPU and PyTorch with CUDA support installed. "
            "To catch this error early, set 'requires_gpu: true' in your config."
        )
    
    device = "cuda"
    
    if not subtasks or not candidate_nodes:
        links = [{"subtask_index": i, "node_ids": []} for i in range(len(subtasks))]
        metadata = {
            "embedding_model": "bert-base-uncased",
            "device": device,
            "similarity_threshold": similarity_threshold,
            "subtask_count": len(subtasks),
            "node_count": len(candidate_nodes),
            "link_count": 0,
            "warning": "No subtasks or nodes to link"
        }
        return links, metadata
    
    subtask_texts = subtasks
    node_texts = []
    node_ids = []
    
    for node in candidate_nodes:
        node_id = str(node.get("node_id", ""))
        code_snippet = str(node.get("code_snippet", "")).strip()
        
        file_path = str(node.get("file", ""))
        function = str(node.get("function", ""))
        context = f"File: {file_path}, Function: {function}\nCode: {code_snippet}"
        
        node_texts.append(context)
        node_ids.append(node_id)
    
    try:
        subtask_embeddings = _compute_embeddings(subtask_texts, device)
        node_embeddings = _compute_embeddings(node_texts, device)
        
        similarity_matrix = _compute_cosine_similarity(subtask_embeddings, node_embeddings)
        
        import torch
        similarity_matrix = similarity_matrix.cpu().numpy()
        
    except Exception as e:
        # If embedding computation fails, return empty links
        links = [{"subtask_index": i, "node_ids": []} for i in range(len(subtasks))]
        metadata = {
            "embedding_model": "bert-base-uncased",
            "device": device,
            "similarity_threshold": similarity_threshold,
            "error": str(e),
            "subtask_count": len(subtasks),
            "node_count": len(candidate_nodes),
            "link_count": 0,
        }
        return links, metadata
    
    links = []
    total_links = 0
    similarity_stats = {
        "min": float(similarity_matrix.min()),
        "max": float(similarity_matrix.max()),
        "mean": float(similarity_matrix.mean()),
    }
    
    for subtask_idx in range(len(subtasks)):
        linked_node_ids = []
        
        for node_idx in range(len(candidate_nodes)):
            similarity = similarity_matrix[subtask_idx, node_idx]
            
            if similarity >= similarity_threshold:
                linked_node_ids.append(node_ids[node_idx])
                total_links += 1
        
        links.append({
            "subtask_index": subtask_idx,
            "node_ids": linked_node_ids
        })
    
    artifact_path = artifact_dir / "similarity_matrix.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import numpy as np
        similarity_data = {
            "subtasks": subtask_texts,
            "node_ids": node_ids,
            "similarity_matrix": similarity_matrix.tolist(),
            "threshold": similarity_threshold,
        }
        with open(artifact_path, "w") as f:
            json.dump(similarity_data, f, indent=2)
    except Exception:
        pass  
    
    metadata = {
        "embedding_model": "bert-base-uncased",
        "device": device,
        "similarity_threshold": similarity_threshold,
        "subtask_count": len(subtasks),
        "node_count": len(candidate_nodes),
        "link_count": total_links,
        "similarity_stats": similarity_stats,
        "artifact_path": str(artifact_path) if artifact_path.exists() else None,
    }
    
    return links, metadata


# Register as embedding-based linker
register_linker("embedding_similarity")(embedding_similarity_linker)
