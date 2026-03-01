"""LLM-based patch parser using JSON output."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.baseline.structural_misalignment.parsers.registry import register_patch_parser
from src.common.llm import LLMClient


def _build_patch_parsing_prompt(patch_text: str) -> str:
    """Build prompt for LLM to parse patch into chunks/nodes."""
    return (
        "You are parsing a unified diff patch to extract code chunks that represent changed or added code.\n\n"
        "Task:\n"
        "- Extract each meaningful code chunk from the patch\n"
        "- For each chunk, identify: file path, function name, line range, change type, and code snippet\n"
        "- Return strict JSON format only\n\n"
        "Output format:\n"
        "{\n"
        '  "chunks": [\n'
        "    {\n"
        '      "file": "path/to/file.py",\n'
        '      "function": "function_name",\n'
        '      "start_line": 10,\n'
        '      "end_line": 20,\n'
        '      "change_type": "added|modified|removed",\n'
        '      "code_snippet": "actual code here"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Only extract Python (.py) files\n"
        "- change_type is 'added' for new lines (+), 'removed' for deleted lines (-), 'modified' for changes\n"
        "- Infer function name from context or use 'unknown'\n"
        "- code_snippet should contain the actual code (without diff markers like + or -)\n"
        "- start_line/end_line should be approximate line numbers from hunk headers\n\n"
        f"Patch to parse:\n{patch_text}"
    )


def _parse_chunks_response(text: str) -> List[Dict[str, Any]]:
    """Parse LLM response to extract chunks."""
    import json
    
    raw = (text or "").strip()
    if not raw:
        return []
    
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "chunks" in parsed:
            chunks = parsed["chunks"]
            if isinstance(chunks, list):
                return chunks
    except json.JSONDecodeError:
        pass
    
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict) and "chunks" in parsed:
                chunks = parsed["chunks"]
                if isinstance(chunks, list):
                    return chunks
        except json.JSONDecodeError:
            pass
    
    return []


def _chunks_to_candidate_nodes(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert LLM chunks to candidate node format."""
    nodes: List[Dict[str, Any]] = []
    
    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        
        file_path = str(chunk.get("file", "")).strip()
        function = str(chunk.get("function", "unknown")).strip()
        start_line = int(chunk.get("start_line", 0))
        end_line = int(chunk.get("end_line", start_line))
        change_type = str(chunk.get("change_type", "modified")).lower()
        code_snippet = str(chunk.get("code_snippet", "")).strip()
        
        if not file_path or not code_snippet:
            continue
        
        node_id = f"{file_path}::{function}::chunk_{idx}"
        
        code_hash = hashlib.md5(code_snippet.encode("utf-8")).hexdigest()[:8]
        
        nodes.append({
            "node_id": node_id,
            "change_type": change_type,
            "file": file_path,
            "function": function,
            "start_line": start_line,
            "end_line": end_line,
            "node_type": "code_chunk",
            "code_snippet": code_snippet,
            "code_hash": code_hash,
            "contains_calls": [],  # LLM doesn't extract this, leave empty
        })
    
    return nodes


def _create_cfg_diff_from_nodes(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a cfg_diff structure from candidate nodes."""
    nodes_added = []
    nodes_changed = []
    nodes_removed = []
    
    for node in nodes:
        change_type = node.get("change_type", "modified")
        if change_type == "added":
            nodes_added.append({
                "file": node["file"],
                "function": node["function"],
                "node": node
            })
        elif change_type == "removed":
            nodes_removed.append({
                "file": node["file"],
                "function": node["function"],
                "node": node
            })
        else:  # modified
            nodes_changed.append({
                "change_type": "modified",
                "file": node["file"],
                "function": node["function"],
                "before": {},  # LLM doesn't provide before state
                "after": node
            })
    
    return {
        "nodes_added": nodes_added,
        "nodes_removed": nodes_removed,
        "nodes_changed": nodes_changed,
        "edges_added": [],
        "edges_removed": [],
        "summary": {
            "total_nodes_before": 0,
            "total_nodes_after": len(nodes),
            "total_edges_before": 0,
            "total_edges_after": 0,
            "files_compared": len(set(n["file"] for n in nodes)),
            "functions_compared": len(set(f"{n['file']}::{n['function']}" for n in nodes)),
        }
    }


def llm_chunks_parser(
    patch_text: str,
    *,
    base_repo: Optional[Path],
    allow_hunk_fallback: bool,
    config: Dict[str, Any],
    artifact_dir: Path,
    llm_client: Optional[LLMClient] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Parse patch into chunks using LLM.
    
    This parser uses LLM to extract code chunks from a unified diff patch,
    returning them in the same format as cfg_ast_parser.
    
    Args:
        patch_text: Unified diff format patch
        base_repo: Base repository path (not used, for compatibility)
        allow_hunk_fallback: Whether to use fallback (not used, for compatibility)
        config: Configuration dict with llm settings
        artifact_dir: Directory to save artifacts
        llm_client: Shared LLM client (required)
        **kwargs: Additional arguments
    
    Returns:
        Tuple of (cfg_diff, candidate_nodes, diagnostics)
        - cfg_diff: Dict with nodes_added, nodes_removed, nodes_changed, edges_*, summary
        - candidate_nodes: List of dicts with node_id, change_type, file, function, etc.
        - diagnostics: Dict with parsing metadata
    
    Raises:
        ValueError: If llm_client is not provided
    """
    if llm_client is None:
        raise ValueError("llm_chunks_parser requires llm_client to be provided")
    
    llm_cfg = config.get("llm", {}) if isinstance(config.get("llm"), dict) else {}
    provider = str(llm_cfg.get("provider", config.get("provider", "openai")))
    model = str(llm_cfg.get("model", config.get("model", "gpt-4o-mini")))
    temperature = float(llm_cfg.get("temperature", config.get("temperature", 0.2)))
    seed = llm_cfg.get("seed", config.get("seed"))
    max_retries = int(llm_cfg.get("max_retries", config.get("max_retries", 2)))
    backoff_sec = float(llm_cfg.get("backoff_sec", config.get("backoff_sec", 1.0)))
    allow_provider_fallback = bool(llm_cfg.get("allow_provider_fallback", config.get("allow_provider_fallback", False)))
    
    instance_id = str(kwargs.get("instance_id", config.get("instance_id", "unknown")))
    module_name = str(kwargs.get("module_name", config.get("plugin", "structural_misalignment")))
    module_config_hash = str(kwargs.get("module_config_hash", ""))
    fidelity_mode = str(kwargs.get("fidelity_mode", config.get("fidelity_mode", "full")))
    
    prompt = _build_patch_parsing_prompt(patch_text)
    
    artifact_path = artifact_dir / "llm_patch_parsing"
    artifact_path.mkdir(parents=True, exist_ok=True)
    
    result = llm_client.generate(
        instance_id=instance_id,
        module_kind="defense",
        module_name=f"{module_name}_patch_parsing",
        module_config_hash=module_config_hash,
        fidelity_mode=fidelity_mode,
        provider=provider,
        model=model,
        prompt=prompt,
        artifact_dir=artifact_path,
        temperature=temperature,
        seed=int(seed) if isinstance(seed, int) else None,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
        allow_provider_fallback=allow_provider_fallback,
        surrogate_fn=lambda _: '{"chunks":[]}',
    )
    
    chunks = _parse_chunks_response(result.text)
    candidate_nodes = _chunks_to_candidate_nodes(chunks)
    cfg_diff = _create_cfg_diff_from_nodes(candidate_nodes)
    
    diagnostics = {
        "parser_type": "llm_chunks",
        "chunk_count": len(chunks),
        "candidate_node_count": len(candidate_nodes),
        "touched_python_files": len(set(n["file"] for n in candidate_nodes)),
        "fallback_used": False,
        "llm_metadata": result.to_dict(),
    }
    
    return cfg_diff, candidate_nodes, diagnostics


# Register as LLM-based patch parser
register_patch_parser("llm_chunks")(llm_chunks_parser)
