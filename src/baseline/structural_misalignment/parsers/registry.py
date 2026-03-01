"""Registries for pluggable parser components."""

from __future__ import annotations

from src.common.registry import Registry

# Three registries for each customizable stage
PROMPT_PARSER_REGISTRY = Registry(kind="prompt_parser")
PATCH_PARSER_REGISTRY = Registry(kind="patch_parser")
LINKER_REGISTRY = Registry(kind="linker")


def register_prompt_parser(name: str):
    """Register a prompt/problem statement parser.
    
    Expected signature:
        (*, llm_client, instance_id, module_name, module_config_hash, fidelity_mode,
         provider, model, problem_statement, artifact_dir, temperature, seed,
         max_retries, backoff_sec, allow_provider_fallback, system_prompt, **kwargs)
        -> Tuple[List[str], Dict[str, Any]]
    
    Returns:
        - List[str]: Subtask strings
        - Dict[str, Any]: Metadata with keys like prompt_hash, cache_hit, token_usage, etc.
    """
    return PROMPT_PARSER_REGISTRY.register(name)


def register_patch_parser(name: str):
    """Register a patch/diff parser.
    
    Expected signature:
        (patch_text: str, *, base_repo: Optional[Path], allow_hunk_fallback: bool, **kwargs)
        -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]
    
    Returns:
        - Dict[str, Any]: CFG diff with nodes_added, nodes_removed, nodes_changed, edges_*, summary
        - List[Dict[str, Any]]: Candidate nodes with node_id, change_type, file, function, etc.
        - Dict[str, Any]: Diagnostics with touched_python_files, fallback_used, etc.
    """
    return PATCH_PARSER_REGISTRY.register(name)


def register_linker(name: str):
    """Register a subtask-to-node linker.
    
    Expected signature:
        (*, llm_client, instance_id, module_name, module_config_hash, fidelity_mode,
         provider, model, problem_statement, subtasks, candidate_nodes, artifact_dir,
         temperature, seed, max_retries, backoff_sec, allow_provider_fallback, **kwargs)
        -> Tuple[List[Dict[str, Any]], Dict[str, Any]]
    
    Returns:
        - List[Dict[str, Any]]: Links with subtask_index and node_ids
        - Dict[str, Any]: Metadata with prompt_hash, cache_hit, token_usage, etc.
    """
    return LINKER_REGISTRY.register(name)


def get_prompt_parser(name: str):
    """Get registered prompt parser by name."""
    return PROMPT_PARSER_REGISTRY.get(name)


def get_patch_parser(name: str):
    """Get registered patch parser by name."""
    return PATCH_PARSER_REGISTRY.get(name)


def get_linker(name: str):
    """Get registered linker by name."""
    return LINKER_REGISTRY.get(name)


def list_prompt_parsers():
    """List all registered prompt parsers."""
    return list(PROMPT_PARSER_REGISTRY.names())


def list_patch_parsers():
    """List all registered patch parsers."""
    return list(PATCH_PARSER_REGISTRY.names())


def list_linkers():
    """List all registered linkers."""
    return list(LINKER_REGISTRY.names())
