"""Base protocols/interfaces for parser type safety."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from src.common.llm import LLMClient


class PromptParserProtocol(Protocol):
    """Protocol for prompt/problem statement parsers.
    
    All prompt parsers must accept these parameters and return subtasks + metadata.
    """
    
    def __call__(
        self,
        *,
        llm_client: LLMClient,
        instance_id: str,
        module_name: str,
        module_config_hash: str,
        fidelity_mode: str,
        provider: str,
        model: str,
        problem_statement: str,
        artifact_dir: Path,
        temperature: float,
        seed: Any,
        max_retries: int,
        backoff_sec: float,
        allow_provider_fallback: bool,
        system_prompt: str,
        **kwargs: Any,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Parse problem statement into subtasks.
        
        Returns:
            Tuple of (subtasks, metadata)
            - subtasks: List of subtask strings
            - metadata: Dict with prompt_hash, cache_hit, token_usage, etc.
        """
        ...


class PatchParserProtocol(Protocol):
    """Protocol for patch/diff parsers.
    
    All patch parsers must accept these parameters and return cfg_diff + nodes + diagnostics.
    """
    
    def __call__(
        self,
        patch_text: str,
        *,
        base_repo: Optional[Path],
        allow_hunk_fallback: bool,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """Parse patch into candidate nodes.
        
        Returns:
            Tuple of (cfg_diff, candidate_nodes, diagnostics)
            - cfg_diff: Dict with nodes_added, nodes_removed, nodes_changed, edges_*, summary
            - candidate_nodes: List of dicts with node_id, change_type, file, function, etc.
            - diagnostics: Dict with touched_python_files, fallback_used, etc.
        """
        ...


class LinkerProtocol(Protocol):
    """Protocol for subtask-to-node linkers.
    
    All linkers must accept these parameters and return links + metadata.
    """
    
    def __call__(
        self,
        *,
        llm_client: LLMClient,
        instance_id: str,
        module_name: str,
        module_config_hash: str,
        fidelity_mode: str,
        provider: str,
        model: str,
        problem_statement: str,
        subtasks: List[str],
        candidate_nodes: List[Dict[str, Any]],
        artifact_dir: Path,
        temperature: float,
        seed: Any,
        max_retries: int,
        backoff_sec: float,
        allow_provider_fallback: bool,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Link subtasks to candidate nodes.
        
        Returns:
            Tuple of (links, metadata)
            - links: List of dicts with subtask_index and node_ids
            - metadata: Dict with prompt_hash, cache_hit, token_usage, etc.
        """
        ...
