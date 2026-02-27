"""Base defense contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from src.common.llm import LLMClient
from src.common.types import DefenseReturn


class BaseDefense(ABC):
    name = "base_defense"
    requires_fit = False

    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: LLMClient,
        baseline_config_hash: str,
        run_root: Path,
        fidelity_mode: str,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.baseline_config_hash = baseline_config_hash
        self.run_root = run_root
        self.fidelity_mode = fidelity_mode
        self.last_signals: Dict[str, Any] = {}

    @abstractmethod
    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ) -> DefenseReturn:
        """Return True/False/edited code_or_prompt string."""
