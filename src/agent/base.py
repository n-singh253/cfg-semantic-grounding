"""Base class for patch-generation agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.common.types import Patch


class BaseAgent(ABC):
    name = "base_agent"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def agent(self, repo_code: Dict[str, Any], prompt: str, all_tests: List[Any]) -> Patch:
        """Generate a unified diff patch."""
