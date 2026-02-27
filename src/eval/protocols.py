"""Formal interfaces used by the evaluation runner."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

from src.common.types import DefenseReturn, Patch


class AttackProtocol(Protocol):
    def attack(self, repo_code: Dict[str, Any], ori_prompt: str, all_tests: List[Any]) -> str:
        ...


class DefenseProtocol(Protocol):
    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ) -> DefenseReturn:
        ...


class AgentProtocol(Protocol):
    def agent(self, repo_code: Dict[str, Any], prompt: str, all_tests: List[Any]) -> Patch:
        ...
