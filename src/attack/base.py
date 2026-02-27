"""Base classes for attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.artifact_store import atomic_write_json, atomic_write_text
from src.common.llm import LLMClient


class BaseAttack(ABC):
    name = "base_attack"

    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: LLMClient,
        attack_config_hash: str,
        run_root: Path,
        fidelity_mode: str,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.attack_config_hash = attack_config_hash
        self.run_root = run_root
        self.fidelity_mode = fidelity_mode
        self.last_metadata: Dict[str, Any] = {}

    @abstractmethod
    def attack(self, repo_code: Dict[str, Any], ori_prompt: str, all_tests: List[Any]) -> str:
        """Return adversarial prompt string."""

    def _attack_artifact_dir(self, repo_code: Dict[str, Any]) -> Path:
        instance_id = str(repo_code.get("instance_id", "unknown"))
        return self.run_root / "artifacts" / "attacks" / instance_id / self.name

    def _write_attack_artifacts(
        self,
        *,
        repo_code: Dict[str, Any],
        original_prompt: str,
        adv_prompt: str,
        metadata: Dict[str, Any],
        final_prompt_filename: Optional[str] = None,
    ) -> Path:
        artifact_dir = self._attack_artifact_dir(repo_code)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(metadata)
        payload["artifact_path"] = str(artifact_dir)

        atomic_write_text(artifact_dir / "original_prompt.txt", original_prompt)
        atomic_write_text(artifact_dir / "adv_prompt.txt", adv_prompt)
        if final_prompt_filename:
            atomic_write_text(artifact_dir / final_prompt_filename, adv_prompt)
        atomic_write_json(artifact_dir / "metadata.json", payload)
        return artifact_dir
