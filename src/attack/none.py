"""Pass-through attack."""

from __future__ import annotations

from typing import Any, Dict, List

from src.attack.base import BaseAttack
from src.attack.registry import register_attack
from src.common.artifact_store import atomic_write_json, atomic_write_text
from src.common.hashing import sha256_text


class NoneAttack(BaseAttack):
    name = "none"

    def attack(self, repo_code: Dict[str, Any], ori_prompt: str, all_tests: List[Any]) -> str:
        _ = all_tests
        artifact_dir = self._attack_artifact_dir(repo_code)
        self.last_metadata = {
            "attack_name": self.name,
            "mode": "pass_through",
            "instance_id": repo_code.get("instance_id"),
            "provider": "none",
            "model": "none",
            "attack_provider": "none",
            "attack_model": "none",
            "prompt_hash": sha256_text(ori_prompt),
            "response_hash": sha256_text(ori_prompt),
            "fidelity_mode": self.fidelity_mode,
            "provider_fallback": self.fidelity_mode == "surrogate_debug",
            "call_count": 0,
            "tool_blocked": False,
            "artifact_path": str(artifact_dir),
        }
        self._write_attack_artifacts(
            repo_code=repo_code,
            original_prompt=ori_prompt,
            adv_prompt=ori_prompt,
            metadata=self.last_metadata,
        )
        # Backward-compatible filenames.
        atomic_write_text(artifact_dir / "prompt.txt", ori_prompt)
        atomic_write_text(artifact_dir / "response.txt", ori_prompt)
        atomic_write_json(artifact_dir / "attack_metadata.json", self.last_metadata)
        return ori_prompt


register_attack("none")(NoneAttack)
