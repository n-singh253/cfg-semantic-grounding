"""Dataset-level structural misalignment training/evaluation plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.baseline.base import BaseDefense
from src.baseline.registry import register_baseline
from src.common.types import DefenseReturn


class StructuralMisalignmentEvalDefense(BaseDefense):
    """Train and evaluate structural misalignment models from graph-build outputs."""

    name = "structural_misalignment_eval"
    scope = "dataset"

    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ) -> DefenseReturn:
        _ = (prompt, code_or_patch, all_tests, repo_code)
        raise RuntimeError("structural_misalignment_eval is dataset-level; use run_dataset")

    def run_dataset(
        self,
        *,
        rows_path: Path | None,
        baseline_name: str,
        baseline_plugin: str,
    ) -> List[Dict[str, Any]]:
        from src.eval.structural_misalignment import run_structural_misalignment_eval

        return run_structural_misalignment_eval(
            data_rows_path=rows_path,
            out_dir=self.run_root,
            config=self.config,
            baseline_name=baseline_name,
            baseline_plugin=baseline_plugin,
            baseline_config_hash=self.baseline_config_hash,
        )


register_baseline("structural_misalignment_eval")(StructuralMisalignmentEvalDefense)
