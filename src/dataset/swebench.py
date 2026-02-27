"""SWE-Bench dataset adapters (Lite/Pro/Plus)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.types import DatasetLoadResult, ProblemInstance, RepoSnapshot, TestSpec
from src.dataset.registry import register_dataset


class SWEBenchDataset:
    name = "swebench"
    variant = "lite"

    def __init__(self, variant: str = "lite") -> None:
        self.variant = variant

    def load(
        self,
        split: str,
        config: Dict[str, Any],
        runtime_dir: Path,
        limit: Optional[int] = None,
        instance_ids: Optional[List[str]] = None,
    ) -> DatasetLoadResult:
        variant = str(config.get("variant", self.variant))
        data_path = config.get("data_path")
        if not data_path:
            return DatasetLoadResult(
                instances=[],
                errors=[],
                warnings=[
                    f"No local data_path configured for SWE-Bench {variant}. "
                    "Provide configs/datasets/*.yaml:data_path for real loading."
                ],
            )
        source = Path(str(data_path))
        if not source.exists():
            return DatasetLoadResult(
                instances=[],
                errors=[f"SWE-Bench source not found: {source}"],
                warnings=[],
            )

        rows: List[Dict[str, Any]] = []
        if source.suffix.lower() == ".jsonl":
            for line in source.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        else:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                rows = list(payload.get("rows", []))
            elif isinstance(payload, list):
                rows = payload

        wanted_ids = set(instance_ids or [])
        instances: List[ProblemInstance] = []
        errors: List[str] = []
        for idx, row in enumerate(rows):
            iid = str(row.get("instance_id") or row.get("id") or f"row-{idx}")
            if wanted_ids and iid not in wanted_ids:
                continue
            repo_path = str(row.get("repo_path") or "")
            if not repo_path:
                errors.append(f"{iid}: missing repo_path")
                continue
            test_cmd = row.get("test_command") or ["python3", "-m", "pytest", "-q"]
            if isinstance(test_cmd, str):
                test_cmd = test_cmd.split()
            instances.append(
                ProblemInstance(
                    dataset=f"swebench_{variant}",
                    split=split,
                    instance_id=iid,
                    prompt=str(row.get("problem_statement") or row.get("prompt") or ""),
                    repo_snapshot=RepoSnapshot(
                        repo_id=str(row.get("repo_id") or row.get("repo") or iid.split("::")[0]),
                        path=repo_path,
                        base_commit=str(row.get("base_commit") or "unknown"),
                    ),
                    tests=[TestSpec(name="default", command=[str(x) for x in test_cmd])],
                    metadata={
                        "variant": variant,
                        "source_path": str(source),
                    },
                )
            )
            if limit is not None and len(instances) >= max(0, limit):
                break

        return DatasetLoadResult(instances=instances, errors=errors, warnings=[])


@register_dataset("swebench")
class _SWEBenchDefault(SWEBenchDataset):
    """Default SWE-Bench adapter alias."""


@register_dataset("swebench_lite")
class _SWEBenchLite(SWEBenchDataset):
    def __init__(self) -> None:
        super().__init__(variant="lite")


@register_dataset("swebench_pro")
class _SWEBenchPro(SWEBenchDataset):
    def __init__(self) -> None:
        super().__init__(variant="pro")


@register_dataset("swebench_plus")
class _SWEBenchPlus(SWEBenchDataset):
    def __init__(self) -> None:
        super().__init__(variant="plus")
