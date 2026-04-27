"""FeatureBench dataset adapter (LiberCoders/FeatureBench)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.types import DatasetLoadResult, ProblemInstance, RepoSnapshot, TestSpec
from src.dataset.registry import register_dataset
from src.dataset.swebench import _lazy_clone

REPOS_ROOT = Path.home() / "featurebench_repos"


class FeatureBenchDataset:
    name = "featurebench"

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
                    f"No local data_path configured for FeatureBench {variant}. "
                    "Run: python scripts/setup_featurebench.py"
                ],
            )
        source = Path(str(data_path))
        if not source.is_absolute():
            source = (Path(__file__).resolve().parents[2] / source).resolve()
        if not source.exists():
            return DatasetLoadResult(
                instances=[],
                errors=[f"FeatureBench source not found: {source}"],
                warnings=[],
            )

        rows: List[Dict[str, Any]] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))

        wanted_ids = set(instance_ids or [])
        instances: List[ProblemInstance] = []
        errors: List[str] = []
        for idx, row in enumerate(rows):
            iid = str(row.get("instance_id") or f"row-{idx}")
            if wanted_ids and iid not in wanted_ids:
                continue
            repo_id = str(row.get("repo_id") or row.get("repo") or "")
            repo_path = str(row.get("repo_path") or "")
            if not repo_path:
                errors.append(f"{iid}: missing repo_path")
                continue

            rp = Path(repo_path)
            if not (rp / ".git").exists() and repo_id:
                if not _lazy_clone(repo_id, rp):
                    errors.append(f"{iid}: failed to clone {repo_id}")
                    continue

            test_cmd = row.get("test_command") or ["python3", "-m", "pytest", "-xvs"]
            if isinstance(test_cmd, str):
                test_cmd = test_cmd.split()
            fail_to_pass = row.get("FAIL_TO_PASS") or []
            if isinstance(fail_to_pass, list) and fail_to_pass:
                test_cmd = ["python3", "-m", "pytest", "-xvs"] + fail_to_pass

            instances.append(
                ProblemInstance(
                    dataset=f"featurebench_{variant}",
                    split=split,
                    instance_id=iid,
                    prompt=str(row.get("problem_statement") or ""),
                    repo_snapshot=RepoSnapshot(
                        repo_id=repo_id,
                        path=repo_path,
                        base_commit=str(row.get("base_commit") or "unknown"),
                    ),
                    tests=[TestSpec(name="default", command=[str(x) for x in test_cmd])],
                    metadata={
                        "variant": variant,
                        "source_path": str(source),
                        "image_name": row.get("image_name", ""),
                    },
                )
            )
            if limit is not None and len(instances) >= max(0, limit):
                break

        return DatasetLoadResult(instances=instances, errors=errors, warnings=[])


@register_dataset("featurebench")
class _FeatureBenchDefault(FeatureBenchDataset):
    def __init__(self) -> None:
        super().__init__(variant="lite")


@register_dataset("featurebench_lite")
class _FeatureBenchLite(FeatureBenchDataset):
    def __init__(self) -> None:
        super().__init__(variant="lite")


@register_dataset("featurebench_full")
class _FeatureBenchFull(FeatureBenchDataset):
    def __init__(self) -> None:
        super().__init__(variant="full")
