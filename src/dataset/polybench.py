"""SWE-PolyBench dataset adapter (AmazonScience/SWE-PolyBench)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.types import DatasetLoadResult, ProblemInstance, RepoSnapshot, TestSpec
from src.dataset.registry import register_dataset
from src.dataset.swebench import _lazy_clone

REPOS_ROOT = Path.home() / "polybench_repos"


class PolyBenchDataset:
    name = "polybench"

    def __init__(self, variant: str = "verified") -> None:
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
                    f"No local data_path configured for SWE-PolyBench {variant}. "
                    "Run: python scripts/setup_polybench.py"
                ],
            )
        source = Path(str(data_path))
        if not source.is_absolute():
            source = (Path(__file__).resolve().parents[2] / source).resolve()
        if not source.exists():
            return DatasetLoadResult(
                instances=[],
                errors=[f"SWE-PolyBench source not found: {source}"],
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

            # SWE-PolyBench provides test_command directly (often shell commands).
            test_cmd_raw = row.get("test_command") or ""
            if isinstance(test_cmd_raw, str) and test_cmd_raw.strip():
                # Shell commands — wrap in bash -c for subprocess compatibility.
                test_cmd = ["bash", "-c", test_cmd_raw]
            elif isinstance(test_cmd_raw, list):
                test_cmd = [str(x) for x in test_cmd_raw]
            else:
                lang = row.get("language", "python").lower()
                if lang == "python":
                    test_cmd = ["python3", "-m", "pytest", "-xvs"]
                elif lang == "java":
                    test_cmd = ["mvn", "test", "-pl", "."]
                elif lang in ("javascript", "js", "typescript", "ts"):
                    test_cmd = ["npm", "test"]
                else:
                    test_cmd = ["python3", "-m", "pytest", "-q"]

            instances.append(
                ProblemInstance(
                    dataset=f"polybench_{variant}",
                    split=split,
                    instance_id=iid,
                    prompt=str(row.get("problem_statement") or ""),
                    repo_snapshot=RepoSnapshot(
                        repo_id=repo_id,
                        path=repo_path,
                        base_commit=str(row.get("base_commit") or "unknown"),
                    ),
                    tests=[TestSpec(name="default", command=test_cmd)],
                    metadata={
                        "variant": variant,
                        "source_path": str(source),
                        "language": row.get("language", ""),
                        "task_category": row.get("task_category", ""),
                        "num_nodes": row.get("num_nodes"),
                        "modified_nodes": row.get("modified_nodes", ""),
                    },
                )
            )
            if limit is not None and len(instances) >= max(0, limit):
                break

        return DatasetLoadResult(instances=instances, errors=errors, warnings=[])


@register_dataset("polybench")
class _PolyBenchDefault(PolyBenchDataset):
    def __init__(self) -> None:
        super().__init__(variant="verified")


@register_dataset("polybench_verified")
class _PolyBenchVerified(PolyBenchDataset):
    def __init__(self) -> None:
        super().__init__(variant="verified")


@register_dataset("polybench_500")
class _PolyBench500(PolyBenchDataset):
    def __init__(self) -> None:
        super().__init__(variant="500")


@register_dataset("polybench_full")
class _PolyBenchFull(PolyBenchDataset):
    def __init__(self) -> None:
        super().__init__(variant="full")
