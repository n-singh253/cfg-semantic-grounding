"""LiveCodeBench dataset adapter for locally materialized code-generation rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.types import DatasetLoadResult, ProblemInstance, RepoSnapshot, TestSpec
from src.dataset.registry import register_dataset


class LiveCodeBenchDataset:
    name = "livecodebench"

    def load(
        self,
        split: str,
        config: Dict[str, Any],
        runtime_dir: Path,
        limit: Optional[int] = None,
        instance_ids: Optional[List[str]] = None,
    ) -> DatasetLoadResult:
        release = str(config.get("release", "release_latest"))
        data_path = config.get("data_path")
        if not data_path:
            return DatasetLoadResult(
                instances=[],
                errors=[],
                warnings=[
                    "No local data_path configured for LiveCodeBench. "
                    "Run: python3 scripts/setup_livecodebench.py"
                ],
            )

        source = Path(str(data_path))
        if not source.is_absolute():
            source = (Path(__file__).resolve().parents[2] / source).resolve()
        if not source.exists():
            return DatasetLoadResult(
                instances=[],
                errors=[f"LiveCodeBench source not found: {source}"],
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

            repo_path = str(row.get("repo_path") or "")
            if not repo_path:
                errors.append(f"{iid}: missing repo_path")
                continue
            rp = Path(repo_path)
            if not rp.exists():
                errors.append(f"{iid}: repo_path does not exist: {rp}")
                continue
            if not (rp / ".git").exists():
                errors.append(f"{iid}: repo_path is not a git checkout: {rp}")
                continue

            test_command = row.get("test_command") or ["python3", "tests/run_public_tests.py"]
            if isinstance(test_command, str):
                test_command = test_command.split()

            instances.append(
                ProblemInstance(
                    dataset="livecodebench",
                    split=split,
                    instance_id=iid,
                    prompt=str(row.get("problem_statement") or ""),
                    repo_snapshot=RepoSnapshot(
                        repo_id=str(row.get("repo_id") or "livecodebench/code_generation_lite"),
                        path=repo_path,
                        base_commit=str(row.get("base_commit") or "HEAD"),
                    ),
                    tests=[TestSpec(name="public", command=[str(x) for x in test_command])],
                    metadata={
                        "release": row.get("release", release),
                        "variant": row.get("variant", "code_generation_lite"),
                        "platform": row.get("platform", ""),
                        "question_id": row.get("question_id", ""),
                        "difficulty": row.get("difficulty", ""),
                        "contest_date": row.get("contest_date", ""),
                        "source_path": str(source),
                        "private_tests_path": row.get("private_tests_path", ""),
                        "private_tests_hash": row.get("private_tests_hash", ""),
                    },
                )
            )
            if limit is not None and len(instances) >= max(0, limit):
                break

        return DatasetLoadResult(instances=instances, errors=errors, warnings=[])


@register_dataset("livecodebench")
class _LiveCodeBenchDefault(LiveCodeBenchDataset):
    pass
