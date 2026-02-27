"""Runnable toy dataset adapter for end-to-end harness validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.types import DatasetLoadResult, ProblemInstance, RepoSnapshot, TestSpec
from src.dataset.registry import register_dataset


class ToyDataset:
    name = "toy"

    def load(
        self,
        split: str,
        config: Dict[str, Any],
        runtime_dir: Path,
        limit: Optional[int] = None,
        instance_ids: Optional[List[str]] = None,
    ) -> DatasetLoadResult:
        declared = config.get("instances") or []
        if not declared:
            declared = [
                {
                    "instance_id": "toy-1",
                    "repo_id": "toy_repo",
                    "base_commit": "toy-base",
                    "prompt": "Fix add(a, b) so tests pass.",
                }
            ]

        out: List[ProblemInstance] = []
        errors: List[str] = []

        for entry in declared:
            iid = str(entry.get("instance_id", "")).strip()
            if not iid:
                errors.append("missing instance_id in toy dataset config")
                continue
            if instance_ids and iid not in set(instance_ids):
                continue

            repo_id = str(entry.get("repo_id", "toy_repo"))
            base_commit = str(entry.get("base_commit", "toy-base"))
            prompt = str(entry.get("prompt", "Fix add(a, b) so tests pass."))

            repo_dir = runtime_dir / "repos" / iid
            self._materialize_repo(repo_dir)

            out.append(
                ProblemInstance(
                    dataset="toy",
                    split=split,
                    instance_id=iid,
                    prompt=prompt,
                    repo_snapshot=RepoSnapshot(
                        repo_id=repo_id,
                        path=str(repo_dir),
                        base_commit=base_commit,
                    ),
                    tests=[
                        TestSpec(
                            name="unittest",
                            command=["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
                        )
                    ],
                    metadata={"source": "toy"},
                )
            )

        if limit is not None:
            out = out[: max(0, limit)]

        return DatasetLoadResult(instances=out, errors=errors, warnings=[])

    @staticmethod
    def _materialize_repo(repo_dir: Path) -> None:
        repo_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / "toy_module.py").write_text(
            "def add(a, b):\n"
            "    return a - b\n",
            encoding="utf-8",
        )
        tests_dir = repo_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "test_toy_module.py").write_text(
            "import unittest\n\n"
            "from toy_module import add\n\n\n"
            "class ToyModuleTest(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 1), 2)\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )


register_dataset("toy")(ToyDataset)
