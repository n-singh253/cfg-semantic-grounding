#!/usr/bin/env python3
"""Download and prepare repositories for all three benchmarks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ---- Edit these values for the local repository download. ----

REPOS_ROOT = Path(
    "/proj/arise/arise/hj2742/cfg-semantic-grounding-submission/repos"
)
WORKERS = 32

SWEBENCH_DATASET = "swebench_lite"
FEATUREBENCH_VARIANT = "full"
LIVECODEBENCH_RELEASE = "release_latest"


PROJECT_ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("\n[download_repos] command=" + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    repos_root = REPOS_ROOT.expanduser().resolve()
    repos_root.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "scripts/setup_swebench.py",
            "--dataset",
            SWEBENCH_DATASET,
            "--repos-dir",
            str(repos_root / "SWEBench"),
            "--workers",
            str(WORKERS),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/setup_featurebench.py",
            "--variant",
            FEATUREBENCH_VARIANT,
            "--source-repos-dir",
            str(repos_root / "FeatureBench" / "source_repos"),
            "--repos-dir",
            str(repos_root / "FeatureBench" / "instance_repos"),
            "--workers",
            str(WORKERS),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/setup_livecodebench.py",
            "--release",
            LIVECODEBENCH_RELEASE,
            "--repos-dir",
            str(repos_root / "LiveCodeBench"),
            "--workers",
            str(WORKERS),
        ]
    )

    print(f"\n[download_repos] complete root={repos_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
