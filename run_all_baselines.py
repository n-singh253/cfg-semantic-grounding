#!/usr/bin/env python3
"""Local helper for running defense baselines over all prepared row files."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


# ---- Edit these values for local experiment batches. ----

SWEBENCH_REPOS_DIR = Path(
    "/proj/arise/arise/hj2742/cfg-semantic-grounding-submission/repos/SWEBench"
)
FEATUREBENCH_REPOS_DIR = Path(
    "/proj/arise/arise/hj2742/cfg-semantic-grounding-submission/repos/FeatureBench/instance_repos"
)
LIVECODEBENCH_REPOS_DIR = Path(
    "/proj/arise/arise/hj2742/cfg-semantic-grounding-submission/repos/LiveCodeBench"
)

DATA_ROOT = Path(__file__).resolve().parent / "data" / "synthesized_results"

SWEBENCH_ROW_DIRS = [
    DATA_ROOT / "Non-Obfuscated/FCV-78_SWE-Bench_Claude-Sonnet-4.6",
    DATA_ROOT / "Non-Obfuscated/FCV-78_SWE-Bench_MINI-Gemini-3",
    DATA_ROOT / "Non-Obfuscated/FCV_SWE-Bench_OpenHands-Qwen3-Coder-30B",
    DATA_ROOT / "Non-Obfuscated/SWExploit_SWE-Bench_Claude-Sonnet-4.6",
    DATA_ROOT / "Non-Obfuscated/SWExploit_SWE-Bench_MINI-Gemini-3",
    DATA_ROOT / "Non-Obfuscated/SWExploit_SWE-Bench_OpenHands-Qwen3-Coder-30B",
    DATA_ROOT / "Obfuscated/FCV-78_SWE-Bench_MINI-Gemini-3",
    DATA_ROOT / "Obfuscated/FCV-78_SWE-Bench_SWEAgent-Claude-3.7",
    DATA_ROOT / "Obfuscated/FCV_SWE-Bench_OpenHands-Qwen3-Coder-30B",
    DATA_ROOT / "Obfuscated/SWExploit_SWE-Bench_MINI-Gemini-3",
    DATA_ROOT / "Obfuscated/SWExploit_SWE-Bench_OpenHands-Qwen3-Coder-30B",
    DATA_ROOT / "Obfuscated/SWExploit_SWE-Bench_SWEAgent-Claude-3.7",
]

FEATUREBENCH_ROW_DIRS = [
    DATA_ROOT / "Non-Obfuscated/FCV-78_FeatureBench_Claude-Sonnet-4.6",
    DATA_ROOT / "Non-Obfuscated/FCV-78_FeatureBench_MINI-Gemini-3",
    DATA_ROOT / "Non-Obfuscated/FCV_FeatureBench_OpenHands-Qwen3-Coder-30B",
    DATA_ROOT / "Non-Obfuscated/SWExploit_FeatureBench_Claude-Sonnet-4.6",
    DATA_ROOT / "Non-Obfuscated/SWExploit_FeatureBench_MINI-Gemini-3",
    DATA_ROOT / "Non-Obfuscated/SWExploit_FeatureBench_OpenHands-Qwen3-Coder-30B",
    DATA_ROOT / "Obfuscated/FCV-78_FeatureBench_MINI-Gemini-3",
    DATA_ROOT / "Obfuscated/FCV-78_FeatureBench_SWEAgent-Claude-3.7",
    DATA_ROOT / "Obfuscated/FCV_FeatureBench_OpenHands-Qwen3-Coder-30B",
    DATA_ROOT / "Obfuscated/SWExploit_FeatureBench_MINI-Gemini-3",
    DATA_ROOT / "Obfuscated/SWExploit_FeatureBench_OpenHands-Qwen3-Coder-30B",
    DATA_ROOT / "Obfuscated/SWExploit_FeatureBench_SWEAgent-Claude-3.7",
]


LIVECODEBENCH_ROW_DIRS = [
    DATA_ROOT / "Non-Obfuscated/FCV-78_LiveCodeBench_MINI-Gemini-3",
    DATA_ROOT / "Non-Obfuscated/FCV-78_LiveCodeBench_SWEAgent-Claude-3.7",
    DATA_ROOT / "Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B",
    DATA_ROOT / "Non-Obfuscated/SWExploit_LiveCodeBench_MINI-Gemini-3",
    DATA_ROOT / "Non-Obfuscated/SWExploit_LiveCodeBench_OpenHands-Qwen3-Coder-30B",
    DATA_ROOT / "Non-Obfuscated/SWExploit_LiveCodeBench_SWEAgent-Claude-3.7",
    DATA_ROOT / "Obfuscated/FCV-78_LiveCodeBench_MINI-Gemini-3",
    DATA_ROOT / "Obfuscated/FCV-78_LiveCodeBench_SWEAgent-Claude-3.7",
    DATA_ROOT / "Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B",
    DATA_ROOT / "Obfuscated/SWExploit_LiveCodeBench_MINI-Gemini-3",
    DATA_ROOT / "Obfuscated/SWExploit_LiveCodeBench_OpenHands-Qwen3-Coder-30B",
    DATA_ROOT / "Obfuscated/SWExploit_LiveCodeBench_SWEAgent-Claude-3.7",
]


BASELINES = [
    "semgrep",
    "bandit",
]

WORKERS = 48


# ---- Less commonly edited batch controls. ----

OUTPUT_ROOT = Path(
    "/proj/arise/arise/hj2742/cfg-semantic-grounding-submission/results/defense"
)
CONFIG_DIR = Path("configs")
FIDELITY_MODE = "llm"
SCANNER_TIMEOUT_SEC = 600
LIMIT: int | None = None
RESUME = True
DRY_RUN = False
STOP_ON_FAILURE = False


BENCHMARKS = [
    ("swebench", SWEBENCH_REPOS_DIR, SWEBENCH_ROW_DIRS),
]
if "FEATUREBENCH_ROW_DIRS" in globals():
    BENCHMARKS.append(("featurebench", FEATUREBENCH_REPOS_DIR, FEATUREBENCH_ROW_DIRS))
BENCHMARKS.append(("livecodebench", LIVECODEBENCH_REPOS_DIR, LIVECODEBENCH_ROW_DIRS))


def resolve_rows_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path
    rows_path = path / "rows.jsonl"
    if rows_path.exists():
        return rows_path
    synthetic_path = path / "rows_synthetic.jsonl"
    if synthetic_path.exists():
        return synthetic_path
    raise FileNotFoundError(f"No rows.jsonl or rows_synthetic.jsonl under {path}")


def bucket_and_setting(rows_path: Path) -> tuple[str, str]:
    setting_dir = rows_path.parent
    bucket = setting_dir.parent.name
    return bucket, setting_dir.name


def build_command(
    *,
    rows_path: Path,
    baseline: str,
    repos_dir: Path,
    out_dir: Path,
) -> list[str]:
    baseline_workers = 1 if baseline == "llama_guard" else WORKERS
    command = [
        sys.executable,
        "-m",
        "src.eval.cli",
        "run_defense",
        "--rows",
        str(rows_path),
        "--baseline",
        baseline,
        "--fidelity-mode",
        FIDELITY_MODE,
        "--repos-root",
        str(repos_dir),
        "--out",
        str(out_dir),
        "--config-dir",
        str(CONFIG_DIR),
        "--workers",
        str(baseline_workers),
        "--scanner-timeout-sec",
        str(SCANNER_TIMEOUT_SEC),
    ]
    if LIMIT is not None:
        command.extend(["--limit", str(LIMIT)])
    if not RESUME:
        command.append("--no-resume")
    return command


def main() -> int:
    failures: list[tuple[str, Path, str, int]] = []
    total = sum(len(row_dirs) * len(BASELINES) for _, _, row_dirs in BENCHMARKS)
    current = 0
    start_time = time.time()
    child_env = os.environ.copy()
    environment_bin = str(Path(sys.executable).resolve().parent)
    child_env["PATH"] = os.pathsep.join(
        [environment_bin, child_env.get("PATH", "")]
    )

    for benchmark, repos_dir, row_dirs in BENCHMARKS:
        if not repos_dir.exists():
            raise FileNotFoundError(f"{benchmark} repos dir does not exist: {repos_dir}")

        for row_dir in row_dirs:
            rows_path = resolve_rows_path(row_dir)
            bucket, setting = bucket_and_setting(rows_path)

            for baseline in BASELINES:
                current += 1
                out_dir = OUTPUT_ROOT / bucket / setting / baseline
                command = build_command(
                    rows_path=rows_path,
                    baseline=baseline,
                    repos_dir=repos_dir,
                    out_dir=out_dir,
                )

                print(
                    f"\n[run_all_baselines] {current}/{total} "
                    f"benchmark={benchmark} baseline={baseline} rows={rows_path}",
                    flush=True,
                )
                print("[run_all_baselines] command=" + " ".join(command), flush=True)

                if DRY_RUN:
                    continue

                result = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parent,
                    env=child_env,
                )
                if result.returncode != 0:
                    failures.append((benchmark, rows_path, baseline, result.returncode))
                    print(
                        f"[run_all_baselines] failed returncode={result.returncode}",
                        flush=True,
                    )
                    if STOP_ON_FAILURE:
                        return result.returncode

    runtime = round(time.time() - start_time, 3)
    print(
        f"\n[run_all_baselines] complete commands={current} "
        f"failures={len(failures)} runtime_sec={runtime}",
        flush=True,
    )
    for benchmark, rows_path, baseline, returncode in failures:
        print(
            f"[run_all_baselines] failure benchmark={benchmark} "
            f"baseline={baseline} rows={rows_path} returncode={returncode}",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
