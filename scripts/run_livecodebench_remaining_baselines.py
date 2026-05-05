#!/usr/bin/env python3
"""Run remaining LiveCodeBench baselines for Gemini, then Claude.

The script is intentionally resumable: each underlying run_defense command skips
rows already present for the same baseline config hash, and sharded LLM-judge
runs reuse their existing _shards outputs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT = "ucr-ursa-major-congliu-lab"
BASELINE_ROOT = Path("outputs/baselines/livecodebench")
ATTACKS = ("fcv_cwe94", "swexploit_gemini_vertex")
STATIC_BASELINES = ("semgrep", "bandit")
LLAMA_GUARD_BASELINE = "agentic_guard_llama_guard_patch"


def _set_common_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(Path.home() / ".config" / "cfg-semantic-grounding" / "gemini_adc.json"),
    )
    env.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT)
    env.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    env.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    env.setdefault("VERTEXAI_PROJECT", env["GOOGLE_CLOUD_PROJECT"])
    return env


def _gemini_env() -> dict[str, str]:
    env = _set_common_env()
    env.setdefault("VERTEXAI_LOCATION", "global")
    return env


def _claude_env() -> dict[str, str]:
    env = _set_common_env()
    env["VERTEXAI_LOCATION"] = "us-east5"
    env.setdefault("ANTHROPIC_VERTEX_PROJECT_ID", env["GOOGLE_CLOUD_PROJECT"])
    env.setdefault("ANTHROPIC_VERTEX_REGION", "us-east5")
    return env


def _check_credentials(env: dict[str, str]) -> None:
    creds = Path(env["GOOGLE_APPLICATION_CREDENTIALS"])
    if not creds.exists():
        raise FileNotFoundError(f"GOOGLE_APPLICATION_CREDENTIALS does not exist: {creds}")


def _print_env(label: str, env: dict[str, str]) -> None:
    print(f"[livecodebench-baselines] {label} env:")
    for key in [
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "VERTEXAI_PROJECT",
        "VERTEXAI_LOCATION",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "ANTHROPIC_VERTEX_REGION",
    ]:
        if key in env:
            print(f"  {key}={env.get(key)}")


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n[livecodebench-baselines] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _attack_dataset(model_key: str, attack: str) -> Path:
    return Path("outputs/attacks") / model_key / "full" / f"livecodebench_{attack}" / "attack_dataset.jsonl"


def _baseline_out(model_key: str, baseline: str, attack: str) -> Path:
    return BASELINE_ROOT / model_key / baseline / attack


def _run_defense(
    *,
    model_key: str,
    attack: str,
    baseline: str,
    env: dict[str, str],
) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "src.eval.cli",
            "run_defense",
            "--attack-results",
            str(_attack_dataset(model_key, attack)),
            "--baseline",
            baseline,
            "--fidelity-mode",
            "llm",
            "--max-patch-attempts",
            "1",
            "--no-retry-on-apply-failure",
            "--out",
            str(_baseline_out(model_key, baseline, attack)),
        ],
        env,
    )


def _run_defense_sharded(
    *,
    model_key: str,
    attack: str,
    baseline: str,
    shards: int,
    parallel: int,
    env: dict[str, str],
    isolate_repos: bool = False,
    cleanup_repo_copies: bool = False,
) -> None:
    cmd = [
        sys.executable,
        "-u",
        "scripts/run_defense_sharded.py",
        "--attack-results",
        str(_attack_dataset(model_key, attack)),
        "--baseline",
        baseline,
        "--fidelity-mode",
        "llm",
        "--out",
        str(_baseline_out(model_key, baseline, attack)),
        "--shards",
        str(shards),
        "--parallel",
        str(parallel),
        "--stale-timeout-sec",
        "1800",
        "--poll-interval-sec",
        "15",
    ]
    if isolate_repos:
        cmd.append("--isolate-repos")
    if cleanup_repo_copies:
        cmd.append("--cleanup-repo-copies")
    _run(cmd, env)


def _run_gemini() -> None:
    env = _gemini_env()
    _check_credentials(env)
    _print_env("gemini", env)

    # Already-completed FCV semgrep/bandit are skipped by omission; run the
    # remaining non-Llama Gemini work first so the slow local guard is last.
    _run_defense_sharded(
        model_key="gemini3_flash",
        attack="fcv_cwe94",
        baseline="llm_judge_gemini_vertex",
        shards=8,
        parallel=8,
        env=env,
    )

    for baseline in [*STATIC_BASELINES, "llm_judge_gemini_vertex"]:
        if baseline == "llm_judge_gemini_vertex":
            _run_defense_sharded(
                model_key="gemini3_flash",
                attack="swexploit_gemini_vertex",
                baseline=baseline,
                shards=8,
                parallel=8,
                env=env,
            )
        else:
            _run_defense_sharded(
                model_key="gemini3_flash",
                attack="swexploit_gemini_vertex",
                baseline=baseline,
                shards=8,
                parallel=8,
                env=env,
                isolate_repos=True,
                cleanup_repo_copies=True,
            )


def _run_claude() -> None:
    env = _claude_env()
    _check_credentials(env)
    _print_env("claude", env)

    for attack in ATTACKS:
        for baseline in [*STATIC_BASELINES, "llm_judge_claude37_sonnet_vertex"]:
            if baseline == "llm_judge_claude37_sonnet_vertex":
                _run_defense_sharded(
                    model_key="claude37_sonnet_sweagent",
                    attack=attack,
                    baseline=baseline,
                    shards=8,
                    parallel=8,
                    env=env,
                )
            else:
                _run_defense_sharded(
                    model_key="claude37_sonnet_sweagent",
                    attack=attack,
                    baseline=baseline,
                    shards=8,
                    parallel=8,
                    env=env,
                    isolate_repos=True,
                    cleanup_repo_copies=True,
                )


def _run_llama_guard_last() -> None:
    env = _gemini_env()
    _check_credentials(env)
    _print_env("llama_guard", env)

    for model_key in ["gemini3_flash", "claude37_sonnet_sweagent"]:
        for attack in ATTACKS:
            _run_defense(
                model_key=model_key,
                attack=attack,
                baseline=LLAMA_GUARD_BASELINE,
                env=env,
            )


def main() -> int:
    _run_gemini()
    _run_claude()
    _run_llama_guard_last()
    print("\n[livecodebench-baselines] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
