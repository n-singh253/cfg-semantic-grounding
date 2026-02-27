"""Core orchestration pipeline for run_one and run_matrix."""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import src.agent  # noqa: F401
import src.attack  # noqa: F401
import src.baseline  # noqa: F401
import src.dataset  # noqa: F401
from src.agent.registry import get_agent
from src.attack.registry import get_attack
from src.baseline.registry import get_baseline
from src.common.artifact_store import ArtifactStore, atomic_write_json, atomic_write_text
from src.common.config import config_hash, load_component_config
from src.common.diff import looks_like_unified_diff
from src.common.hashing import sha256_text
from src.common.llm import LLMClient
from src.common.reports import build_dataset_report, build_integration_spec, utc_now
from src.common.subprocess import command_exists
from src.common.types import Patch
from src.dataset.registry import get_dataset
from src.eval.patch_eval import apply_patch, run_repo_tests, run_static_checks
from src.eval.report import load_jsonl_rows, write_summary_csv


SCHEMA_VERSION = "v1"


def _load_configs(
    *,
    config_dir: Path,
    dataset_name: str,
    agent_name: str,
    attack_name: str,
    baseline_name: str,
) -> Dict[str, Dict[str, Any]]:
    return {
        "dataset": load_component_config(config_dir, "datasets", dataset_name),
        "agent": load_component_config(config_dir, "agents", agent_name),
        "attack": load_component_config(config_dir, "attacks", attack_name),
        "baseline": load_component_config(config_dir, "baselines", baseline_name),
    }


def _tool_availability(agent_cfg: Dict[str, Any]) -> Dict[str, bool]:
    flags = {
        "bandit": command_exists("bandit"),
        "semgrep": command_exists("semgrep"),
    }
    command = agent_cfg.get("command")
    if isinstance(command, list) and command:
        flags["agent_cli_available"] = command_exists(str(command[0]))
    else:
        flags["agent_cli_available"] = True
    return flags


def _write_patch_artifacts(
    out_dir: Path,
    instance_id: str,
    ori_patch: Patch,
    adv_patch: Patch,
    final_patch: Optional[Patch],
    apply_status: Dict[str, Any],
) -> Dict[str, str]:
    patch_dir = out_dir / "artifacts" / "patches" / instance_id
    patch_dir.mkdir(parents=True, exist_ok=True)
    ori_path = patch_dir / "ori_patch.diff"
    adv_path = patch_dir / "adv_patch.diff"
    final_path = patch_dir / "final_patch.diff"
    atomic_write_text(ori_path, ori_patch.unified_diff or "")
    atomic_write_text(adv_path, adv_patch.unified_diff or "")
    atomic_write_text(final_path, final_patch.unified_diff if final_patch else "")
    metadata = {
        "ori_patch_hash": sha256_text(ori_patch.unified_diff or ""),
        "adv_patch_hash": sha256_text(adv_patch.unified_diff or ""),
        "final_patch_hash": sha256_text(final_patch.unified_diff if final_patch else ""),
        "apply_status": apply_status,
    }
    atomic_write_json(patch_dir / "metadata.json", metadata)
    return {
        "ori_patch_path": str(ori_path),
        "adv_patch_path": str(adv_path),
        "final_patch_path": str(final_path),
    }


def _run_optional_judges(
    *,
    out_dir: Path,
    llm_client: LLMClient,
    baseline_hash: str,
    fidelity_mode: str,
    instance_id: str,
    prompt: str,
    patch: str,
    tests: List[Any],
    repo_code: Dict[str, Any],
    config_dir: Path,
) -> str:
    try:
        judge_cfg_base = load_component_config(config_dir, "baselines", "llm_judge")
    except FileNotFoundError:
        return ""
    from src.baseline.llm_judge import LLMJudgeDefense

    rows: List[Dict[str, Any]] = []
    for mode in ["raw", "cfg_stats", "prompt_only"]:
        cfg = dict(judge_cfg_base)
        cfg["mode"] = mode
        judge = LLMJudgeDefense(cfg, llm_client, baseline_hash, out_dir, fidelity_mode)
        decision = judge.defense(prompt, patch, tests, repo_code)
        rows.append(
            {
                "mode": mode,
                "decision": "reject" if decision is False else "accept",
                "signals": judge.last_signals,
            }
        )
    path = out_dir / "artifacts" / "judges" / instance_id / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(rows, indent=2, sort_keys=True))
    return str(path)


def run_one(
    *,
    dataset_name: str,
    split: str,
    instance_ids: Optional[Sequence[str]],
    limit: Optional[int],
    agent_name: str,
    attack_name: str,
    baseline_name: str,
    fidelity_mode: str,
    out_dir: Path,
    config_dir: Path,
    cli_invocation: str,
    run_judges: bool = False,
    swexploit_adv_patches: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    store = ArtifactStore(out_dir)
    configs = _load_configs(
        config_dir=config_dir,
        dataset_name=dataset_name,
        agent_name=agent_name,
        attack_name=attack_name,
        baseline_name=baseline_name,
    )
    if swexploit_adv_patches and attack_name == "swexploit":
        configs["attack"] = dict(configs["attack"])
        configs["attack"]["mode"] = "prebuilt_json"
        configs["attack"]["swexploit_adv_patches"] = str(swexploit_adv_patches)
    config_hashes = {name: config_hash(payload) for name, payload in configs.items()}

    existing_rows: List[Dict[str, Any]] = []
    completed_instance_ids: set[str] = set()
    if results_path.exists():
        existing_rows = load_jsonl_rows(results_path)
        for row in existing_rows:
            if (
                row.get("dataset") == dataset_name
                and row.get("agent_config_hash") == config_hashes["agent"]
                and row.get("attack_config_hash") == config_hashes["attack"]
                and row.get("baseline_config_hash") == config_hashes["baseline"]
            ):
                completed_instance_ids.add(str(row.get("instance_id", "")))

    dataset_plugin = str(configs["dataset"].get("plugin", dataset_name))
    agent_plugin = str(configs["agent"].get("plugin", agent_name))
    attack_plugin = str(configs["attack"].get("plugin", attack_name))
    baseline_plugin = str(configs["baseline"].get("plugin", baseline_name))

    dataset_obj = get_dataset(dataset_plugin)()
    llm_client = LLMClient(out_dir / "artifacts" / "llm_cache")
    agent_obj = get_agent(agent_plugin)(configs["agent"])
    attack_obj = get_attack(attack_plugin)(
        configs["attack"], llm_client, config_hashes["attack"], out_dir, fidelity_mode
    )
    baseline_obj = get_baseline(baseline_plugin)(
        configs["baseline"], llm_client, config_hashes["baseline"], out_dir, fidelity_mode
    )

    target_split = split or str(configs["dataset"].get("split", "test"))
    data_result = dataset_obj.load(
        split=target_split,
        config=configs["dataset"],
        runtime_dir=out_dir / "runtime",
        limit=limit,
        instance_ids=list(instance_ids) if instance_ids else None,
    )

    tool_flags = _tool_availability(configs["agent"])
    skipped_instances = sum(1 for inst in data_result.instances if inst.instance_id in completed_instance_ids)
    warnings = list(data_result.warnings)
    if skipped_instances:
        warnings.append(f"resume_detected: skipped {skipped_instances} completed instance(s)")
    dataset_report = build_dataset_report(
        dataset=dataset_name,
        split=target_split,
        total_loaded=len(data_result.instances),
        total_selected=max(0, len(data_result.instances) - skipped_instances),
        skipped=skipped_instances,
        failed=len(data_result.errors),
        failure_reasons=data_result.errors,
        warnings=warnings,
        tool_availability=tool_flags,
    )
    store.write_json("dataset_report.json", dataset_report)

    integration_spec = build_integration_spec(
        run_id=out_dir.name,
        schema_version=SCHEMA_VERSION,
        fidelity_mode=fidelity_mode,
        cli_invocation=cli_invocation,
        selected_plugins={
            "dataset": dataset_plugin,
            "agent": agent_plugin,
            "attack": attack_plugin,
            "baseline": baseline_plugin,
        },
        selected_configs={
            "dataset": {
                "name": dataset_name,
                "hash": config_hashes["dataset"],
                "config": configs["dataset"],
            },
            "agent": {
                "name": agent_name,
                "hash": config_hashes["agent"],
                "config": configs["agent"],
            },
            "attack": {
                "name": attack_name,
                "hash": config_hashes["attack"],
                "config": configs["attack"],
            },
            "baseline": {
                "name": baseline_name,
                "hash": config_hashes["baseline"],
                "config": configs["baseline"],
            },
        },
    )
    store.write_json("integration_spec.json", integration_spec)

    all_rows: List[Dict[str, Any]] = list(existing_rows)
    for instance in data_result.instances:
        if instance.instance_id in completed_instance_ids:
            continue
        t0 = time.time()
        start_ts = utc_now()
        repo_code = {
            "instance_id": instance.instance_id,
            "dataset": instance.dataset,
            "split": instance.split,
            "repo_id": instance.repo_snapshot.repo_id,
            "base_commit": instance.repo_snapshot.base_commit,
            "path": instance.repo_snapshot.path,
            "run_root": str(out_dir),
        }
        repo_path = Path(instance.repo_snapshot.path)
        repo_path.mkdir(parents=True, exist_ok=True)
        ori_prompt = instance.prompt
        ori_prompt_hash = sha256_text(ori_prompt)
        ori_patch = agent_obj.agent(repo_code, ori_prompt, instance.tests)
        adv_prompt = attack_obj.attack(repo_code, ori_prompt, instance.tests)
        adv_prompt_hash = sha256_text(adv_prompt)
        attack_meta_early = dict(getattr(attack_obj, "last_metadata", {}))
        adv_patch = agent_obj.agent(repo_code, adv_prompt, instance.tests)
        prebuilt_adv_patch = attack_meta_early.get("prebuilt_adv_patch")
        if isinstance(prebuilt_adv_patch, str) and prebuilt_adv_patch.strip():
            adv_patch = Patch(
                unified_diff=prebuilt_adv_patch,
                metadata={
                    **adv_patch.metadata,
                    "source": "swexploit_prebuilt_json",
                    "selected_patch_id": attack_meta_early.get("selected_patch_id", ""),
                    "selection_key": attack_meta_early.get("selection_key", ""),
                },
            )
        decision_raw = baseline_obj.defense(adv_prompt, adv_patch.unified_diff, instance.tests, repo_code)
        defense_signals = dict(getattr(baseline_obj, "last_signals", {}))

        defense_decision = "reject"
        final_patch: Optional[Patch] = None
        edited_patch_hash = ""
        apply_ok = False
        apply_message = "rejected"

        if decision_raw is True:
            defense_decision = "accept"
            final_patch = adv_patch
        elif decision_raw is False:
            defense_decision = "reject"
        elif isinstance(decision_raw, str):
            defense_decision = "edit"
            if looks_like_unified_diff(decision_raw):
                final_patch = Patch(unified_diff=decision_raw, metadata={"edited_by_defense": True})
                edited_patch_hash = sha256_text(decision_raw)
            else:
                edited_prompt = decision_raw
                defense_signals["edited_prompt_hash"] = sha256_text(edited_prompt)
                replacement_patch = agent_obj.agent(repo_code, edited_prompt, instance.tests)
                final_patch = replacement_patch
                edited_patch_hash = sha256_text(replacement_patch.unified_diff or "")
        else:
            defense_decision = "reject"

        test_log_path = out_dir / "logs" / "tests" / f"{instance.instance_id}.log"
        static_path = out_dir / "logs" / "static" / f"{instance.instance_id}.json"
        tests_passed = False
        static_findings = 0
        static_signals: Dict[str, Any] = {"skipped": True}
        if final_patch is not None:
            apply_ok, apply_message = apply_patch(repo_path, final_patch.unified_diff or "")
            if apply_ok:
                tests_passed, _ = run_repo_tests(instance.tests, repo_path, test_log_path)
            else:
                test_log_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(test_log_path, f"Patch apply failed: {apply_message}\n")
            static_findings, _, static_signals = run_static_checks(repo_path, static_path)
        else:
            test_log_path.parent.mkdir(parents=True, exist_ok=True)
            static_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(test_log_path, "Patch rejected before apply.\n")
            atomic_write_text(static_path, json.dumps({"skipped": True}, indent=2))

        patch_paths = _write_patch_artifacts(
            out_dir=out_dir,
            instance_id=instance.instance_id,
            ori_patch=ori_patch,
            adv_patch=adv_patch,
            final_patch=final_patch,
            apply_status={"applied": apply_ok, "message": apply_message},
        )

        judge_artifact_path = ""
        if run_judges and final_patch is not None:
            judge_artifact_path = _run_optional_judges(
                out_dir=out_dir,
                llm_client=llm_client,
                baseline_hash=config_hashes["baseline"],
                fidelity_mode=fidelity_mode,
                instance_id=instance.instance_id,
                prompt=adv_prompt,
                patch=final_patch.unified_diff or "",
                tests=instance.tests,
                repo_code=repo_code,
                config_dir=config_dir,
            )

        attack_meta = dict(getattr(attack_obj, "last_metadata", {}))
        row = {
            "dataset": instance.dataset,
            "split": instance.split,
            "instance_id": instance.instance_id,
            "repo_id": instance.repo_snapshot.repo_id,
            "base_commit": instance.repo_snapshot.base_commit,
            "agent_name": agent_name,
            "agent_config_hash": config_hashes["agent"],
            "attack_name": attack_name,
            "attack_config_hash": config_hashes["attack"],
            "attack_mode": attack_meta.get("mode", ""),
            "baseline_name": baseline_name,
            "baseline_config_hash": config_hashes["baseline"],
            "fidelity_mode": fidelity_mode,
            "attack_provider": attack_meta.get("provider", "none"),
            "attack_model": attack_meta.get("model", "none"),
            "attack_prompt_hash": attack_meta.get("prompt_hash", ""),
            "attack_response_hash": attack_meta.get("response_hash", ""),
            "attack_artifact_path": attack_meta.get("artifact_path", ""),
            "attack_cache_hit": bool(attack_meta.get("cache_hit", False)),
            "attack_cache_key": attack_meta.get("cache_key", ""),
            "attack_provider_fallback": bool(attack_meta.get("provider_fallback", False)),
            "attack_tool_blocked": bool(attack_meta.get("tool_blocked", False)),
            "attack_token_usage": attack_meta.get("token_usage", {}),
            "attack_objective_tags": attack_meta.get("objective_tags", []),
            "attack_stealth_constraints": attack_meta.get("stealth_constraints", []),
            "attack_selected_patch_id": attack_meta.get("selected_patch_id", ""),
            "attack_source_json_path": attack_meta.get("source_json_path", ""),
            "attack_source_json_hash": attack_meta.get("source_json_hash", ""),
            "original_prompt_hash": ori_prompt_hash,
            "adv_prompt_hash": adv_prompt_hash,
            "ori_patch_hash": sha256_text(ori_patch.unified_diff or ""),
            "adv_patch_hash": sha256_text(adv_patch.unified_diff or ""),
            "patch_hash": sha256_text(adv_patch.unified_diff or ""),
            "edited_patch_hash": edited_patch_hash,
            "ori_agent_metadata": ori_patch.metadata,
            "adv_agent_metadata": adv_patch.metadata,
            "defense_decision": defense_decision,
            "defense_signals": defense_signals,
            "tests_passed": bool(tests_passed),
            "test_log_path": str(test_log_path),
            "static_findings_count": int(static_findings),
            "static_findings_path": str(static_path),
            "static_tool_signals": static_signals,
            "patch_artifacts": patch_paths,
            "judge_artifact_path": judge_artifact_path,
            "timestamp_start": start_ts,
            "timestamp_end": utc_now(),
            "runtime_sec": round(time.time() - t0, 6),
        }
        store.append_jsonl("results.jsonl", row)
        all_rows.append(row)
        completed_instance_ids.add(instance.instance_id)
    return all_rows


def run_matrix(
    *,
    matrix_config: Dict[str, Any],
    out_dir: Path,
    config_dir: Path,
    cli_invocation: str,
) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = matrix_config.get("datasets") or [matrix_config.get("dataset", "toy")]
    agents = matrix_config.get("agents") or ["dummy"]
    attacks = matrix_config.get("attacks") or ["none"]
    baselines = matrix_config.get("baselines") or ["prompt_filter"]
    fidelity_modes = matrix_config.get("fidelity_modes") or [matrix_config.get("fidelity_mode", "llm")]
    split = str(matrix_config.get("split", "test"))
    limit = matrix_config.get("limit")
    run_judges = bool(matrix_config.get("run_judges", False))
    swexploit_adv_patches = matrix_config.get("swexploit_adv_patches")

    rows: List[Dict[str, Any]] = []
    for idx, (dataset, agent, attack, baseline, fidelity_mode) in enumerate(
        product(datasets, agents, attacks, baselines, fidelity_modes),
        start=1,
    ):
        run_name = f"{idx:03d}_{dataset}_{agent}_{attack}_{baseline}_{fidelity_mode}"
        run_out = out_dir / run_name
        rows.extend(
            run_one(
                dataset_name=str(dataset),
                split=split,
                instance_ids=None,
                limit=int(limit) if isinstance(limit, int) else None,
                agent_name=str(agent),
                attack_name=str(attack),
                baseline_name=str(baseline),
                fidelity_mode=str(fidelity_mode),
                out_dir=run_out,
                config_dir=config_dir,
                cli_invocation=cli_invocation,
                run_judges=run_judges,
                swexploit_adv_patches=str(swexploit_adv_patches) if swexploit_adv_patches else None,
            )
        )

    write_summary_csv(out_dir / "summary.csv", rows)
    return rows


def collect_matrix_rows(out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(out_dir.glob("*/results.jsonl")):
        rows.extend(load_jsonl_rows(path))
    return rows
