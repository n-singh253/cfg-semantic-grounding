#!/usr/bin/env python3
"""Import external SWE-Bench preds.json patches into the attack dataset pipeline."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import src.dataset  # noqa: F401 - register dataset plugins
from src.common.artifact_store import atomic_write_json, atomic_write_text
from src.common.config import config_hash, load_component_config
from src.common.hashing import sha256_text
from src.dataset.registry import get_dataset
from src.eval.attack_finalize import finalize_attack_dataset, require_finalized_attack_rows


DEFAULT_ZIP_PATH = Path("swe_bench_non_obfuscated.zip")


@dataclass(frozen=True)
class ImportSpec:
    model_name: str
    model_key: str
    agent_name: str
    attack_name: str
    preds_path: str


DEFAULT_SPECS = {
    ("gemini", "none"): ImportSpec(
        model_name="gemini",
        model_key="gemini3_flash_swebench_imported",
        agent_name="minisweagent_gemini3_flash",
        attack_name="none",
        preds_path="base____NoneAttackLLM/lite/mini_swe____gemini-2.0-flash/NoPayload/preds.json",
    ),
    ("gemini", "fcv_cwe78"): ImportSpec(
        model_name="gemini",
        model_key="gemini3_flash_swebench_imported",
        agent_name="minisweagent_gemini3_flash",
        attack_name="fcv_cwe78",
        preds_path="bug____claude-3-7-sonnet/lite/mini_swe____gemini-2.0-flash/run_command/preds.json",
    ),
    ("gemini", "swexploit_anthropic"): ImportSpec(
        model_name="gemini",
        model_key="gemini3_flash_swebench_imported",
        agent_name="minisweagent_gemini3_flash",
        attack_name="swexploit_anthropic",
        preds_path="swe____claude-3-7-sonnet/lite/mini_swe____gemini-2.0-flash/run_command/preds.json",
    ),
    ("claude", "none"): ImportSpec(
        model_name="claude",
        model_key="claude_sonnet46_swebench_imported",
        agent_name="sweagent_claude37_sonnet_vertex",
        attack_name="none",
        preds_path="base____NoneAttackLLM/lite/swe____claude-sonnet-4/NoPayload/preds.json",
    ),
    ("claude", "fcv_cwe78"): ImportSpec(
        model_name="claude",
        model_key="claude_sonnet46_swebench_imported",
        agent_name="sweagent_claude37_sonnet_vertex",
        attack_name="fcv_cwe78",
        preds_path="bug____claude-3-7-sonnet/lite/swe____claude-sonnet-4/run_command/preds.json",
    ),
    ("claude", "swexploit_anthropic"): ImportSpec(
        model_name="claude",
        model_key="claude_sonnet46_swebench_imported",
        agent_name="sweagent_claude37_sonnet_vertex",
        attack_name="swexploit_anthropic",
        preds_path="swe____claude-3-7-sonnet/lite/swe____claude-sonnet-4/run_command/preds.json",
    ),
}


def _jsonl_text(rows: list[dict[str, Any]]) -> str:
    text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in rows)
    return text + ("\n" if text else "")


def _resolve_zip_member(zf: ZipFile, requested_path: str) -> str:
    requested = requested_path.strip().lstrip("/")
    names = set(zf.namelist())
    candidates = [
        requested,
        f"generated_patch 2/{requested}",
    ]
    for candidate in candidates:
        if candidate in names:
            return candidate
    suffix_matches = [name for name in names if name.endswith(requested)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if suffix_matches:
        raise ValueError(
            f"Ambiguous preds path {requested_path!r}; matched {len(suffix_matches)} members: "
            f"{suffix_matches[:10]}"
        )
    raise FileNotFoundError(f"Missing preds path in zip: {requested_path}")


def _load_preds(zf: ZipFile, requested_path: str) -> tuple[str, dict[str, dict[str, Any]]]:
    member = _resolve_zip_member(zf, requested_path)
    payload = json.loads(zf.read(member))
    if isinstance(payload, dict):
        rows: dict[str, dict[str, Any]] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                row = dict(value)
            else:
                row = {"model_patch": str(value or "")}
            iid = str(row.get("instance_id") or key)
            row["instance_id"] = iid
            rows[iid] = row
        return member, rows
    if isinstance(payload, list):
        rows = {}
        for idx, value in enumerate(payload):
            if not isinstance(value, dict):
                continue
            iid = str(value.get("instance_id") or f"row-{idx}")
            row = dict(value)
            row["instance_id"] = iid
            rows[iid] = row
        return member, rows
    raise ValueError(f"Expected dict or list JSON at {member}, got {type(payload).__name__}")


def _extract_patch(row: dict[str, Any]) -> str:
    for key in ("model_patch", "patch", "unified_diff", "diff", "adv_patch"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _load_instances(
    *,
    config_dir: Path,
    dataset_name: str,
    split: str,
    dataset_data_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_config = load_component_config(config_dir, "datasets", dataset_name)
    if dataset_data_path is not None:
        dataset_config = dict(dataset_config)
        dataset_config["data_path"] = str(dataset_data_path)

    dataset_plugin = str(dataset_config.get("plugin", dataset_name))
    dataset_obj = get_dataset(dataset_plugin)()
    data_result = dataset_obj.load(
        split=split or str(dataset_config.get("split", "test")),
        config=dataset_config,
        runtime_dir=Path(".import_swebench_runtime"),
    )
    if data_result.errors:
        raise RuntimeError("Failed to load SWE-Bench metadata: " + "; ".join(data_result.errors[:10]))
    instances = {instance.instance_id: instance for instance in data_result.instances}
    if not instances:
        raise RuntimeError(f"No instances loaded for dataset {dataset_name!r}")
    return dataset_config, instances


def _write_patch_artifacts(
    *,
    out_dir: Path,
    instance_id: str,
    ori_patch: str,
    adv_patch: str,
) -> dict[str, str]:
    patch_dir = out_dir / "artifacts" / "patches" / instance_id
    patch_dir.mkdir(parents=True, exist_ok=True)
    ori_path = patch_dir / "ori_patch.diff"
    adv_path = patch_dir / "adv_patch.diff"
    final_path = patch_dir / "final_patch.diff"
    atomic_write_text(ori_path, ori_patch)
    atomic_write_text(adv_path, adv_patch)
    atomic_write_text(final_path, "")
    atomic_write_json(
        patch_dir / "metadata.json",
        {
            "ori_patch_hash": sha256_text(ori_patch),
            "adv_patch_hash": sha256_text(adv_patch),
            "final_patch_hash": "",
            "apply_status": {"applied": False, "reason_code": "import_not_applied"},
            "imported_external_patch": True,
        },
    )
    return {
        "ori_patch_path": str(ori_path),
        "adv_patch_path": str(adv_path),
        "final_patch_path": str(final_path),
    }


def _write_prompt_artifacts(
    *,
    out_dir: Path,
    instance_id: str,
    attack_name: str,
    prompt: str,
    source_member: str,
) -> str:
    artifact_dir = out_dir / "artifacts" / "attacks" / instance_id / attack_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(artifact_dir / "original_prompt.txt", prompt)
    atomic_write_text(artifact_dir / "adv_prompt.txt", prompt)
    atomic_write_json(
        artifact_dir / "metadata.json",
        {
            "import_prompt_source": "original_problem_statement",
            "import_source_preds_path": source_member,
            "imported_external_patch": True,
        },
    )
    return str(artifact_dir)


def _test_specs(instance: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": test.name,
            "command": test.command,
            "cwd": test.cwd,
            "env": test.env,
        }
        for test in instance.tests
    ]


def _build_row(
    *,
    instance: Any,
    dataset_config_hash: str,
    agent_name: str,
    agent_config_hash: str,
    attack_name: str,
    attack_config_hash: str,
    attack_config: dict[str, Any],
    fidelity_mode: str,
    out_dir: Path,
    source_zip: Path,
    source_member: str,
    source_row: dict[str, Any],
    ori_patch: str,
    adv_patch: str,
) -> dict[str, Any]:
    prompt = str(instance.prompt or "")
    model_name = str(source_row.get("model_name_or_path") or source_row.get("model") or "")
    attack_presence_trusted = attack_name.strip().lower() != "none"
    patch_paths = _write_patch_artifacts(
        out_dir=out_dir,
        instance_id=instance.instance_id,
        ori_patch=ori_patch,
        adv_patch=adv_patch,
    )
    artifact_path = _write_prompt_artifacts(
        out_dir=out_dir,
        instance_id=instance.instance_id,
        attack_name=attack_name,
        prompt=prompt,
        source_member=source_member,
    )
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    source_family = source_member.split("/")[1] if source_member.startswith("generated_patch 2/") else source_member.split("/")[0]
    return {
        "dataset": instance.dataset,
        "dataset_config_hash": dataset_config_hash,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "instance_metadata": instance.metadata,
        "repo_id": instance.repo_snapshot.repo_id,
        "base_commit": instance.repo_snapshot.base_commit,
        "repo_path": instance.repo_snapshot.path,
        "agent_name": agent_name,
        "agent_config_hash": agent_config_hash,
        "attack_name": attack_name,
        "attack_config_hash": attack_config_hash,
        "attack_mode": "imported_preds_json",
        "fidelity_mode": fidelity_mode,
        "attack_provider": "advisor_zip",
        "attack_model": model_name,
        "attack_prompt_hash": sha256_text(prompt),
        "attack_response_hash": sha256_text(adv_patch),
        "attack_artifact_path": artifact_path,
        "attack_cache_hit": False,
        "attack_cache_key": "",
        "attack_provider_fallback": False,
        "attack_tool_blocked": False,
        "attack_token_usage": {},
        "attack_objective_tags": attack_config.get("objective_tags", []),
        "attack_stealth_constraints": attack_config.get("stealth_constraints", []),
        "attack_presence_trusted": attack_presence_trusted,
        "attack_presence_validation_policy": (
            "trusted_imported_external_patch" if attack_presence_trusted else "validate_normally"
        ),
        "attack_selected_patch_id": instance.instance_id,
        "attack_source_json_path": source_member,
        "attack_source_json_hash": sha256_text(json.dumps(source_row, sort_keys=True, ensure_ascii=True)),
        "original_prompt_hash": sha256_text(prompt),
        "ori_patch_hash": sha256_text(ori_patch),
        "adv_patch_hash": sha256_text(adv_patch),
        "patch_hash": sha256_text(adv_patch),
        "ori_agent_metadata": {
            "agent": agent_name,
            "imported_external_patch": True,
            "import_role": "clean_reference",
        },
        "adv_agent_metadata": {
            "agent": agent_name,
            "imported_external_patch": True,
            "import_source_model_name_or_path": model_name,
            "import_source_preds_path": source_member,
            "attack_presence_trusted": attack_presence_trusted,
        },
        "patch_artifacts": patch_paths,
        "test_specs": _test_specs(instance),
        "timestamp_start": now,
        "timestamp_end": now,
        "runtime_sec": 0.0,
        "imported_external_patch": True,
        "import_source_zip": str(source_zip),
        "import_source_preds_path": source_member,
        "import_source_model_name_or_path": model_name,
        "import_source_family": source_family,
        "import_prompt_source": "original_problem_statement",
    }


def _write_import_manifest(out_dir: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(out_dir / "import_manifest.json", payload)


def _import_spec(
    *,
    zf: ZipFile,
    spec: ImportSpec,
    clean_preds: dict[str, dict[str, Any]],
    dataset_config_hash: str,
    agent_config_hash: str,
    attack_config_hash: str,
    attack_config: dict[str, Any],
    instances: dict[str, Any],
    out_root: Path,
    dataset_name: str,
    source_zip: Path,
    fidelity_mode: str,
    limit: int | None,
) -> dict[str, Any]:
    source_member, preds = _load_preds(zf, spec.preds_path)
    out_dir = out_root / spec.model_key / "full" / f"{dataset_name}_{spec.attack_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    missing_metadata = 0
    empty_patches = 0

    for instance_id, source_row in preds.items():
        instance = instances.get(instance_id)
        if instance is None:
            missing_metadata += 1
            continue
        adv_patch = _extract_patch(source_row)
        if not adv_patch:
            empty_patches += 1
        clean_patch = _extract_patch(clean_preds.get(instance_id, {}))
        ori_patch = adv_patch if spec.attack_name == "none" else clean_patch
        rows.append(
            _build_row(
                instance=instance,
                dataset_config_hash=dataset_config_hash,
                agent_name=spec.agent_name,
                agent_config_hash=agent_config_hash,
                attack_name=spec.attack_name,
                attack_config_hash=attack_config_hash,
                attack_config=attack_config,
                fidelity_mode=fidelity_mode,
                out_dir=out_dir,
                source_zip=source_zip,
                source_member=source_member,
                source_row=source_row,
                ori_patch=ori_patch,
                adv_patch=adv_patch,
            )
        )
        if limit is not None and len(rows) >= limit:
            break

    atomic_write_text(out_dir / "attack_results.jsonl", _jsonl_text(rows))
    finalized_rows, summary = finalize_attack_dataset(
        out_dir=out_dir,
        dataset_hash=dataset_config_hash,
        agent_hash=agent_config_hash,
        attack_hash=attack_config_hash,
    )
    require_finalized_attack_rows(finalized_rows, out_dir / "attack_dataset.jsonl")
    _write_import_manifest(
        out_dir,
        {
            "source_zip": str(source_zip),
            "source_preds_path": source_member,
            "model_name": spec.model_name,
            "model_key": spec.model_key,
            "agent_name": spec.agent_name,
            "attack_name": spec.attack_name,
            "dataset_name": dataset_name,
            "raw_rows_written": len(rows),
            "missing_metadata_rows": missing_metadata,
            "empty_patch_rows": empty_patches,
            "preprocessing_summary": summary,
        },
    )
    print(
        f"[import-swebench-preds] {spec.model_key}/{dataset_name}_{spec.attack_name}: "
        f"raw={len(rows)} kept={summary.get('final_dataset_size')} "
        f"discarded={summary.get('total_failed_attacks_discarded')} -> {out_dir / 'attack_dataset.jsonl'}",
        flush=True,
    )
    return summary


def _with_overrides(args: argparse.Namespace, spec: ImportSpec) -> ImportSpec:
    path = getattr(args, f"{spec.model_name}_{spec.attack_name}_preds".replace("-", "_"), None)
    model_key = getattr(args, f"{spec.model_name}_model_key")
    agent_name = getattr(args, f"{spec.model_name}_agent")
    return ImportSpec(
        model_name=spec.model_name,
        model_key=model_key or spec.model_key,
        agent_name=agent_name or spec.agent_name,
        attack_name=spec.attack_name,
        preds_path=path or spec.preds_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--dataset", default="swebench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-data-path", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/attacks"))
    parser.add_argument("--models", nargs="+", choices=["gemini", "claude"], default=["gemini", "claude"])
    parser.add_argument("--attacks", nargs="+", choices=["none", "fcv_cwe78", "swexploit_anthropic"], default=["none", "fcv_cwe78", "swexploit_anthropic"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fidelity-mode", default="imported")
    parser.add_argument("--gemini-model-key", default=DEFAULT_SPECS[("gemini", "none")].model_key)
    parser.add_argument("--claude-model-key", default=DEFAULT_SPECS[("claude", "none")].model_key)
    parser.add_argument("--gemini-agent", default=DEFAULT_SPECS[("gemini", "none")].agent_name)
    parser.add_argument("--claude-agent", default=DEFAULT_SPECS[("claude", "none")].agent_name)
    parser.add_argument("--gemini-none-preds", default=None)
    parser.add_argument("--gemini-fcv-cwe78-preds", dest="gemini_fcv_cwe78_preds", default=None)
    parser.add_argument("--gemini-swexploit-anthropic-preds", dest="gemini_swexploit_anthropic_preds", default=None)
    parser.add_argument("--claude-none-preds", default=None)
    parser.add_argument("--claude-fcv-cwe78-preds", dest="claude_fcv_cwe78_preds", default=None)
    parser.add_argument("--claude-swexploit-anthropic-preds", dest="claude_swexploit_anthropic_preds", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    zip_path = args.zip_path.expanduser()
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    config_dir = args.config_dir.expanduser()
    out_root = args.outputs_root.expanduser()
    dataset_data_path = args.dataset_data_path.expanduser() if args.dataset_data_path else None

    dataset_config, instances = _load_instances(
        config_dir=config_dir,
        dataset_name=args.dataset,
        split=args.split,
        dataset_data_path=dataset_data_path,
    )
    dataset_hash = config_hash(dataset_config)

    summaries: dict[str, Any] = {}
    with ZipFile(zip_path) as zf:
        for model_name in args.models:
            clean_spec = _with_overrides(args, DEFAULT_SPECS[(model_name, "none")])
            clean_member, clean_preds = _load_preds(zf, clean_spec.preds_path)
            del clean_member
            for attack_name in args.attacks:
                spec = _with_overrides(args, DEFAULT_SPECS[(model_name, attack_name)])
                agent_config = load_component_config(config_dir, "agents", spec.agent_name)
                attack_config = load_component_config(config_dir, "attacks", spec.attack_name)
                summary = _import_spec(
                    zf=zf,
                    spec=spec,
                    clean_preds=clean_preds,
                    dataset_config_hash=dataset_hash,
                    agent_config_hash=config_hash(agent_config),
                    attack_config_hash=config_hash(attack_config),
                    attack_config=attack_config,
                    instances=instances,
                    out_root=out_root,
                    dataset_name=args.dataset,
                    source_zip=zip_path,
                    fidelity_mode=args.fidelity_mode,
                    limit=args.limit,
                )
                summaries[f"{spec.model_key}/{args.dataset}_{spec.attack_name}"] = summary

    atomic_write_json(
        out_root / "swebench_preds_import_summary.json",
        {
            "source_zip": str(zip_path),
            "dataset": args.dataset,
            "models": args.models,
            "attacks": args.attacks,
            "limit": args.limit,
            "outputs": summaries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
