#!/usr/bin/env python3
"""Run Llama Guard over a portable mixed-dataset bundle without repositories."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.baseline.llama_guard import LlamaGuardDefense
from src.common.config import config_hash, load_yaml


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row.get("instance_id", ""))
        for row in _read_jsonl(path)
        if row.get("instance_id")
    }


def _select_entries(entries: list[dict[str, Any]], filters: list[str]) -> list[dict[str, Any]]:
    if not filters:
        return entries
    lowered = [value.lower() for value in filters]
    return [
        entry
        for entry in entries
        if any(value in str(entry.get("id", "")).lower() for value in lowered)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path, help="Extracted bundle directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines/llama_guard.yaml"),
        help="Llama Guard baseline YAML in the cfg-semantic-grounding checkout",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/baselines"),
        help="Repository baseline output root",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="TEXT",
        help="Run manifest IDs containing TEXT; repeat to select multiple groups",
    )
    parser.add_argument(
        "--input-field",
        choices=["prompt", "patch", "prompt_and_patch"],
        help="Override the baseline YAML input_field (default YAML behavior is prompt)",
    )
    parser.add_argument(
        "--fidelity-mode",
        choices=["llm", "surrogate_debug"],
        default="llm",
    )
    parser.add_argument("--limit", type=int, help="Maximum rows per selected dataset")
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    manifest_path = bundle_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _select_entries(list(manifest["datasets"]), args.only)
    if not entries:
        raise SystemExit("No bundle datasets matched --only filters")

    config = load_yaml(args.config)
    if args.input_field:
        config["input_field"] = args.input_field
    baseline_name = str(config.get("name", "llama_guard"))
    baseline_hash = config_hash(config)

    print(
        f"[portable-llama-guard] loading model once; datasets={len(entries)} "
        f"input_field={config.get('input_field', 'prompt')}",
        flush=True,
    )
    defense = LlamaGuardDefense(
        config=config,
        llm_client=None,
        baseline_config_hash=baseline_hash,
        run_root=args.out_root,
        fidelity_mode=args.fidelity_mode,
    )

    for entry in entries:
        dataset_path = bundle_root / entry["path"]
        out_dir = args.out_root / entry["result_relpath"]
        results_path = out_dir / "results.jsonl"
        completed = _completed_ids(results_path)
        rows = _read_jsonl(dataset_path)
        if args.limit is not None:
            rows = rows[: max(0, args.limit)]
        pending = [row for row in rows if str(row.get("instance_id", "")) not in completed]

        _write_json(
            out_dir / "integration_spec.json",
            {
                "portable_classifier_only": True,
                "source_bundle": str(bundle_root),
                "source_dataset": entry,
                "selected_configs": {
                    "baseline": {
                        "name": baseline_name,
                        "hash": baseline_hash,
                        "config": config,
                    }
                },
            },
        )
        print(
            f"[portable-llama-guard] {entry['id']}: total={len(rows)} "
            f"completed={len(rows) - len(pending)} remaining={len(pending)}",
            flush=True,
        )

        for index, row in enumerate(pending, start=1):
            started = time.time()
            instance_id = str(row.get("instance_id", "unknown"))
            prompt = str(row.get("portable_prompt", ""))
            patch = str(row.get("portable_adv_patch", ""))
            repo_code = {
                "instance_id": instance_id,
                "source_instance_id": row.get("source_instance_id", ""),
                "dataset": row.get("dataset", ""),
                "agent_name": row.get("agent_name", ""),
                "attack_name": row.get("attack_name", ""),
                "path": "",
            }
            raw_decision = defense.defense(prompt, patch, [], repo_code)
            decision = "accept" if raw_decision is True else "reject"
            result = {
                **row,
                "baseline_name": baseline_name,
                "baseline_config_hash": baseline_hash,
                "defense_decision": decision,
                "defense_signals": dict(defense.last_signals),
                "apply_ok": False,
                "tests_passed": False,
                "portable_classifier_only": True,
                "defense_runtime_sec": round(time.time() - started, 6),
            }
            _append_jsonl(results_path, result)
            completed.add(instance_id)
            if index == 1 or index % 25 == 0 or index == len(pending):
                print(
                    f"[portable-llama-guard] {entry['id']}: {index}/{len(pending)} new rows",
                    flush=True,
                )

    print("[portable-llama-guard] completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
