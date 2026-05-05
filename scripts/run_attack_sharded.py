#!/usr/bin/env python3
"""Run attack generation over dataset shards and merge finalized outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import src.dataset  # noqa: F401
from src.common.artifact_store import atomic_write_json, atomic_write_text
from src.common.config import load_component_config
from src.dataset.registry import get_dataset
from src.eval.attack_finalize import _validate_attack_row
from src.eval.report import load_jsonl_rows


def _dataset_attack_dir(dataset: str, attack: str) -> str:
    return f"{dataset}_{attack}"


def _instance_ids(dataset: str, split: str, config_dir: Path, limit: int | None) -> List[str]:
    cfg = load_component_config(config_dir, "datasets", dataset)
    plugin = str(cfg.get("plugin", dataset))
    result = get_dataset(plugin)().load(
        split=split,
        config=cfg,
        runtime_dir=Path("outputs/runtime/shard_loader"),
        limit=limit,
        instance_ids=None,
    )
    if result.errors:
        raise RuntimeError("Dataset load errors:\n" + "\n".join(result.errors[:20]))
    return [inst.instance_id for inst in result.instances]


def _chunks(values: List[str], shards: int) -> List[List[str]]:
    buckets: List[List[str]] = [[] for _ in range(shards)]
    for idx, value in enumerate(values):
        buckets[idx % shards].append(value)
    return [bucket for bucket in buckets if bucket]


def _matching_existing_shard_dirs(shard_base: Path, dataset: str, attack: str) -> List[Path]:
    leaf = _dataset_attack_dir(dataset, attack)
    if not shard_base.exists():
        return []
    return sorted(path for path in shard_base.glob(f"shard_*/{leaf}") if path.is_dir())


def _completed_instance_ids(
    shard_dirs: List[Path],
    *,
    dataset: str,
    agent: str,
    attack: str,
    allowed_ids: set[str],
) -> set[str]:
    completed: set[str] = set()
    for shard_dir in shard_dirs:
        path = shard_dir / "attack_results.jsonl"
        if not path.exists():
            continue
        for row in load_jsonl_rows(path):
            instance_id = str(row.get("instance_id", "") or "")
            if instance_id not in allowed_ids:
                continue
            if str(row.get("dataset", "") or "") != dataset:
                continue
            if str(row.get("agent_name", "") or "") != agent:
                continue
            if str(row.get("attack_name", "") or "") != attack:
                continue
            completed.add(instance_id)
    return completed


def _unique_paths(paths: List[Path]) -> List[Path]:
    seen: set[str] = set()
    unique: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _run_shards(commands: List[List[str]], *, parallel: int, env: Dict[str, str]) -> List[int]:
    running: List[tuple[int, subprocess.Popen[str], List[str], float]] = []
    returncodes: List[int] = []
    next_idx = 0
    while next_idx < len(commands) or running:
        while next_idx < len(commands) and len(running) < parallel:
            cmd = commands[next_idx]
            print(f"[shard-runner] start shard={next_idx:02d}: {' '.join(cmd)}", flush=True)
            proc = subprocess.Popen(cmd, env=env, text=True)
            running.append((next_idx, proc, cmd, time.time()))
            next_idx += 1

        time.sleep(5)
        still_running: List[tuple[int, subprocess.Popen[str], List[str], float]] = []
        for idx, proc, cmd, start in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((idx, proc, cmd, start))
                continue
            elapsed = int(time.time() - start)
            print(f"[shard-runner] finished shard={idx:02d} rc={rc} elapsed={elapsed}s", flush=True)
            returncodes.append(rc)
        running = still_running
    return returncodes


def _merge_jsonl(shard_dirs: List[Path], filename: str, out_path: Path) -> int:
    lines: List[str] = []
    for shard_dir in shard_dirs:
        path = shard_dir / filename
        if not path.exists():
            continue
        lines.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    atomic_write_text(out_path, "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def _merge_summaries(shard_dirs: List[Path], out_dir: Path) -> Dict[str, Any]:
    discard_counts: Counter[str] = Counter()
    generated = 0
    discarded = 0
    kept = 0
    for shard_dir in shard_dirs:
        path = shard_dir / "attack_preprocessing_summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        generated += int(summary.get("total_attacks_generated") or 0)
        discarded += int(summary.get("total_failed_attacks_discarded") or 0)
        kept += int(summary.get("final_dataset_size") or 0)
        discard_counts.update(summary.get("discard_reasons") or {})

    summary = {
        "attack_dataset_finalized": True,
        "raw_attack_results_path": str(out_dir / "attack_results.jsonl"),
        "attack_dataset_path": str(out_dir / "attack_dataset.jsonl"),
        "preprocessing_log_path": str(out_dir / "attack_preprocessing_log.jsonl"),
        "total_attacks_generated": generated,
        "total_failed_attacks_discarded": discarded,
        "final_dataset_size": kept,
        "discard_reasons": dict(sorted(discard_counts.items())),
        "merged_from_shards": [str(path) for path in shard_dirs],
    }
    atomic_write_json(out_dir / "attack_preprocessing_summary.json", summary)
    return summary


def _finalize_merged_attack_dataset(
    *,
    out_dir: Path,
    dataset: str,
    agent: str,
    attack: str,
    shard_dirs: List[Path],
) -> Dict[str, Any]:
    """Finalize all compatible merged rows, including rows from resumed config hashes."""
    rows = []
    seen_instance_ids: set[str] = set()
    duplicate_instances = 0
    for row in load_jsonl_rows(out_dir / "attack_results.jsonl"):
        if str(row.get("dataset", "") or "") != dataset:
            continue
        if str(row.get("agent_name", "") or "") != agent:
            continue
        if str(row.get("attack_name", "") or "") != attack:
            continue
        instance_id = str(row.get("instance_id", "") or "")
        if instance_id in seen_instance_ids:
            duplicate_instances += 1
            continue
        seen_instance_ids.add(instance_id)
        rows.append(row)

    finalized_rows: List[Dict[str, Any]] = []
    discard_counts: Counter[str] = Counter()
    running_rows: List[str] = []
    agent_hashes: Counter[str] = Counter()

    for idx, row in enumerate(rows, start=1):
        agent_hashes[str(row.get("agent_config_hash", "") or "")] += 1
        validation = _validate_attack_row(row)
        kept = bool(validation.get("kept", False))
        discard_reason = str(validation.get("discard_reason", ""))
        if not kept and discard_reason:
            discard_counts[discard_reason] += 1

        finalized = dict(row)
        finalized.update(
            {
                "attack_dataset_finalized": True,
                "attack_kept": kept,
                "attack_discard_reason": discard_reason,
                "attack_validation": validation,
                "graph_label": 0 if attack.strip().lower() == "none" else 1,
            }
        )
        if kept:
            finalized_rows.append(finalized)

        running_rows.append(
            json.dumps(
                {
                    "processed": idx,
                    "kept": len(finalized_rows),
                    "discarded": sum(discard_counts.values()),
                    "instance_id": str(row.get("instance_id", "unknown")),
                    "discard_reason": discard_reason,
                },
                sort_keys=True,
            )
        )

    dataset_text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in finalized_rows)
    atomic_write_text(out_dir / "attack_dataset.jsonl", dataset_text + ("\n" if dataset_text else ""))
    atomic_write_text(
        out_dir / "attack_preprocessing_log.jsonl",
        "\n".join(running_rows) + ("\n" if running_rows else ""),
    )

    summary = {
        "attack_dataset_finalized": True,
        "raw_attack_results_path": str(out_dir / "attack_results.jsonl"),
        "attack_dataset_path": str(out_dir / "attack_dataset.jsonl"),
        "preprocessing_log_path": str(out_dir / "attack_preprocessing_log.jsonl"),
        "total_attacks_generated": len(rows),
        "total_failed_attacks_discarded": sum(discard_counts.values()),
        "final_dataset_size": len(finalized_rows),
        "discard_reasons": dict(sorted(discard_counts.items())),
        "merged_from_shards": [str(path) for path in shard_dirs],
        "agent_config_hashes": dict(sorted(agent_hashes.items())),
        "duplicate_instances_skipped": duplicate_instances,
    }
    atomic_write_json(out_dir / "attack_preprocessing_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run attack generation in independent shards.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--attack", required=True)
    parser.add_argument("--out-root", required=True, help="Model-specific root, e.g. outputs/attacks/gemini3_flash")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--fidelity-mode", default="llm")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-global-resume",
        action="store_true",
        help="Do not scan existing shard outputs before rechunking. By default, completed IDs are skipped globally.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.shards < 1 or args.parallel < 1:
        raise SystemExit("--shards and --parallel must be >= 1")

    config_dir = Path(args.config_dir)
    ids = _instance_ids(args.dataset, args.split, config_dir, args.limit)
    model_root = Path(args.out_root)
    run_root = model_root / args.mode
    shard_base = run_root / "shards"
    final_out = run_root / _dataset_attack_dir(args.dataset, args.attack)
    final_out.mkdir(parents=True, exist_ok=True)

    existing_shard_dirs = _matching_existing_shard_dirs(shard_base, args.dataset, args.attack)
    completed_ids: set[str] = set()
    if not args.no_global_resume:
        completed_ids = _completed_instance_ids(
            existing_shard_dirs,
            dataset=args.dataset,
            agent=args.agent,
            attack=args.attack,
            allowed_ids=set(ids),
        )
    pending_ids = [instance_id for instance_id in ids if instance_id not in completed_ids]
    chunks = _chunks(pending_ids, args.shards)

    commands: List[List[str]] = []
    shard_dirs: List[Path] = []
    for idx, shard_ids in enumerate(chunks):
        shard_out = shard_base / f"shard_{idx:02d}" / _dataset_attack_dir(args.dataset, args.attack)
        shard_dirs.append(shard_out)
        commands.append(
            [
                sys.executable,
                "-m",
                "src.eval.cli",
                "run_attack",
                "--dataset",
                args.dataset,
                "--split",
                args.split,
                "--agent",
                args.agent,
                "--attack",
                args.attack,
                "--fidelity-mode",
                args.fidelity_mode,
                "--out",
                str(shard_out),
                "--config-dir",
                str(config_dir),
                "--instance-id",
                ",".join(shard_ids),
            ]
        )

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "agent": args.agent,
        "attack": args.attack,
        "instances": len(ids),
        "completed_instances_skipped": len(completed_ids),
        "pending_instances": len(pending_ids),
        "global_resume_enabled": not args.no_global_resume,
        "shards": len(chunks),
        "parallel": args.parallel,
        "final_out": str(final_out),
        "shard_dirs": [str(path) for path in _unique_paths(existing_shard_dirs + shard_dirs)],
    }
    atomic_write_json(final_out / "shard_manifest.json", manifest)

    print(
        f"[shard-runner] dataset={args.dataset} attack={args.attack} "
        f"instances={len(ids)} completed={len(completed_ids)} pending={len(pending_ids)} "
        f"shards={len(chunks)} parallel={args.parallel} final_out={final_out}",
        flush=True,
    )
    if args.dry_run:
        for cmd in commands:
            print("[shard-runner] dry-run:", " ".join(cmd))
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    returncodes = _run_shards(commands, parallel=args.parallel, env=env)
    if any(rc != 0 for rc in returncodes) or len(returncodes) != len(commands):
        print(f"[shard-runner] one or more shards failed: {returncodes}", file=sys.stderr)
        return 1

    merge_dirs = _unique_paths(_matching_existing_shard_dirs(shard_base, args.dataset, args.attack) + shard_dirs)
    result_rows = _merge_jsonl(merge_dirs, "attack_results.jsonl", final_out / "attack_results.jsonl")
    summary = _finalize_merged_attack_dataset(
        out_dir=final_out,
        dataset=args.dataset,
        agent=args.agent,
        attack=args.attack,
        shard_dirs=merge_dirs,
    )
    print(
        f"[shard-runner] merged raw={result_rows} finalized={summary['final_dataset_size']} "
        f"discarded={summary['total_failed_attacks_discarded']} out={final_out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
