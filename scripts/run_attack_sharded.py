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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TextIO

import src.dataset  # noqa: F401
from src.common.artifact_store import atomic_write_json, atomic_write_text
from src.common.config import config_hash, load_component_config
from src.dataset.registry import get_dataset
from src.eval.attack_finalize import _validate_attack_row
from src.eval.report import load_jsonl_rows

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - progress bar is optional
    tqdm = None

RunningShard = tuple[
    int,
    subprocess.Popen[str],
    List[str],
    Path,
    float,
    int,
    TextIO,
    TextIO,
]

EMPTY_PATCH_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _dataset_attack_dir(dataset: str, attack: str) -> str:
    return f"{dataset}_{attack}"


def _instance_ids(
    dataset: str,
    split: str,
    config_dir: Path,
    limit: int | None,
    dataset_data_path: str | None,
) -> List[str]:
    cfg = load_component_config(config_dir, "datasets", dataset)
    if dataset_data_path:
        cfg = dict(cfg)
        cfg["data_path"] = dataset_data_path
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
    dataset_hash: str,
    agent_hash: str,
    attack_hash: str,
    allowed_ids: set[str],
    completed_after_epoch: float | None = None,
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
            if str(row.get("dataset_config_hash", "") or "") != dataset_hash:
                continue
            if str(row.get("agent_config_hash", "") or "") != agent_hash:
                continue
            if str(row.get("attack_config_hash", "") or "") != attack_hash:
                continue
            if completed_after_epoch is not None:
                row_epoch = _row_timestamp_epoch(row)
                if row_epoch is None or row_epoch <= completed_after_epoch:
                    continue
            patch_hash = str(row.get("patch_hash", "") or row.get("adv_patch_hash", "") or "")
            if patch_hash == EMPTY_PATCH_HASH:
                continue
            patch_artifacts = row.get("patch_artifacts")
            if isinstance(patch_artifacts, dict):
                patch_path = patch_artifacts.get("adv_patch_path") or patch_artifacts.get("patch_path")
                if patch_path and Path(str(patch_path)).exists():
                    if not Path(str(patch_path)).read_text(encoding="utf-8").strip():
                        continue
            completed.add(instance_id)
    return completed


def _row_timestamp_epoch(row: Dict[str, Any]) -> float | None:
    for key in ("timestamp_end", "timestamp_start"):
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            continue
    return None


def _finalized_instance_ids(
    final_out: Path,
    *,
    dataset: str,
    agent: str,
    attack: str,
    dataset_hash: str,
    agent_hash: str,
    attack_hash: str,
    allowed_ids: set[str],
) -> set[str]:
    path = final_out / "attack_dataset.jsonl"
    if not path.exists():
        return set()

    finalized: set[str] = set()
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
        if str(row.get("dataset_config_hash", "") or "") != dataset_hash:
            continue
        if str(row.get("agent_config_hash", "") or "") != agent_hash:
            continue
        if str(row.get("attack_config_hash", "") or "") != attack_hash:
            continue
        finalized.add(instance_id)
    return finalized


def _has_finalized_dataset(final_out: Path) -> bool:
    path = final_out / "attack_dataset.jsonl"
    return path.exists()


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


def _result_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _current_config_hashes(
    *,
    config_dir: Path,
    dataset: str,
    agent: str,
    attack: str,
    dataset_data_path: str | None,
) -> Dict[str, str]:
    dataset_cfg = load_component_config(config_dir, "datasets", dataset)
    if dataset_data_path:
        dataset_cfg = dict(dataset_cfg)
        dataset_cfg["data_path"] = dataset_data_path
    agent_cfg = load_component_config(config_dir, "agents", agent)
    attack_cfg = load_component_config(config_dir, "attacks", attack)
    return {
        "dataset": config_hash(dataset_cfg),
        "agent": config_hash(agent_cfg),
        "attack": config_hash(attack_cfg),
    }


def _run_shards(
    commands: List[tuple[List[str], Path]],
    *,
    parallel: int,
    env: Dict[str, str],
    total_remaining: int,
) -> List[int]:
    running: List[RunningShard] = []
    returncodes: List[int] = []
    next_idx = 0
    progress = (
        tqdm(total=total_remaining, desc="attack shards", unit="row", dynamic_ncols=True) if tqdm else None
    )
    try:
        while next_idx < len(commands) or running:
            while next_idx < len(commands) and len(running) < parallel:
                cmd, result_path = commands[next_idx]
                shard_out = result_path.parent
                logs_dir = shard_out / "logs" / "runner"
                logs_dir.mkdir(parents=True, exist_ok=True)
                stdout_f = (logs_dir / "stdout.log").open("a", encoding="utf-8")
                stderr_f = (logs_dir / "stderr.log").open("a", encoding="utf-8")
                print(f"[shard-runner] start shard={next_idx:02d} logs={logs_dir}", flush=True)
                child_env = dict(env)
                child_env["CFG_DISABLE_TQDM"] = "1"
                proc = subprocess.Popen(
                    cmd,
                    env=child_env,
                    text=True,
                    stdout=stdout_f,
                    stderr=stderr_f,
                )
                running.append(
                    (
                        next_idx,
                        proc,
                        cmd,
                        result_path,
                        time.time(),
                        _result_line_count(result_path),
                        stdout_f,
                        stderr_f,
                    )
                )
                next_idx += 1

            time.sleep(5)
            still_running: List[RunningShard] = []
            for idx, proc, cmd, result_path, start, last_count, stdout_f, stderr_f in running:
                rc = proc.poll()
                current_count = _result_line_count(result_path)
                if current_count > last_count:
                    delta = current_count - last_count
                    if progress is not None:
                        progress.update(delta)
                    last_count = current_count
                if rc is None:
                    still_running.append(
                        (idx, proc, cmd, result_path, start, last_count, stdout_f, stderr_f)
                    )
                    continue
                stdout_f.close()
                stderr_f.close()
                elapsed = int(time.time() - start)
                if progress is None:
                    print(
                        f"[shard-runner] finished shard={idx:02d} rc={rc} "
                        f"rows={current_count} elapsed={elapsed}s",
                        flush=True,
                    )
                returncodes.append(rc)
            running = still_running
    finally:
        if progress is not None:
            progress.close()
        for _, proc, _, _, _, _, stdout_f, stderr_f in running:
            if proc.poll() is None:
                proc.terminate()
            stdout_f.close()
            stderr_f.close()
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
    dataset_hash: str,
    agent_hash: str,
    attack_hash: str,
    shard_dirs: List[Path],
) -> Dict[str, Any]:
    """Finalize compatible merged rows for the current config hashes only."""
    rows_by_instance_id: Dict[str, Dict[str, Any]] = {}
    duplicate_instances = 0
    skipped_hash_mismatch = 0
    for row in load_jsonl_rows(out_dir / "attack_results.jsonl"):
        if str(row.get("dataset", "") or "") != dataset:
            continue
        if str(row.get("agent_name", "") or "") != agent:
            continue
        if str(row.get("attack_name", "") or "") != attack:
            continue
        if (
            str(row.get("dataset_config_hash", "") or "") != dataset_hash
            or str(row.get("agent_config_hash", "") or "") != agent_hash
            or str(row.get("attack_config_hash", "") or "") != attack_hash
        ):
            skipped_hash_mismatch += 1
            continue
        instance_id = str(row.get("instance_id", "") or "")
        if instance_id in rows_by_instance_id:
            duplicate_instances += 1
        rows_by_instance_id[instance_id] = row
    rows = list(rows_by_instance_id.values())

    finalized_rows: List[Dict[str, Any]] = []
    discard_counts: Counter[str] = Counter()
    running_rows: List[str] = []
    agent_hashes: Counter[str] = Counter()

    progress = tqdm(total=len(rows), desc="finalizing attacks", unit="row", dynamic_ncols=True) if tqdm else None
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
        if progress is not None:
            progress.update(1)
        elif idx % 25 == 0 or idx == len(rows):
            print(
                f"[shard-runner] finalizing {idx}/{len(rows)} kept={len(finalized_rows)} "
                f"discarded={sum(discard_counts.values())}",
                flush=True,
            )
    if progress is not None:
        progress.close()

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
        "skipped_hash_mismatch": skipped_hash_mismatch,
        "current_config_hashes": {
            "dataset": dataset_hash,
            "agent": agent_hash,
            "attack": attack_hash,
        },
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
        "--dataset-data-path",
        default=None,
        help="Optional local dataset JSONL path override for datasets that use data_path.",
    )
    parser.add_argument(
        "--no-global-resume",
        action="store_true",
        help="Do not scan existing shard outputs before rechunking. By default, completed IDs are skipped globally.",
    )
    parser.add_argument(
        "--retry-discarded",
        action="store_true",
        help="Resume from the finalized dataset instead of raw shard rows, rerunning instances that were discarded.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.shards < 1 or args.parallel < 1:
        raise SystemExit("--shards and --parallel must be >= 1")
    if args.retry_discarded and args.no_global_resume:
        raise SystemExit("--retry-discarded and --no-global-resume are mutually exclusive")

    config_dir = Path(args.config_dir)
    ids = _instance_ids(args.dataset, args.split, config_dir, args.limit, args.dataset_data_path)
    model_root = Path(args.out_root)
    run_root = model_root / args.mode
    shard_base = run_root / "shards"
    final_out = run_root / _dataset_attack_dir(args.dataset, args.attack)
    final_out.mkdir(parents=True, exist_ok=True)
    current_hashes = _current_config_hashes(
        config_dir=config_dir,
        dataset=args.dataset,
        agent=args.agent,
        attack=args.attack,
        dataset_data_path=args.dataset_data_path,
    )

    existing_shard_dirs = _matching_existing_shard_dirs(shard_base, args.dataset, args.attack)
    completed_ids: set[str] = set()
    partial_retry_completed_ids: set[str] = set()
    if args.retry_discarded:
        completed_ids = _finalized_instance_ids(
            final_out,
            dataset=args.dataset,
            agent=args.agent,
            attack=args.attack,
            dataset_hash=current_hashes["dataset"],
            agent_hash=current_hashes["agent"],
            attack_hash=current_hashes["attack"],
            allowed_ids=set(ids),
        )
        finalized_dataset_path = final_out / "attack_dataset.jsonl"
        if _has_finalized_dataset(final_out):
            partial_retry_completed_ids = _completed_instance_ids(
                existing_shard_dirs,
                dataset=args.dataset,
                agent=args.agent,
                attack=args.attack,
                dataset_hash=current_hashes["dataset"],
                agent_hash=current_hashes["agent"],
                attack_hash=current_hashes["attack"],
                allowed_ids=set(ids),
                completed_after_epoch=finalized_dataset_path.stat().st_mtime,
            )
            completed_ids.update(partial_retry_completed_ids)
        else:
            completed_ids.update(
                _completed_instance_ids(
                    existing_shard_dirs,
                    dataset=args.dataset,
                    agent=args.agent,
                    attack=args.attack,
                    dataset_hash=current_hashes["dataset"],
                    agent_hash=current_hashes["agent"],
                    attack_hash=current_hashes["attack"],
                    allowed_ids=set(ids),
                )
            )
    elif not args.no_global_resume:
        completed_ids = _completed_instance_ids(
            existing_shard_dirs,
            dataset=args.dataset,
            agent=args.agent,
            attack=args.attack,
            dataset_hash=current_hashes["dataset"],
            agent_hash=current_hashes["agent"],
            attack_hash=current_hashes["attack"],
            allowed_ids=set(ids),
        )
    pending_ids = [instance_id for instance_id in ids if instance_id not in completed_ids]
    chunks = _chunks(pending_ids, args.shards)

    commands: List[tuple[List[str], Path]] = []
    shard_dirs: List[Path] = []
    for idx, shard_ids in enumerate(chunks):
        shard_out = shard_base / f"shard_{idx:02d}" / _dataset_attack_dir(args.dataset, args.attack)
        shard_dirs.append(shard_out)
        command = [
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
        if args.dataset_data_path:
            command.extend(["--dataset-data-path", args.dataset_data_path])
        commands.append((command, shard_out / "attack_results.jsonl"))

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "agent": args.agent,
        "attack": args.attack,
        "instances": len(ids),
        "completed_instances_skipped": len(completed_ids),
        "partial_retry_instances_skipped": len(partial_retry_completed_ids),
        "pending_instances": len(pending_ids),
        "global_resume_enabled": not args.no_global_resume,
        "retry_discarded": args.retry_discarded,
        "dataset_data_path": args.dataset_data_path or "",
        "current_config_hashes": current_hashes,
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
        for cmd, _ in commands:
            print("[shard-runner] dry-run:", " ".join(cmd))
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    returncodes = _run_shards(commands, parallel=args.parallel, env=env, total_remaining=len(pending_ids))
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
        dataset_hash=current_hashes["dataset"],
        agent_hash=current_hashes["agent"],
        attack_hash=current_hashes["attack"],
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
