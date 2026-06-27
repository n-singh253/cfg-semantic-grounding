#!/usr/bin/env python3
"""Run defense evaluation over disjoint instance shards and merge results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from src.common.artifact_store import atomic_write_json, atomic_write_text
from src.common.config import config_hash, load_component_config
from src.common.subprocess import command_exists, run_command
from src.eval.report import load_jsonl_rows

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - progress bar is optional
    tqdm = None


def _chunks(values: List[str], shards: int) -> List[List[str]]:
    buckets: List[List[str]] = [[] for _ in range(shards)]
    for idx, value in enumerate(values):
        buckets[idx % shards].append(value)
    return [bucket for bucket in buckets if bucket]


def _completed_ids(paths: List[Path], baseline_hash: str) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl_rows(path):
            if str(row.get("baseline_config_hash", "")) != baseline_hash:
                continue
            instance_id = str(row.get("instance_id", "") or "")
            if instance_id:
                completed.add(instance_id)
    return completed


def _runtime_env_snapshot() -> dict:
    keys = [
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "VERTEXAI_PROJECT",
        "VERTEXAI_LOCATION",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "ANTHROPIC_VERTEX_REGION",
        "CFG_GEMINI_VERTEX_SAFETY_THRESHOLD",
        "CFG_GEMINI_VERTEX_MAX_CONCURRENT_CALLS",
        "CFG_GEMINI_VERTEX_TIMEOUT_MS",
        "CFG_GEMINI_VERTEX_MAX_OUTPUT_TOKENS",
        "CFG_GEMINI_VERTEX_THINKING_BUDGET",
        "CFG_GEMINI_VERTEX_RESPONSE_MIME_TYPE",
        "CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC",
        "CFG_ANTHROPIC_VERTEX_MAX_CONCURRENT_CALLS",
        "CFG_LLM_MAX_OUTPUT_TOKENS",
        "CFG_LLM_THINKING_BUDGET",
        "CFG_LLM_RESPONSE_MIME_TYPE",
        "CFG_TEST_TIMEOUT_SEC",
    ]
    snapshot = {key: os.environ[key] for key in keys if key in os.environ}
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if credentials_path:
        snapshot["GOOGLE_APPLICATION_CREDENTIALS_SET"] = True
        snapshot["GOOGLE_APPLICATION_CREDENTIALS_EXISTS"] = Path(credentials_path).expanduser().exists()
    return snapshot


def _result_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _terminate_process_tree(proc: subprocess.Popen[str], grace_sec: int = 10) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()
    deadline = time.time() + grace_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            proc.kill()


def _run_commands(
    commands: List[tuple[List[str], Path]],
    parallel: int,
    env: Dict[str, str],
    *,
    stale_timeout_sec: int,
    poll_interval_sec: int,
    total_remaining: int,
) -> List[int]:
    running: List[tuple[int, subprocess.Popen[str], List[str], Path, float, float, int, object, object]] = []
    returncodes: List[int] = []
    next_idx = 0
    progress = tqdm(total=total_remaining, desc="defense shards", unit="row", dynamic_ncols=True) if tqdm else None
    completed_rows = 0
    try:
        while next_idx < len(commands) or running:
            while next_idx < len(commands) and len(running) < parallel:
                cmd, result_path = commands[next_idx]
                shard_out = result_path.parent
                logs_dir = shard_out / "logs" / "runner"
                logs_dir.mkdir(parents=True, exist_ok=True)
                stdout_f = (logs_dir / "stdout.log").open("a", encoding="utf-8")
                stderr_f = (logs_dir / "stderr.log").open("a", encoding="utf-8")
                print(f"[defense-shards] start shard={next_idx:02d} logs={logs_dir}", flush=True)
                child_env = dict(env)
                child_env["CFG_DISABLE_TQDM"] = "1"
                proc = subprocess.Popen(
                    cmd,
                    env=child_env,
                    text=True,
                    start_new_session=True,
                    stdout=stdout_f,
                    stderr=stderr_f,
                )
                now = time.time()
                running.append(
                    (next_idx, proc, cmd, result_path, now, now, _result_line_count(result_path), stdout_f, stderr_f)
                )
                next_idx += 1

            time.sleep(max(1, poll_interval_sec))
            still_running: List[tuple[int, subprocess.Popen[str], List[str], Path, float, float, int, object, object]] = []
            for idx, proc, cmd, result_path, started, last_progress, last_count, stdout_f, stderr_f in running:
                rc = proc.poll()
                now = time.time()
                current_count = _result_line_count(result_path)
                if current_count > last_count:
                    delta = current_count - last_count
                    completed_rows += delta
                    if progress is not None:
                        progress.update(delta)
                    last_progress = now
                    last_count = current_count

                if rc is not None:
                    stdout_f.close()
                    stderr_f.close()
                    elapsed = int(now - started)
                    if progress is None:
                        print(
                            f"[defense-shards] finished shard={idx:02d} rc={rc} rows={current_count} "
                            f"elapsed={elapsed}s",
                            flush=True,
                        )
                    returncodes.append(rc)
                    continue

                if stale_timeout_sec > 0 and now - last_progress > stale_timeout_sec:
                    elapsed = int(now - started)
                    stale = int(now - last_progress)
                    print(
                        f"[defense-shards] stale shard={idx:02d} rows={current_count} "
                        f"no_progress={stale}s elapsed={elapsed}s; terminating",
                        file=sys.stderr,
                        flush=True,
                    )
                    _terminate_process_tree(proc)
                    stdout_f.close()
                    stderr_f.close()
                    returncodes.append(124)
                    continue

                still_running.append((idx, proc, cmd, result_path, started, last_progress, last_count, stdout_f, stderr_f))
            running = still_running
    except KeyboardInterrupt:
        print("[defense-shards] interrupted; terminating shard workers", file=sys.stderr, flush=True)
        for _, proc, _, _, _, _, _, stdout_f, stderr_f in running:
            _terminate_process_tree(proc)
            stdout_f.close()
            stderr_f.close()
        raise
    finally:
        if progress is not None:
            progress.close()
    return returncodes


def _merge_results(final_results: Path, shard_results: List[Path], baseline_hash: str) -> int:
    rows_by_instance: Dict[str, dict] = {}
    passthrough_rows: List[dict] = []
    for path in [final_results, *shard_results]:
        if not path.exists():
            continue
        for row in load_jsonl_rows(path):
            if str(row.get("baseline_config_hash", "")) != baseline_hash:
                passthrough_rows.append(row)
                continue
            instance_id = str(row.get("instance_id", "") or "")
            if not instance_id:
                continue
            rows_by_instance[instance_id] = row

    merged_rows = [*passthrough_rows, *rows_by_instance.values()]
    text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in merged_rows)
    atomic_write_text(final_results, text + ("\n" if text else ""))
    return len(rows_by_instance)


def _write_isolated_attack_results(
    *,
    rows: List[dict],
    shard_out: Path,
    repo_copy_root: Path,
    refresh_repo_copies: bool,
) -> Path:
    """Write a shard-local attack_dataset JSONL with repo_path rewritten.

    The original finalized attack datasets point at shared mutable checkouts.
    Static defenses reset/apply/test each row, so parallel workers need private
    repo copies to avoid races.
    """
    shard_name = shard_out.name
    shard_repo_root = repo_copy_root / shard_name
    shard_repo_root.mkdir(parents=True, exist_ok=True)

    repo_paths_by_source: Dict[str, Path] = {}

    def repo_key(src_repo: Path) -> str:
        raw = str(src_repo.resolve())
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def repo_label(src_repo: Path) -> str:
        parts = [part for part in src_repo.parts if part not in {"/", ""}]
        if len(parts) >= 2:
            label = "__".join(parts[-2:])
        else:
            label = src_repo.name or "repo"
        return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in label)

    def materialize_repo_copy(src_repo: Path, instance_id: str) -> Path:
        key = repo_key(src_repo)
        existing = repo_paths_by_source.get(key)
        if existing is not None:
            return existing

        dst_repo = shard_repo_root / f"{repo_label(src_repo)}__{key}"
        if refresh_repo_copies and dst_repo.exists():
            shutil.rmtree(dst_repo)
        if not dst_repo.exists():
            if (src_repo / ".git").exists() and command_exists("git"):
                print(
                    f"[defense-shards] clone shared repo for shard {shard_name}: {src_repo} -> {dst_repo}",
                    flush=True,
                )
                clone = run_command(
                    ["git", "clone", "--quiet", "--shared", str(src_repo), str(dst_repo)],
                    timeout_sec=300,
                )
                if clone.returncode != 0:
                    msg = (clone.stderr or clone.stdout or "").strip()
                    raise RuntimeError(f"git clone failed for {instance_id}: {msg}")
            else:
                print(
                    f"[defense-shards] copy repo for shard {shard_name}: {src_repo} -> {dst_repo}",
                    flush=True,
                )
                shutil.copytree(src_repo, dst_repo, symlinks=True)
        repo_paths_by_source[key] = dst_repo
        return dst_repo

    rewritten_rows: List[dict] = []
    for row in rows:
        instance_id = str(row.get("instance_id", "") or "unknown")
        src_repo = Path(str(row.get("repo_path", "") or ""))
        if not src_repo.exists():
            raise FileNotFoundError(f"repo_path for {instance_id} does not exist: {src_repo}")
        dst_repo = materialize_repo_copy(src_repo, instance_id)

        rewritten = dict(row)
        rewritten["repo_path"] = str(dst_repo)
        rewritten_rows.append(rewritten)

    attack_results_path = shard_out / "attack_dataset.isolated.jsonl"
    text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in rewritten_rows)
    atomic_write_text(attack_results_path, text + ("\n" if text else ""))
    return attack_results_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run src.eval.cli run_defense in independent shards.")
    parser.add_argument("--attack-results", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--fidelity-mode", default="llm", choices=["llm", "surrogate_debug"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--instance-id",
        default=None,
        help="Optional single instance id or comma-separated ids to include before sharding.",
    )
    parser.add_argument(
        "--instance-id-file",
        default=None,
        help="Optional file with one instance id per line to include before sharding.",
    )
    parser.add_argument("--max-patch-attempts", type=int, default=1)
    parser.add_argument("--retry-on-apply-failure", action="store_true", default=False)
    parser.add_argument(
        "--stale-timeout-sec",
        type=int,
        default=1800,
        help="Terminate a shard if its results.jsonl does not gain a row for this many seconds. Use 0 to disable.",
    )
    parser.add_argument("--poll-interval-sec", type=int, default=15)
    parser.add_argument(
        "--isolate-repos",
        action="store_true",
        help="Create shard-private repo checkouts under OUT/_repo_copies and rewrite shard rows to use them.",
    )
    parser.add_argument(
        "--repo-copy-root",
        default=None,
        help="Optional root for --isolate-repos copies. Defaults to OUT/_repo_copies.",
    )
    parser.add_argument(
        "--refresh-repo-copies",
        action="store_true",
        help="Delete and recreate existing isolated repo copies before running a shard.",
    )
    parser.add_argument(
        "--cleanup-repo-copies",
        action="store_true",
        help="Delete isolated repo copies after a complete successful merge.",
    )
    args = parser.parse_args()

    attack_results = Path(args.attack_results)
    out_dir = Path(args.out)
    config_dir = Path(args.config_dir)
    if not attack_results.exists():
        raise SystemExit(f"--attack-results does not exist: {attack_results}")
    out_dir.mkdir(parents=True, exist_ok=True)
    final_results = out_dir / "results.jsonl"

    baseline_cfg = load_component_config(config_dir, "baselines", args.baseline)
    baseline_hash = config_hash(baseline_cfg)
    runtime_env = _runtime_env_snapshot()

    attack_rows = load_jsonl_rows(attack_results)
    requested_ids: set[str] = set()
    if args.instance_id:
        requested_ids.update(x.strip() for x in str(args.instance_id).split(",") if x.strip())
    if args.instance_id_file:
        id_file = Path(args.instance_id_file)
        requested_ids.update(line.strip() for line in id_file.read_text(encoding="utf-8").splitlines() if line.strip())
    if requested_ids:
        attack_rows = [row for row in attack_rows if str(row.get("instance_id", "")) in requested_ids]
    if args.limit is not None and args.limit > 0:
        attack_rows = attack_rows[: args.limit]
    all_ids = [str(row.get("instance_id", "") or "") for row in attack_rows]
    all_ids = [instance_id for instance_id in all_ids if instance_id]
    rows_by_id = {str(row.get("instance_id", "") or ""): row for row in attack_rows}

    shard_base = out_dir / "_shards"
    repo_copy_root = Path(args.repo_copy_root) if args.repo_copy_root else out_dir / "_repo_copies"
    if args.isolate_repos and args.refresh_repo_copies and repo_copy_root.exists():
        print(f"[defense-shards] refresh repo copies: {repo_copy_root}", flush=True)
        shutil.rmtree(repo_copy_root)
    shard_result_paths = sorted(shard_base.glob("shard_*/results.jsonl"))
    completed = _completed_ids([final_results, *shard_result_paths], baseline_hash)
    remaining = [instance_id for instance_id in all_ids if instance_id not in completed]

    print(
        f"[defense-shards] baseline={args.baseline} total={len(all_ids)} "
        f"completed={len(completed)} remaining={len(remaining)} shards={args.shards} parallel={args.parallel}",
        flush=True,
    )

    commands: List[tuple[List[str], Path]] = []
    for idx, ids in enumerate(_chunks(remaining, max(1, int(args.shards)))):
        shard_out = shard_base / f"shard_{idx:02d}"
        shard_attack_results = attack_results
        shard_instance_arg: List[str] = ["--instance-id", ",".join(ids)]
        if args.isolate_repos:
            shard_rows = [rows_by_id[instance_id] for instance_id in ids]
            shard_attack_results = _write_isolated_attack_results(
                rows=shard_rows,
                shard_out=shard_out,
                repo_copy_root=repo_copy_root,
                refresh_repo_copies=bool(args.refresh_repo_copies),
            )
            shard_instance_arg = []

        cmd = [
            sys.executable,
            "-m",
            "src.eval.cli",
            "run_defense",
            "--attack-results",
            str(shard_attack_results),
            "--baseline",
            args.baseline,
            "--fidelity-mode",
            args.fidelity_mode,
            "--out",
            str(shard_out),
            "--config-dir",
            str(config_dir),
            "--max-patch-attempts",
            str(max(1, int(args.max_patch_attempts))),
        ]
        cmd.extend(shard_instance_arg)
        if args.retry_on_apply_failure:
            cmd.append("--retry-on-apply-failure")
        else:
            cmd.append("--no-retry-on-apply-failure")
        commands.append((cmd, shard_out / "results.jsonl"))

    if commands:
        returncodes = _run_commands(
            commands,
            parallel=max(1, int(args.parallel)),
            env=os.environ.copy(),
            stale_timeout_sec=max(0, int(args.stale_timeout_sec)),
            poll_interval_sec=max(1, int(args.poll_interval_sec)),
            total_remaining=len(remaining),
        )
        if any(rc != 0 for rc in returncodes):
            print(f"[defense-shards] one or more shards failed: {returncodes}", file=sys.stderr, flush=True)
            shard_result_paths = sorted(shard_base.glob("shard_*/results.jsonl"))
            merged = _merge_results(final_results, shard_result_paths, baseline_hash)
            atomic_write_json(
                out_dir / "shard_manifest.json",
                {
                    "attack_results": str(attack_results),
                    "baseline": args.baseline,
                    "baseline_config_hash": baseline_hash,
                    "fidelity_mode": args.fidelity_mode,
                    "total_instances": len(all_ids),
                    "completed_instances": merged,
                    "shards": int(args.shards),
                    "parallel": int(args.parallel),
                    "shard_results": [str(path) for path in shard_result_paths],
                    "status": "partial_failed",
                    "returncodes": returncodes,
                    "stale_timeout_sec": int(args.stale_timeout_sec),
                    "isolate_repos": bool(args.isolate_repos),
                    "runtime_env": runtime_env,
                },
            )
            print(f"[defense-shards] partial merge={merged} final_results={final_results}", flush=True)
            return 1

    shard_result_paths = sorted(shard_base.glob("shard_*/results.jsonl"))
    merged = _merge_results(final_results, shard_result_paths, baseline_hash)
    atomic_write_json(
        out_dir / "shard_manifest.json",
        {
            "attack_results": str(attack_results),
            "baseline": args.baseline,
            "baseline_config_hash": baseline_hash,
            "fidelity_mode": args.fidelity_mode,
            "total_instances": len(all_ids),
            "completed_instances": merged,
            "shards": int(args.shards),
            "parallel": int(args.parallel),
            "shard_results": [str(path) for path in shard_result_paths],
            "status": "complete",
            "stale_timeout_sec": int(args.stale_timeout_sec),
            "isolate_repos": bool(args.isolate_repos),
            "repo_copy_root": str(repo_copy_root) if args.isolate_repos else "",
            "cleanup_repo_copies": bool(args.cleanup_repo_copies),
            "runtime_env": runtime_env,
        },
    )
    if args.isolate_repos and args.cleanup_repo_copies and repo_copy_root.exists():
        print(f"[defense-shards] cleanup repo copies: {repo_copy_root}", flush=True)
        shutil.rmtree(repo_copy_root)
    print(f"[defense-shards] merged={merged} final_results={final_results}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
