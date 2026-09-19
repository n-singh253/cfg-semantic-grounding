"""Defense runner for publication rows."""

from __future__ import annotations

import json
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import src.baseline  # noqa: F401 - register baseline plugins

from src.baseline.registry import get_baseline
from src.common.artifact_store import atomic_write_json
from src.common.config import config_hash, load_component_config
from src.common.diff import apply_unified_diff_detailed
from src.common.hashing import sha256_text
from src.common.rows import (
    infer_benchmark,
    load_rows,
    reset_git_repo,
    resolve_repo_path,
    copy_repo_without_git,
)
from src.common.llm import LLMClient
from src.common.subprocess import command_exists, run_command


_REPO_LOCKS: Dict[str, threading.Lock] = {}
_REPO_LOCKS_GUARD = threading.Lock()


class _ThreadLocalBaselinePool:
    """Create one reusable baseline instance per executor worker thread."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._local = threading.local()

    def get(self) -> Any:
        if hasattr(self._local, "instance"):
            return self._local.instance
        if hasattr(self._local, "load_error"):
            raise self._local.load_error
        try:
            self._local.instance = self._factory()
        except Exception as exc:
            self._local.load_error = exc
            raise
        return self._local.instance


def _effective_worker_count(baseline_plugin: str, requested_workers: int) -> int:
    workers = max(1, int(requested_workers))
    if baseline_plugin == "llama_guard":
        return 1
    return workers


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "unknown"


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _prompt_type(label: int) -> str:
    return "adv" if int(label) == 1 else "ori"


def _artifact_instance_id(code_base: str, row_index: int, label: int) -> str:
    return f"{_safe_filename(code_base)}__{_prompt_type(label)}__row_{row_index:06d}"


def _row_key(row: Dict[str, Any], row_index: int, baseline_hash: str) -> str:
    return sha256_text(
        json.dumps(
            {
                "row_index": row_index,
                "code_base": row["code_base"],
                "label": int(row["label"]),
                "prompt_hash": sha256_text(row["prompt"]),
                "patch_hash": sha256_text(row["patch"]),
                "baseline_config_hash": baseline_hash,
            },
            sort_keys=True,
        )
    )


def _load_completed_keys(results_path: Path, baseline_hash: str) -> set[str]:
    if not results_path.exists():
        return set()
    keys: set[str] = set()
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip().lstrip("\x00")
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("baseline_config_hash") == baseline_hash and row.get("row_key"):
                keys.add(str(row["row_key"]))
    return keys


def _read_results_jsonl(results_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not results_path.exists():
        return rows
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip().lstrip("\x00")
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl_locked(path: Path, row: Dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()


def _build_bandit_command(base_command: List[str], report_path: Path) -> List[str]:
    cmd: List[str] = []
    saw_output = False
    saw_format = False
    i = 0
    while i < len(base_command):
        token = str(base_command[i])
        if token in {"-o", "--output"}:
            cmd.extend([token, str(report_path)])
            saw_output = True
            i += 2
            continue
        if token.startswith("--output="):
            cmd.append(f"--output={report_path}")
            saw_output = True
            i += 1
            continue
        if token in {"-f", "--format"}:
            cmd.extend([token, "json"])
            saw_format = True
            i += 2
            continue
        if token.startswith("--format="):
            cmd.append("--format=json")
            saw_format = True
            i += 1
            continue
        cmd.append(token)
        i += 1
    if not saw_format:
        cmd.extend(["-f", "json"])
    if not saw_output:
        cmd.extend(["-o", str(report_path)])
    return cmd


def _build_semgrep_command(base_command: List[str], report_path: Path) -> List[str]:
    cmd: List[str] = []
    saw_json = False
    saw_output = False
    i = 0
    while i < len(base_command):
        token = str(base_command[i])
        if token == "--json":
            cmd.append(token)
            saw_json = True
            i += 1
            continue
        if token in {"-o", "--output"}:
            cmd.extend([token, str(report_path)])
            saw_output = True
            i += 2
            continue
        if token.startswith("--output="):
            cmd.append(f"--output={report_path}")
            saw_output = True
            i += 1
            continue
        cmd.append(token)
        i += 1
    if not saw_json:
        cmd.append("--json")
    if not saw_output:
        cmd.extend(["--output", str(report_path)])
    return cmd


def _parse_static_report(report_path: Path, tool: str) -> Tuple[int, int, str, Any]:
    if not report_path.exists():
        return 0, 0, f"{tool}_report_missing", None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return 0, 0, f"{type(exc).__name__}: {exc}", None
    results = payload.get("results", [])
    errors = payload.get("errors", [])
    findings = len(results) if isinstance(results, list) else 0
    error_count = len(errors) if isinstance(errors, list) else 0
    parse_error = "" if isinstance(results, list) else f"{tool}_results_not_list"
    return findings, error_count, parse_error, payload


def _static_tool_available(tool: str) -> Tuple[bool, Dict[str, Any]]:
    if not command_exists(tool):
        return False, {"available": False, "failure_reason": f"{tool}_not_on_path"}
    result = run_command([tool, "--version"], timeout_sec=30)
    if result.returncode != 0:
        return (
            False,
            {
                "available": False,
                "failure_reason": f"{tool}_version_failed",
                "version_returncode": result.returncode,
                "version_stdout_preview": (result.stdout or "")[:1000],
                "version_stderr_preview": (result.stderr or "")[:1000],
            },
        )
    return (
        True,
        {
            "available": True,
            "version": (result.stdout or result.stderr or "").strip().splitlines()[0]
            if (result.stdout or result.stderr or "").strip()
            else "",
        },
    )


def _prepare_static_workspace(
    *,
    repo_path: Path | None,
    work_root: Path,
) -> Dict[str, Any]:
    if repo_path is None or not repo_path.is_dir():
        return {
            "ok": False,
            "failure_reason": "invalid_repo_path",
            "repo_path": str(repo_path or ""),
        }

    key = str(repo_path.resolve())
    with _REPO_LOCKS_GUARD:
        repo_lock = _REPO_LOCKS.setdefault(key, threading.Lock())

    with repo_lock:
        reset_before = reset_git_repo(repo_path)
        if not reset_before.get("ok"):
            return {
                "ok": False,
                "failure_reason": "initial_reset_failed",
                "repo_path": str(repo_path),
                "source_reset_before": reset_before,
            }
        try:
            copy_repo_without_git(repo_path, work_root)
            copy_error = ""
        except Exception as exc:
            copy_error = f"{type(exc).__name__}: {exc}"
        reset_after = reset_git_repo(repo_path)

        if copy_error:
            return {
                "ok": False,
                "failure_reason": "repo_copy_failed",
                "copy_error": copy_error,
                "repo_path": str(repo_path),
                "source_reset_before": reset_before,
                "source_reset_after": reset_after,
            }
        if not reset_after.get("ok"):
            return {
                "ok": False,
                "failure_reason": "final_reset_failed",
                "repo_path": str(repo_path),
                "source_reset_before": reset_before,
                "source_reset_after": reset_after,
            }
        return {
            "ok": True,
            "scan_root": work_root,
            "input_mode": "isolated_clean_repo",
            "repo_path": str(repo_path),
            "source_reset_before": reset_before,
            "source_reset_after": reset_after,
        }


def _run_static_scanner(
    *,
    tool: str,
    row: Dict[str, Any],
    row_index: int,
    config: Dict[str, Any],
    out_dir: Path,
    repo_path: Path | None,
    timeout_sec: int,
) -> Tuple[str, Dict[str, Any]]:
    base_command = [str(part) for part in config.get("command", [tool, "-r", ".", "-f", "json"])]
    if not base_command:
        return "error", {
            "tool": tool,
            "failure_reason": "empty_command",
            "stage_failed": True,
        }
    if repo_path is None or not repo_path.is_dir():
        return "error", {
            "tool": tool,
            "failure_reason": "invalid_repo_path",
            "repo_path": str(repo_path or ""),
            "stage_failed": True,
        }
    if not row["patch"].strip():
        return "error", {
            "tool": tool,
            "failure_reason": "empty_patch",
            "stage_failed": True,
        }

    available, availability_signals = _static_tool_available(base_command[0])
    if not available:
        behavior = str(config.get("missing_tool_behavior", "reject")).strip().lower()
        decision = "accept" if behavior == "accept" else "reject"
        availability_signals.update({"tool": tool, "behavior": behavior})
        return decision, {
            **availability_signals,
        }

    artifact_dir = out_dir / "artifacts" / "static" / f"{_safe_filename(row['code_base'])}_{row_index:06d}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    before_report_path = artifact_dir / f"{tool}_before.json"
    after_report_path = artifact_dir / f"{tool}.json"
    before_report_path.unlink(missing_ok=True)
    after_report_path.unlink(missing_ok=True)
    max_new_findings = int(config.get("max_new_findings", config.get("max_findings", 0)))

    prep: Dict[str, Any] = {}
    patch_apply: Dict[str, Any] = {
        "applied": False,
        "method_used": "none",
        "reason_code": "not_attempted",
    }
    before_command: List[str] = []
    after_command: List[str] = []
    before_result = None
    after_result = None
    before_findings = 0
    after_findings = 0
    before_errors = 0
    after_errors = 0
    before_parse_error = ""
    after_parse_error = ""
    before_parsed = None
    after_parsed = None
    failure_reason = ""

    with tempfile.TemporaryDirectory(prefix=f"{tool}-row-defense-") as tmp:
        work_root = Path(tmp) / "repo"
        prep = _prepare_static_workspace(repo_path=repo_path, work_root=work_root)
        if not prep.get("ok"):
            failure_reason = str(prep.get("failure_reason", "workspace_preparation_failed"))
        else:
            before_command = (
                _build_bandit_command(base_command, before_report_path)
                if tool == "bandit"
                else _build_semgrep_command(base_command, before_report_path)
            )
            before_result = run_command(before_command, cwd=work_root, timeout_sec=timeout_sec)
            if before_result.returncode == 124:
                failure_reason = f"before_{tool}_timeout"
            else:
                before_findings, before_errors, before_parse_error, before_parsed = (
                    _parse_static_report(before_report_path, tool)
                )
                if before_parse_error:
                    failure_reason = f"before_{tool}_parse_error"

        if not failure_reason:
            patch_apply = apply_unified_diff_detailed(work_root, row["patch"])
            if not patch_apply.get("applied"):
                failure_reason = "patch_apply_failed"

        if not failure_reason:
            after_command = (
                _build_bandit_command(base_command, after_report_path)
                if tool == "bandit"
                else _build_semgrep_command(base_command, after_report_path)
            )
            after_result = run_command(after_command, cwd=work_root, timeout_sec=timeout_sec)
            if after_result.returncode == 124:
                failure_reason = f"after_{tool}_timeout"
            else:
                after_findings, after_errors, after_parse_error, after_parsed = (
                    _parse_static_report(after_report_path, tool)
                )
                if after_parse_error:
                    failure_reason = f"after_{tool}_parse_error"

    if before_parsed is not None:
        atomic_write_json(
            artifact_dir / f"{tool}_before_parsed.json",
            before_parsed if isinstance(before_parsed, dict) else {"payload": before_parsed},
        )
    if after_parsed is not None:
        atomic_write_json(
            artifact_dir / f"{tool}_parsed.json",
            after_parsed if isinstance(after_parsed, dict) else {"payload": after_parsed},
        )

    new_findings = None if failure_reason else after_findings - before_findings
    if failure_reason:
        decision = "error"
    else:
        decision = "accept" if new_findings <= max_new_findings else "reject"
    signals = {
        "tool": tool,
        **availability_signals,
        "repo_path": str(repo_path),
        "before_command": before_command,
        "after_command": after_command,
        "before_returncode": before_result.returncode if before_result else None,
        "after_returncode": after_result.returncode if after_result else None,
        "timeout_sec": timeout_sec,
        "before_findings": before_findings,
        "after_findings": after_findings if after_result else None,
        "new_findings": new_findings,
        "before_errors_count": before_errors,
        "after_errors_count": after_errors if after_result else None,
        "before_parse_error": before_parse_error,
        "after_parse_error": after_parse_error,
        "max_new_findings": max_new_findings,
        "decision_rule": "accept_if_after_findings_le_before_findings_plus_max_new_findings",
        "failure_reason": failure_reason,
        "stage_failed": bool(failure_reason),
        "before_stdout_preview": (before_result.stdout or "")[:1000] if before_result else "",
        "before_stderr_preview": (before_result.stderr or "")[:1000] if before_result else "",
        "after_stdout_preview": (after_result.stdout or "")[:1000] if after_result else "",
        "after_stderr_preview": (after_result.stderr or "")[:1000] if after_result else "",
        "artifact_dir": str(artifact_dir),
        "before_report_path": str(before_report_path),
        "after_report_path": str(after_report_path),
        "patch_apply": patch_apply,
        "input_preparation": {
            key: value
            for key, value in prep.items()
            if key != "scan_root"
        },
    }
    return decision, signals


def _decision_from_defense_result(decision_raw: Any, signals: Dict[str, Any]) -> str:
    if signals.get("error") or signals.get("stage_failed"):
        return "error"
    if signals.get("extract_only"):
        return "extract"
    if decision_raw is False:
        return "reject"
    if isinstance(decision_raw, str):
        return "edit"
    return "accept"


def _evaluate_plugin_row(
    *,
    row: Dict[str, Any],
    row_index: int,
    baseline_name: str,
    baseline_pool: _ThreadLocalBaselinePool,
    repo_path: Path | None,
) -> Tuple[str, Dict[str, Any]]:
    baseline_obj = baseline_pool.get()
    code_base = row["code_base"]
    label = int(row["label"])
    artifact_instance_id = _artifact_instance_id(code_base, row_index, label)
    repo_code = {
        "instance_id": artifact_instance_id,
        "code_base": code_base,
        "dataset": infer_benchmark(code_base),
        "split": "",
        "repo_id": code_base,
        "base_commit": "HEAD",
        "path": str(repo_path or ""),
        "prompt_type": _prompt_type(label),
        "label_hint": label,
        "patch_hash": sha256_text(row["patch"]),
        "prompt_hash": sha256_text(row["prompt"]),
        "baseline_name": baseline_name,
        "row_index": row_index,
    }
    decision_raw = baseline_obj.defense(row["prompt"], row["patch"], [], repo_code)
    signals = dict(getattr(baseline_obj, "last_signals", {}))
    decision = _decision_from_defense_result(decision_raw, signals)
    return decision, signals


def _evaluate_one_row(
    *,
    row: Dict[str, Any],
    row_index: int,
    total: int,
    row_key: str,
    baseline_name: str,
    baseline_plugin: str,
    baseline_config: Dict[str, Any],
    baseline_hash: str,
    out_dir: Path,
    repos_root: Path | None,
    scanner_timeout_sec: int,
    baseline_pool: _ThreadLocalBaselinePool,
) -> Dict[str, Any]:
    t0 = time.time()
    start_ts = _utc_now()
    code_base = row["code_base"]
    label = int(row["label"])
    repo_path = resolve_repo_path(code_base, repos_root)

    print(
        f"[run_defense] {row_index + 1}/{total} {code_base} "
        f"label={label} start",
        flush=True,
    )
    try:
        if baseline_plugin == "static_bandit":
            decision, signals = _run_static_scanner(
                tool="bandit",
                row=row,
                row_index=row_index,
                config=baseline_config,
                out_dir=out_dir,
                repo_path=repo_path,
                timeout_sec=scanner_timeout_sec,
            )
        elif baseline_plugin == "static_semgrep":
            decision, signals = _run_static_scanner(
                tool="semgrep",
                row=row,
                row_index=row_index,
                config=baseline_config,
                out_dir=out_dir,
                repo_path=repo_path,
                timeout_sec=scanner_timeout_sec,
            )
        else:
            decision, signals = _evaluate_plugin_row(
                row=row,
                row_index=row_index,
                baseline_name=baseline_name,
                baseline_pool=baseline_pool,
                repo_path=repo_path,
            )
        error = str(signals.get("error") or "")
        status = "error" if error or signals.get("stage_failed") else "success"
    except Exception as exc:
        decision = "error"
        signals = {}
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    result = {
        "row_key": row_key,
        "row_index": row_index,
        "artifact_instance_id": _artifact_instance_id(code_base, row_index, label),
        "code_base": code_base,
        "label": label,
        "prompt_hash": sha256_text(row["prompt"]),
        "patch_hash": sha256_text(row["patch"]),
        "patch_bytes": len(row["patch"].encode("utf-8", errors="replace")),
        "baseline_name": baseline_name,
        "baseline_plugin": baseline_plugin,
        "baseline_config_hash": baseline_hash,
        "decision": decision,
        "status": status,
        "error": error,
        "repo_path": str(repo_path or ""),
        "defense_signals": signals,
        "timestamp_defense_start": start_ts,
        "timestamp_defense_end": _utc_now(),
        "defense_runtime_sec": round(time.time() - t0, 6),
    }
    print(
        f"[run_defense] {row_index + 1}/{total} {code_base} "
        f"label={label} done={status} decision={decision} "
        f"runtime_sec={result['defense_runtime_sec']}",
        flush=True,
    )
    return result


def _baseline_scope(config: Dict[str, Any]) -> str:
    return str(config.get("scope", config.get("run_level", "row"))).strip().lower()


def _run_dataset_baseline(
    *,
    rows_path: Path | None,
    baseline_name: str,
    baseline_plugin: str,
    baseline_config: Dict[str, Any],
    baseline_hash: str,
    fidelity_mode: str,
    out_dir: Path,
    resume: bool,
) -> List[Dict[str, Any]]:
    results_path = out_dir / "results.jsonl"
    if not resume and results_path.exists():
        results_path.unlink()

    existing = [
        row
        for row in _read_results_jsonl(results_path)
        if row.get("baseline_config_hash") == baseline_hash
    ]
    if resume and existing:
        print(
            f"[run_defense] start baseline={baseline_name} plugin={baseline_plugin} "
            f"scope=dataset pending=0 resumed={len(existing)}",
            flush=True,
        )
        print(
            f"[run_defense] complete baseline={baseline_name} "
            f"new=0 resumed={len(existing)}",
            flush=True,
        )
        return existing

    print(
        f"[run_defense] start baseline={baseline_name} plugin={baseline_plugin} "
        "scope=dataset pending=1",
        flush=True,
    )
    llm_client = LLMClient(out_dir / "artifacts" / "llm_cache")
    baseline_obj = get_baseline(baseline_plugin)(
        baseline_config,
        llm_client,
        baseline_hash,
        out_dir,
        fidelity_mode,
    )
    run_dataset = getattr(baseline_obj, "run_dataset", None)
    if run_dataset is None:
        raise ValueError(
            f"Baseline {baseline_name} declares scope=dataset but plugin "
            f"{baseline_plugin} has no run_dataset method."
        )
    results = run_dataset(
        rows_path=rows_path,
        baseline_name=baseline_name,
        baseline_plugin=baseline_plugin,
    )
    print(
        f"[run_defense] complete baseline={baseline_name} "
        f"new={len(results)} resumed=0",
        flush=True,
    )
    return results


def run_defense(
    *,
    rows_path: Path | None,
    baseline_name: str,
    out_dir: Path,
    config_dir: Path,
    fidelity_mode: str = "llm",
    repos_root: Path | None = None,
    workers: int = 1,
    limit: int | None = None,
    code_bases: Sequence[str] | None = None,
    recursive: bool = False,
    strict_rows: bool = False,
    resume: bool = True,
    scanner_timeout_sec: int = 120,
) -> List[Dict[str, Any]]:
    rows_path = rows_path.expanduser().resolve() if rows_path is not None else None
    out_dir = out_dir.expanduser().resolve()
    config_dir = config_dir.expanduser().resolve()
    repos_root = repos_root.expanduser().resolve() if repos_root is not None else None

    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_config_path = config_dir / "baselines" / f"{baseline_name}.yaml"
    baseline_config = load_component_config(config_dir, "baselines", baseline_name)
    baseline_hash = config_hash(baseline_config)
    baseline_plugin = str(baseline_config.get("plugin", baseline_name))
    baseline_runtime_config = dict(baseline_config)
    baseline_runtime_config["_config_path"] = str(baseline_config_path)

    if _baseline_scope(baseline_config) == "dataset":
        return _run_dataset_baseline(
            rows_path=rows_path,
            baseline_name=baseline_name,
            baseline_plugin=baseline_plugin,
            baseline_config=baseline_runtime_config,
            baseline_hash=baseline_hash,
            fidelity_mode=fidelity_mode,
            out_dir=out_dir,
            resume=resume,
        )

    if rows_path is None:
        raise ValueError(f"--rows is required for row-level baseline {baseline_name}")

    rows = load_rows(
        rows_path,
        recursive=recursive,
        strict_fields=strict_rows,
        limit=limit,
        code_bases=code_bases,
    )
    results_path = out_dir / "results.jsonl"
    if not resume and results_path.exists():
        results_path.unlink()
    completed = _load_completed_keys(results_path, baseline_hash) if resume else set()

    indexed_rows = [
        (idx, row, _row_key(row, idx, baseline_hash))
        for idx, row in enumerate(rows)
    ]
    pending = [(idx, row, key) for idx, row, key in indexed_rows if key not in completed]
    requested_workers = max(1, int(workers))
    effective_workers = _effective_worker_count(baseline_plugin, requested_workers)
    if effective_workers != requested_workers:
        print(
            f"[run_defense] baseline={baseline_name} requested_workers={requested_workers} "
            f"effective_workers={effective_workers} reason=single_model_instance",
            flush=True,
        )
    print(
        f"[run_defense] start rows={len(rows)} pending={len(pending)} "
        f"baseline={baseline_name} plugin={baseline_plugin} workers={effective_workers}",
        flush=True,
    )

    integration_spec = {
        "schema_version": "rows_v1",
        "rows_path": str(rows_path),
        "baseline_name": baseline_name,
        "baseline_plugin": baseline_plugin,
        "baseline_config_hash": baseline_hash,
        "baseline_config": baseline_config,
        "fidelity_mode": fidelity_mode,
        "repos_root": str(repos_root or ""),
        "workers": effective_workers,
    }
    atomic_write_json(out_dir / "integration_spec.json", integration_spec)

    baseline_pool = _ThreadLocalBaselinePool(
        lambda: get_baseline(baseline_plugin)(
            baseline_runtime_config,
            LLMClient(out_dir / "artifacts" / "llm_cache"),
            baseline_hash,
            out_dir,
            fidelity_mode,
        )
    )

    write_lock = threading.Lock()
    new_results: List[Dict[str, Any]] = []
    if pending:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_one_row,
                    row=row,
                    row_index=idx,
                    total=len(rows),
                    row_key=key,
                    baseline_name=baseline_name,
                    baseline_plugin=baseline_plugin,
                    baseline_config=baseline_runtime_config,
                    baseline_hash=baseline_hash,
                    out_dir=out_dir,
                    repos_root=repos_root,
                    scanner_timeout_sec=scanner_timeout_sec,
                    baseline_pool=baseline_pool,
                ): key
                for idx, row, key in pending
            }
            for future in as_completed(futures):
                result = future.result()
                _append_jsonl_locked(results_path, result, write_lock)
                new_results.append(result)

    all_results = _read_results_jsonl(results_path)
    print(
        f"[run_defense] complete rows={len(rows)} "
        f"new={len(new_results)} resumed={len(rows) - len(pending)}",
        flush=True,
    )
    return all_results
