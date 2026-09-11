"""Defense runner for publication rows."""

from __future__ import annotations

import json
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import src.baseline  # noqa: F401 - register baseline plugins

from src.baseline.registry import get_baseline
from src.common.artifact_store import atomic_write_json
from src.common.config import config_hash, load_component_config
from src.common.diff import apply_unified_diff_detailed, looks_like_unified_diff
from src.common.hashing import sha256_text
from src.common.rows import (
    infer_benchmark,
    load_rows,
    resolve_repo_path,
    copy_repo_without_git,
    write_jsonl,
)
from src.common.llm import LLMClient
from src.common.subprocess import command_exists, run_command


_WORKER_BASELINE = threading.local()


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


def _row_key(row: Dict[str, Any], row_index: int, baseline_hash: str, execution_hash: str = "") -> str:
    return sha256_text(
        json.dumps(
            {
                "row_index": row_index,
                "code_base": row["code_base"],
                "label": int(row["label"]),
                "prompt_hash": sha256_text(row["prompt"]),
                "patch_hash": sha256_text(row["patch"]),
                "baseline_config_hash": baseline_hash,
                "execution_hash": execution_hash,
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
            if row.get("baseline_config_hash") == baseline_hash and row.get("row_key") and row.get("status") == "success":
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


def _extract_added_python_files(patch_text: str) -> Dict[str, str]:
    """Best-effort patch-to-Python-materialization fallback."""

    files: Dict[str, List[str]] = {}
    current = "candidate.py"
    saw_diff = False
    for line in (patch_text or "").splitlines():
        if line.startswith("+++ "):
            raw = line[4:].strip().split("\t", 1)[0]
            if raw != "/dev/null":
                current = raw[2:] if raw.startswith("b/") else raw
            saw_diff = True
            continue
        if line.startswith("+") and not line.startswith("+++"):
            if current.endswith(".py"):
                files.setdefault(current, []).append(line[1:])
            continue

    if not saw_diff and patch_text.strip():
        files["candidate.py"] = [patch_text]

    return {
        name if name.endswith(".py") else f"{name}.py": "\n".join(lines).rstrip() + "\n"
        for name, lines in files.items()
        if any(line.strip() for line in lines)
    }


def _write_patch_snippet_repo(root: Path, patch_text: str) -> Dict[str, Any]:
    files = _extract_added_python_files(patch_text)
    if not files and patch_text.strip():
        files = {"candidate.py": patch_text.rstrip() + "\n"}
    for rel, content in files.items():
        path = root / rel
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Patch path escapes scan workspace: {rel}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return {
        "mode": "patch_snippets",
        "file_count": len(files),
        "files": sorted(files),
    }


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
    if not isinstance(payload, dict):
        return 0, 0, f"{tool}_report_not_object", payload
    results = payload.get("results")
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
    row: Dict[str, Any],
    repo_path: Path | None,
    work_root: Path,
) -> Dict[str, Any]:
    patch_text = row["patch"]
    if not patch_text.strip():
        raise ValueError("empty_patch: a scanner cannot evaluate a missing candidate")
    if repo_path is not None and not repo_path.is_dir():
        raise FileNotFoundError(f"Benchmark repository missing: {repo_path}")
    if repo_path and repo_path.exists() and repo_path.is_dir() and looks_like_unified_diff(patch_text):
        copy_repo_without_git(repo_path, work_root)
        apply_details = apply_unified_diff_detailed(work_root, patch_text)
        if bool(apply_details.get("applied")):
            return {
                "scan_root": work_root,
                "input_mode": "isolated_repo_after_patch",
                "repo_path": str(repo_path),
                "patch_apply": apply_details,
            }
        raise ValueError(f"Patch does not apply to {repo_path}: {apply_details.get('reason_code')}")

    work_root.mkdir(parents=True, exist_ok=True)
    snippet_details = _write_patch_snippet_repo(work_root, patch_text)
    return {
        "scan_root": work_root,
        "input_mode": "patch_snippets",
        "repo_path": str(repo_path) if repo_path else "",
        "snippet_materialization": snippet_details,
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
        return "error", {"tool": tool, "error": "empty_command"}

    available, availability_signals = _static_tool_available(base_command[0])
    if not available:
        return "error", {
            **availability_signals,
            "tool": tool,
            "error": availability_signals["failure_reason"],
        }

    artifact_dir = out_dir / "artifacts" / "static" / f"{_safe_filename(row['code_base'])}_{row_index:06d}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / f"{tool}.json"
    report_path.unlink(missing_ok=True)
    max_findings = int(config.get("max_findings", config.get("max_new_findings", 0)))
    baseline_findings = 0
    baseline_payload = None

    with tempfile.TemporaryDirectory(prefix=f"{tool}-row-defense-") as tmp:
        work_root = Path(tmp) / "repo"
        prep = _prepare_static_workspace(row=row, repo_path=repo_path, work_root=work_root)
        scan_root = Path(prep["scan_root"])
        if "max_new_findings" in config and "max_findings" not in config and prep["input_mode"] == "isolated_repo_after_patch":
            before_root = Path(tmp) / "before"
            copy_repo_without_git(repo_path, before_root)
            before_report = artifact_dir / f"{tool}_before.json"
            before_report.unlink(missing_ok=True)
            before_command = (
                _build_bandit_command(base_command, before_report)
                if tool == "bandit"
                else _build_semgrep_command(base_command, before_report)
            )
            before_result = run_command(before_command, cwd=before_root, timeout_sec=timeout_sec)
            baseline_findings, before_errors, before_parse_error, baseline_payload = _parse_static_report(before_report, tool)
            if before_result.returncode not in {0, 1} or before_errors or before_parse_error:
                raise RuntimeError(f"{tool} baseline scan failed: {before_parse_error or before_errors or before_result.returncode}")
        command = (
            _build_bandit_command(base_command, report_path)
            if tool == "bandit"
            else _build_semgrep_command(base_command, report_path)
        )
        result = run_command(command, cwd=scan_root, timeout_sec=timeout_sec)
        findings, errors_count, parse_error, parsed = _parse_static_report(report_path, tool)
        total_findings = findings
        if baseline_payload is not None and not parse_error:
            findings = sum((_finding_counts(parsed) - _finding_counts(baseline_payload)).values())

    # Keep a copy of the scanner output outside the temporary workspace.
    if parsed is not None:
        atomic_write_json(artifact_dir / f"{tool}_parsed.json", parsed if isinstance(parsed, dict) else {"payload": parsed})

    allowed_returncodes = {0, 1}  # Both scanners may exit 1 when findings exist.
    error = parse_error or (f"{tool}_scan_errors:{errors_count}" if errors_count else "")
    if result.returncode not in allowed_returncodes:
        error = error or f"{tool}_exit_{result.returncode}"
    decision = "error" if error else ("reject" if findings > max_findings else "accept")
    signals = {
        "tool": tool,
        **availability_signals,
        "command": command,
        "returncode": result.returncode,
        "timeout_sec": timeout_sec,
        "findings": findings,
        "total_findings": total_findings,
        "baseline_findings": baseline_findings,
        "errors_count": errors_count,
        "parse_error": parse_error,
        "max_findings": max_findings,
        "decision_rule": "error_if_scan_incomplete_else_reject_if_findings_gt_threshold",
        "failure_reason": error,
        "error": error,
        "stdout_preview": (result.stdout or "")[:1000],
        "stderr_preview": (result.stderr or "")[:1000],
        "artifact_dir": str(artifact_dir),
        "report_path": str(report_path),
        "input_preparation": {
            key: value
            for key, value in prep.items()
            if key != "scan_root"
        },
    }
    return decision, signals


def _finding_counts(payload: Dict[str, Any]) -> Counter:
    """Compare findings without treating line shifts as newly introduced issues."""
    def identity(finding):
        extra = finding.get("extra", {})
        code = finding.get("code", extra.get("lines", ""))
        code = "\n".join(re.sub(r"^\s*\d+\s+", "", line).strip() for line in code.splitlines())
        return (
            str(finding.get("filename", finding.get("path", ""))).removeprefix("./"),
            finding.get("test_id", finding.get("check_id", "")),
            code,
        )
    return Counter(identity(finding) for finding in payload.get("results", []))


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
    baseline_plugin: str,
    baseline_config: Dict[str, Any],
    baseline_hash: str,
    fidelity_mode: str,
    out_dir: Path,
    repo_path: Path | None,
) -> Tuple[str, Dict[str, Any]]:
    # Local transformer guards load model weights in __init__. Reuse one instance
    # per worker, while keeping mutable last_signals isolated between threads.
    cache_key = (baseline_plugin, baseline_hash, str(out_dir), fidelity_mode)
    if getattr(_WORKER_BASELINE, "key", None) != cache_key:
        llm_client = LLMClient(out_dir / "artifacts" / "llm_cache")
        _WORKER_BASELINE.instance = get_baseline(baseline_plugin)(
            baseline_config, llm_client, baseline_hash, out_dir, fidelity_mode,
        )
        _WORKER_BASELINE.key = cache_key
    baseline_obj = _WORKER_BASELINE.instance
    baseline_obj.last_signals = {}
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
    fidelity_mode: str,
    out_dir: Path,
    repos_root: Path | None,
    scanner_timeout_sec: int,
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
                baseline_plugin=baseline_plugin,
                baseline_config=baseline_config,
                baseline_hash=baseline_hash,
                fidelity_mode=fidelity_mode,
                out_dir=out_dir,
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
    results_path.unlink(missing_ok=True)

    # Training inputs and graph artifacts can change in place with the same YAML.
    # Recompute dataset-level evaluations instead of reusing unverified metrics.

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
    execution_hash = config_hash({
        "fidelity_mode": fidelity_mode,
        "repos_root": str(repos_root or ""),
        "scanner_timeout_sec": scanner_timeout_sec,
    })

    indexed_rows = [
        (idx, row, _row_key(row, idx, baseline_hash, execution_hash))
        for idx, row in enumerate(rows)
    ]
    pending = [(idx, row, key) for idx, row, key in indexed_rows if key not in completed]
    current_keys = {key for _, _, key in indexed_rows}
    retained = {
        result["row_key"]: result
        for result in _read_results_jsonl(results_path)
        if result.get("row_key") in current_keys and result.get("status") == "success"
    }
    # Keep one successful result per current row; retries replace failed records.
    retained_path = results_path.with_suffix(".resume.tmp")
    write_jsonl(retained_path, retained.values())
    retained_path.replace(results_path)
    print(
        f"[run_defense] start rows={len(rows)} pending={len(pending)} "
        f"baseline={baseline_name} plugin={baseline_plugin} workers={workers}",
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
        "execution_hash": execution_hash,
        "repos_root": str(repos_root or ""),
        "workers": workers,
    }
    atomic_write_json(out_dir / "integration_spec.json", integration_spec)

    write_lock = threading.Lock()
    new_results: List[Dict[str, Any]] = []
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
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
                    fidelity_mode=fidelity_mode,
                    out_dir=out_dir,
                    repos_root=repos_root,
                    scanner_timeout_sec=scanner_timeout_sec,
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
