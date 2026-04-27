#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def log(event: str, **kwargs: Any) -> None:
    suffix = ""
    if kwargs:
        suffix = " | " + " ".join(f"{key}={value}" for key, value in sorted(kwargs.items()))
    print(f"[run-eval] {event}{suffix}", flush=True)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_attack_rows(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "attack_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Attack results not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_patch_file(run_dir: Path, instance_id: str, patch_type: str) -> Optional[Path]:
    path = run_dir / "artifacts" / "patches" / instance_id / f"{patch_type}_patch.diff"
    return path if path.exists() else None


def write_predictions(
    rows: List[Dict[str, Any]],
    run_dir: Path,
    patch_type: str,
    pred_path: Path,
    model_name: str,
) -> None:
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            instance_id = str(row["instance_id"])
            patch_path = find_patch_file(run_dir, instance_id, patch_type)
            patch_text: Optional[str] = None
            if patch_path:
                text = patch_path.read_text(encoding="utf-8", errors="replace").strip()
                patch_text = text if text else None
            handle.write(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "model_name_or_path": model_name,
                        "model_patch": patch_text,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    log("wrote-predictions", rows=len(rows), path=pred_path)


def run_swebench_eval(
    *,
    dataset_name: str,
    split: str,
    pred_path: Path,
    run_id: str,
    out_dir: Path,
    instance_ids: List[str],
    workers: int,
    timeout: int,
    cache_level: str,
    clean: bool,
) -> None:
    try:
        from swebench.harness.run_evaluation import main as swebench_main
    except ImportError:
        raise RuntimeError("swebench is not installed. Install it with: pip install swebench")

    out_dir.mkdir(parents=True, exist_ok=True)
    original_dir = Path.cwd()
    try:
        os.chdir(out_dir)
        swebench_main(
            dataset_name=dataset_name,
            split=split,
            instance_ids=list(instance_ids),
            predictions_path=str(pred_path),
            max_workers=workers,
            force_rebuild=False,
            cache_level=cache_level,
            clean=clean,
            open_file_limit=4096,
            run_id=run_id,
            timeout=timeout,
            namespace=None,
            rewrite_reports=False,
            modal=False,
            report_dir=str(out_dir),
        )
    finally:
        os.chdir(original_dir)


def parse_json_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [str(item) for item in json.loads(value)]
    return []


def empty_test_summary(test_ids: List[str]) -> Dict[str, Any]:
    return {
        "tests": test_ids,
        "passed": [],
        "failed": list(test_ids),
        "pass_count": 0,
        "fail_count": len(test_ids),
        "all_passed": False,
    }


def extract_test_summary(test_ids: List[str], tests_status: Dict[str, Any], category_key: str) -> Dict[str, Any]:
    category = tests_status.get(category_key, {})
    passed = category.get("passed", []) if isinstance(category, dict) else []
    failed = category.get("failed", []) if isinstance(category, dict) else []
    if not passed and not failed:
        passed = [test_id for test_id in test_ids if tests_status.get(test_id) == "PASSED"]
        failed = [test_id for test_id in test_ids if tests_status.get(test_id) != "PASSED"]
    return {
        "tests": test_ids,
        "passed": passed,
        "failed": failed,
        "pass_count": len(passed),
        "fail_count": len(failed),
        "all_passed": len(passed) == len(test_ids) and not failed,
    }


def parse_eval_results(
    *,
    log_dir: Path,
    run_id: str,
    instance_ids: List[str],
    patch_type: str,
    run_dir: Path,
    hf_by_id: Dict[str, Any],
) -> List[Dict[str, Any]]:
    try:
        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as exc:
        raise RuntimeError(f"Could not import swebench grading helpers: {exc}") from exc

    results: List[Dict[str, Any]] = []
    run_name = run_id.rsplit("__", 1)[0]
    model_name = f"{run_name}_{patch_type}".replace("/", "__")

    for instance_id in instance_ids:
        hf_row = hf_by_id.get(instance_id)
        if hf_row is None:
            log("warn", instance_id=instance_id, reason="missing_hf_metadata")
            continue

        fail_to_pass = parse_json_list(hf_row["FAIL_TO_PASS"])
        pass_to_pass = parse_json_list(hf_row["PASS_TO_PASS"])
        test_output_path = log_dir / "run_evaluation" / run_id / model_name / instance_id / "test_output.txt"
        candidates = [test_output_path, *list(log_dir.rglob(f"*/{instance_id}/test_output.txt"))]
        log_path = next((path for path in candidates if path.exists()), None)

        patch_file = find_patch_file(run_dir, instance_id, patch_type)
        patch_text = ""
        if patch_file:
            patch_text = patch_file.read_text(encoding="utf-8", errors="replace").strip()

        row: Dict[str, Any] = {
            "instance_id": instance_id,
            "patch_type": patch_type,
            "patch_empty": not bool(patch_text),
            "patch_applied": False,
            "fail_to_pass": empty_test_summary(fail_to_pass),
            "pass_to_pass": empty_test_summary(pass_to_pass),
            "resolved": False,
            "log_path": str(log_path) if log_path else "",
            "error": "",
        }

        if log_path is None:
            row["error"] = "no eval log found"
            results.append(row)
            continue

        try:
            test_spec = make_test_spec(hf_row)
            prediction = {
                "instance_id": instance_id,
                "model_name_or_path": "eval",
                "model_patch": patch_text or None,
            }
            report_map = get_eval_report(
                test_spec=test_spec,
                prediction=prediction,
                test_log_path=str(log_path),
                include_tests_status=True,
            )
            instance_report = report_map.get(instance_id, {})
            row["patch_applied"] = instance_report.get("patch_successfully_applied", False)
            row["resolved"] = instance_report.get("resolved", False)
            tests_status = instance_report.get("tests_status", {})
            row["fail_to_pass"] = extract_test_summary(fail_to_pass, tests_status, "FAIL_TO_PASS")
            row["pass_to_pass"] = extract_test_summary(pass_to_pass, tests_status, "PASS_TO_PASS")
        except Exception as exc:
            row["error"] = str(exc)

        results.append(row)

    return results


def write_summary(out_dir: Path, run_name: str) -> None:
    jsonl_path = out_dir / "patch_eval.jsonl"
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    def pct(numerator: int, denominator: int) -> float:
        return round(100.0 * numerator / denominator, 2) if denominator else 0.0

    def stats(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(subset)
        resolved = sum(1 for row in subset if row.get("resolved"))
        f2p_total = sum(len(row.get("fail_to_pass", {}).get("tests", [])) for row in subset)
        f2p_passed = sum(row.get("fail_to_pass", {}).get("pass_count", 0) for row in subset)
        p2p_total = sum(len(row.get("pass_to_pass", {}).get("tests", [])) for row in subset)
        p2p_passed = sum(row.get("pass_to_pass", {}).get("pass_count", 0) for row in subset)
        return {
            "total": total,
            "empty_patch": sum(1 for row in subset if row.get("patch_empty")),
            "patch_apply_failed": sum(1 for row in subset if not row.get("patch_applied")),
            "resolved": resolved,
            "resolution_rate_pct": pct(resolved, total),
            "total_f2p_tests": f2p_total,
            "passed_f2p_tests": f2p_passed,
            "f2p_test_pass_rate_pct": pct(f2p_passed, f2p_total),
            "total_p2p_tests": p2p_total,
            "passed_p2p_tests": p2p_passed,
            "p2p_test_pass_rate_pct": pct(p2p_passed, p2p_total),
        }

    patch_types = sorted({str(row.get("patch_type", "")) for row in rows})
    summary = {
        "run_name": run_name,
        "total_rows": len(rows),
        "by_patch_type": {
            patch_type: stats([row for row in rows if row.get("patch_type") == patch_type])
            for patch_type in patch_types
        },
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log("wrote-summary", path=path)


def run_evaluation(
    *,
    dataset_name: str,
    split: str,
    run_dir: Path,
    out_dir: Path,
    patch_types: List[str],
    instance_ids_filter: Optional[List[str]],
    workers: int,
    timeout: int,
    cache_level: str,
    clean: bool,
    run_name: str,
    resume: bool,
    hf_cache: Optional[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "patch_eval.jsonl"
    pred_dir = out_dir / "predictions"
    log_dir = out_dir / "logs"

    attack_rows = load_attack_rows(run_dir)
    if instance_ids_filter:
        wanted = set(instance_ids_filter)
        attack_rows = [row for row in attack_rows if row["instance_id"] in wanted]

    all_instance_ids = [str(row["instance_id"]) for row in attack_rows]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'datasets'. Install it with: pip install datasets") from exc

    log("load-dataset", dataset=dataset_name, split=split, cache=hf_cache or "")
    dataset = load_dataset(dataset_name, split=split, cache_dir=hf_cache or None)
    hf_by_id: Dict[str, Any] = {str(row["instance_id"]): row for row in dataset}
    log("loaded-dataset", rows=len(hf_by_id))

    completed: set[tuple[str, str]] = set()
    if resume and jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if not row.get("error") and row.get("log_path"):
                        completed.add((str(row["instance_id"]), str(row["patch_type"])))
        log("resume", completed=len(completed))

    for patch_type in patch_types:
        pending_ids = [instance_id for instance_id in all_instance_ids if (instance_id, patch_type) not in completed]
        if not pending_ids:
            log("skip-patch-type", patch_type=patch_type, reason="already_complete")
            continue

        pending_set = set(pending_ids)
        pending_rows = [row for row in attack_rows if row["instance_id"] in pending_set]
        pred_path = pred_dir / f"{patch_type}_predictions.jsonl"
        eval_run_id = f"{run_name}__{patch_type}"

        write_predictions(pending_rows, run_dir, patch_type, pred_path, f"{run_name}_{patch_type}")
        log("start-harness", patch_type=patch_type, instances=len(pending_ids), run_id=eval_run_id, workers=workers)

        started = time.time()
        try:
            run_swebench_eval(
                dataset_name=dataset_name,
                split=split,
                pred_path=pred_path,
                run_id=eval_run_id,
                out_dir=out_dir,
                instance_ids=pending_ids,
                workers=workers,
                timeout=timeout,
                cache_level=cache_level,
                clean=clean,
            )
        except SystemExit as exc:
            if exc.code != 0:
                log("warn", reason="harness_exit", code=exc.code)
        except Exception as exc:
            log("error", reason="harness_failed", detail=exc)
        log("harness-done", seconds=round(time.time() - started, 3))

        results = parse_eval_results(
            log_dir=log_dir,
            run_id=eval_run_id,
            instance_ids=pending_ids,
            patch_type=patch_type,
            run_dir=run_dir,
            hf_by_id=hf_by_id,
        )

        with jsonl_path.open("a", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                f2p = row.get("fail_to_pass", {})
                p2p = row.get("pass_to_pass", {})
                log(
                    "result",
                    instance_id=row["instance_id"],
                    patch_type=patch_type,
                    resolved=row.get("resolved"),
                    f2p=f"{f2p.get('pass_count', 0)}/{len(f2p.get('tests', []))}",
                    p2p=f"{p2p.get('pass_count', 0)}/{len(p2p.get('tests', []))}",
                    error=str(row.get("error", ""))[:80],
                )

    write_summary(out_dir, run_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SWE-bench patches with the official harness")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--patch-type", choices=["ori", "adv", "both"], default="both")
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--workers", type=int, default=int(env_value("CFG_SWEBENCH_EVAL_WORKERS", default="4")))
    parser.add_argument("--timeout", type=int, default=int(env_value("CFG_SWEBENCH_EVAL_TIMEOUT", default="300")))
    parser.add_argument("--cache-level", choices=["none", "base", "env", "instance"], default=env_value("CFG_SWEBENCH_CACHE_LEVEL", default="env"))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dataset-name", default=env_value("CFG_SWEBENCH_EVAL_DATASET", default="SWE-bench/SWE-bench_Lite"))
    parser.add_argument("--split", default=env_value("CFG_SWEBENCH_EVAL_SPLIT", default="test"))
    parser.add_argument("--hf-cache", default=env_value("CFG_HF_CACHE", "HF_HOME", "HF_DATASETS_CACHE"))
    parser.add_argument("--offline", action="store_true", default=env_bool("CFG_HF_OFFLINE"))
    args = parser.parse_args()

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["HF_DATASETS_CACHE"] = args.hf_cache

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    run_dir = Path(args.run).expanduser().resolve()
    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    run_name = args.run_id or run_dir.name
    out_dir = Path(args.out).expanduser().resolve() if args.out else repo_root() / "outputs" / "eval" / run_dir.name
    patch_types = ["ori", "adv"] if args.patch_type == "both" else [args.patch_type]
    instance_ids_filter = [item.strip() for item in args.instance_id.split(",") if item.strip()] if args.instance_id else None

    run_evaluation(
        dataset_name=args.dataset_name,
        split=args.split,
        run_dir=run_dir,
        out_dir=out_dir,
        patch_types=patch_types,
        instance_ids_filter=instance_ids_filter,
        workers=args.workers,
        timeout=args.timeout,
        cache_level=args.cache_level,
        clean=bool(args.clean),
        run_name=run_name,
        resume=not args.no_resume,
        hf_cache=args.hf_cache or None,
    )


if __name__ == "__main__":
    main()
