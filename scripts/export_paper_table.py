#!/usr/bin/env python3
"""Export paper-style baseline tables from recorded results.jsonl files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_COLUMNS = ["SWE-Bench", "Feature-Bench", "LiveCodeBench"]
ATTACK_ORDER = ["SWExploit", "FCV-94", "FCV-78"]
OBFUSCATED_ATTACK_ORDER = ["Obfuscated SWExploit", "Obfuscated FCV-78"]
AGENT_ORDER = ["SWEAgent-Claude 3.7", "MINI-Gemini 3", "OpenHands-Qwen3-Coder-30B"]
METHOD_ORDER = ["LLM Judge", "Llama Guard", "Semgrep", "Bandit", "Ours"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def row_label(row: dict[str, Any]) -> int | None:
    if "mixed_eval_label" in row:
        return int(row["mixed_eval_label"])
    if "graph_label" in row:
        return int(row["graph_label"])
    attack = str(row.get("attack_name", ""))
    if attack:
        return 0 if attack == "none" else 1
    return None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    labeled = 0
    for row in rows:
        label = row_label(row)
        if label is None:
            continue
        labeled += 1
        pred = str(row.get("defense_decision", "")).lower() == "reject"
        if pred and label == 1:
            tp += 1
        elif pred and label == 0:
            fp += 1
        elif (not pred) and label == 1:
            fn += 1
        else:
            tn += 1
    accuracy = (tp + tn) / labeled if labeled else None
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    labels = Counter(row_label(row) for row in rows)
    labels.pop(None, None)
    return {
        "total_rows": len(rows),
        "labeled_rows": labeled,
        "label_counts": {str(k): v for k, v in sorted(labels.items())},
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
    }


def dataset_label(value: str) -> str:
    normalized = value.lower().replace("_", "").replace("-", "")
    if "swebench" in normalized:
        return "SWE-Bench"
    if "featurebench" in normalized:
        return "Feature-Bench"
    if "livecodebench" in normalized:
        return "LiveCodeBench"
    return value or "unknown"


def attack_label(rows: list[dict[str, Any]], run_dir: Path) -> str:
    haystack = " ".join(
        [
            str(run_dir),
            *[str(row.get("attack_name", "")) for row in rows[:20]],
            *[str(row.get("mixed_eval_condition", "")) for row in rows[:20]],
            *[str(row.get("mixed_eval_source_attack_name", "")) for row in rows[:20]],
        ]
    ).lower()
    if "swexploit_base64_obfuscated" in haystack:
        return "Obfuscated SWExploit"
    if "fcv_cwe78_base64_obfuscated" in haystack:
        return "Obfuscated FCV-78"
    if "swexploit" in haystack:
        return "SWExploit"
    if "fcv_cwe78" in haystack or "fcv78" in haystack:
        return "FCV-78"
    if "fcv_cwe94" in haystack or "fcv94" in haystack:
        return "FCV-94"
    if "fcv" in haystack:
        return "FCV"
    return "unknown"


def agent_label(agent_name: str, run_dir: Path) -> str:
    haystack = f"{agent_name} {run_dir}".lower()
    if "claude37" in haystack or "claude_3_7" in haystack or "claude-3-7" in haystack:
        return "SWEAgent-Claude 3.7"
    if "gemini3" in haystack or "gemini_3" in haystack or "gemini-3" in haystack:
        return "MINI-Gemini 3"
    if "openhands" in haystack or "qwen" in haystack:
        return "OpenHands-Qwen3-Coder-30B"
    return agent_name or "unknown"


def method_label(baseline_name: str) -> str:
    name = baseline_name.lower()
    if "llm_judge" in name:
        return "LLM Judge"
    if "llama_guard" in name or "agentic_guard" in name:
        return "Llama Guard"
    if "semgrep" in name:
        return "Semgrep"
    if "bandit" in name:
        return "Bandit"
    if "structural_misalignment" in name:
        return "Ours"
    return baseline_name


def first_integration_spec(run_dir: Path) -> dict[str, Any]:
    direct = load_json(run_dir / "integration_spec.json")
    if direct:
        return direct
    for path in sorted((run_dir / "_shards").glob("shard_*/integration_spec.json")):
        payload = load_json(path)
        if payload:
            return payload
    return {}


def summarize_result(path: Path) -> dict[str, Any] | None:
    if (
        "_shards" in path.parts
        or "_summary" in path.parts
        or any("smoke" in part.lower() for part in path.parts)
    ):
        return None
    rows = load_jsonl(path)
    if not rows:
        return None
    run_dir = path.parent
    first = rows[0]
    manifest = load_json(run_dir / "shard_manifest.json")
    spec = first_integration_spec(run_dir)
    selected = spec.get("selected_configs") if isinstance(spec.get("selected_configs"), dict) else {}
    baseline_entry = selected.get("baseline") if isinstance(selected.get("baseline"), dict) else {}
    baseline_config = baseline_entry.get("config") if isinstance(baseline_entry.get("config"), dict) else {}
    baseline_name = str(first.get("baseline_name") or baseline_entry.get("name") or path.parent.parent.name)
    dataset = dataset_label(str(first.get("dataset") or path.parts[path.parts.index("baselines") + 1]))
    attack = attack_label(rows, run_dir)
    agent = agent_label(str(first.get("agent_name", "")), run_dir)
    method = method_label(baseline_name)
    compact = {
        "threshold": baseline_config.get("threshold"),
        "reject_threshold": baseline_config.get("reject_threshold"),
        "plugin": baseline_config.get("plugin"),
        "llm_provider": (baseline_config.get("llm") or {}).get("provider") if isinstance(baseline_config.get("llm"), dict) else baseline_config.get("provider"),
        "llm_model": (baseline_config.get("llm") or {}).get("model") if isinstance(baseline_config.get("llm"), dict) else baseline_config.get("model"),
        "gnn_model_path": baseline_config.get("gnn_model_path") or baseline_config.get("model_path"),
    }
    return {
        "run_dir": str(run_dir),
        "results_path": str(path),
        "dataset": dataset,
        "attack": attack,
        "agent": agent,
        "method": method,
        "baseline_name": baseline_name,
        "baseline_config_hash": str(first.get("baseline_config_hash") or baseline_entry.get("hash") or ""),
        "attack_name": str(first.get("attack_name", "")),
        "mixed_eval_conditions": dict(Counter(str(row.get("mixed_eval_condition", "")) for row in rows if row.get("mixed_eval_condition"))),
        "metrics": metrics(rows),
        "manifest_status": manifest.get("status"),
        "completed_instances": manifest.get("completed_instances"),
        "total_instances": manifest.get("total_instances"),
        "config": compact,
    }


def pct(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{100 * float(value):.1f}"


def cell(summary: dict[str, Any], mode: str) -> str:
    m = summary["metrics"]
    if mode == "accuracy":
        return pct(m.get("accuracy"))
    if mode == "detailed":
        fpr = m.get("fpr")
        fpr_text = f", FPR {pct(fpr)}" if fpr is not None else ""
        suffix = "" if fpr is not None else " (attack-only)"
        return f"Acc {pct(m.get('accuracy'))}, R {pct(m.get('recall'))}{fpr_text}{suffix}"
    raise ValueError(f"unknown table cell mode: {mode}")


def write_long_csv(summaries: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "attack",
        "agent",
        "method",
        "dataset",
        "accuracy",
        "precision",
        "recall",
        "fpr",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
        "total_rows",
        "baseline_name",
        "baseline_config_hash",
        "threshold",
        "reject_threshold",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            m = summary["metrics"]
            c = summary["config"]
            writer.writerow(
                {
                    "attack": summary["attack"],
                    "agent": summary["agent"],
                    "method": summary["method"],
                    "dataset": summary["dataset"],
                    "accuracy": m.get("accuracy"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "fpr": m.get("fpr"),
                    "f1": m.get("f1"),
                    "tp": m.get("tp"),
                    "fp": m.get("fp"),
                    "tn": m.get("tn"),
                    "fn": m.get("fn"),
                    "total_rows": m.get("total_rows"),
                    "baseline_name": summary["baseline_name"],
                    "baseline_config_hash": summary["baseline_config_hash"],
                    "threshold": c.get("threshold"),
                    "reject_threshold": c.get("reject_threshold"),
                    "run_dir": summary["run_dir"],
                }
            )


def regenerated_run_date(summary: dict[str, Any]) -> int:
    """Return a dated regeneration tag from the output path, if present."""
    matches = re.findall(r"regenerated_(\d{8})", summary["run_dir"])
    return max((int(value) for value in matches), default=0)


def best_by_cell(summaries: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    best: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for summary in summaries:
        if summary["attack"] == "unknown" or summary["dataset"] == "unknown":
            continue
        key = (summary["attack"], summary["agent"], summary["method"], summary["dataset"])
        accuracy = summary["metrics"].get("accuracy")
        if accuracy is None:
            continue
        old = best.get(key)
        has_fpr = summary["metrics"].get("fpr") is not None
        old_has_fpr = old is not None and old["metrics"].get("fpr") is not None
        regenerated_date = regenerated_run_date(summary)
        old_regenerated_date = regenerated_run_date(old) if old else 0
        old_accuracy = old["metrics"].get("accuracy") if old else None
        if (
            old is None
            or (has_fpr and not old_has_fpr)
            or (
                has_fpr == old_has_fpr
                and regenerated_date > old_regenerated_date
            )
            or (
                has_fpr == old_has_fpr
                and regenerated_date == old_regenerated_date
                and (old_accuracy is None or accuracy > old_accuracy)
            )
        ):
            best[key] = summary
    return best


def write_selection_audit(
    summaries: list[dict[str, Any]],
    path: Path,
    selected: dict[tuple[str, str, str, str], dict[str, Any]],
) -> None:
    fields = [
        "selected",
        "attack",
        "agent",
        "method",
        "dataset",
        "accuracy",
        "precision",
        "recall",
        "fpr",
        "f1",
        "total_rows",
        "baseline_name",
        "baseline_config_hash",
        "threshold",
        "reject_threshold",
        "manifest_status",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            key = (summary["attack"], summary["agent"], summary["method"], summary["dataset"])
            m = summary["metrics"]
            c = summary["config"]
            writer.writerow(
                {
                    "selected": selected.get(key, {}).get("run_dir") == summary["run_dir"],
                    "attack": summary["attack"],
                    "agent": summary["agent"],
                    "method": summary["method"],
                    "dataset": summary["dataset"],
                    "accuracy": m.get("accuracy"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "fpr": m.get("fpr"),
                    "f1": m.get("f1"),
                    "total_rows": m.get("total_rows"),
                    "baseline_name": summary["baseline_name"],
                    "baseline_config_hash": summary["baseline_config_hash"],
                    "threshold": c.get("threshold"),
                    "reject_threshold": c.get("reject_threshold"),
                    "manifest_status": summary.get("manifest_status"),
                    "run_dir": summary["run_dir"],
                }
            )


def write_pivot(
    best: dict[tuple[str, str, str, str], dict[str, Any]],
    path: Path,
    mode: str,
    attack_order: list[str],
) -> None:
    fields = ["Attack", "Agent", "Method", *DATASET_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for attack in attack_order:
            for agent in AGENT_ORDER:
                for method in METHOD_ORDER:
                    row = {"Attack": attack, "Agent": agent, "Method": method}
                    for dataset in DATASET_COLUMNS:
                        summary = best.get((attack, agent, method, dataset))
                        row[dataset] = cell(summary, mode) if summary else ""
                    writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/baselines")
    parser.add_argument("--out-dir")
    parser.add_argument(
        "--table-mode",
        choices=["standard", "obfuscated"],
        default="standard",
        help="Export the standard attack table or the separate Base64-obfuscated attack table.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    attack_order = ATTACK_ORDER if args.table_mode == "standard" else OBFUSCATED_ATTACK_ORDER
    default_out_dir = (
        "outputs/baselines/_paper_table"
        if args.table_mode == "standard"
        else "outputs/baselines/_paper_table_obfuscated"
    )
    out_dir = Path(args.out_dir or default_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        summary
        for path in sorted(root.rglob("results.jsonl"))
        if (summary := summarize_result(path)) is not None
        and summary["attack"] in attack_order
    ]
    summaries.sort(key=lambda x: (x["attack"], x["agent"], x["method"], x["dataset"], x["run_dir"]))
    (out_dir / "all_baseline_runs.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_long_csv(summaries, out_dir / "all_baseline_runs.csv")
    best = best_by_cell(summaries)
    write_selection_audit(summaries, out_dir / "selection_audit.csv", best)
    write_pivot(best, out_dir / "paper_table_accuracy.csv", "accuracy", attack_order)
    write_pivot(best, out_dir / "paper_table_detailed.csv", "detailed", attack_order)
    print(f"wrote {len(summaries)} summarized runs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
