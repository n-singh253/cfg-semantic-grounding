#!/usr/bin/env python3
"""Summarize baseline result directories with metrics and config snapshots."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def first_integration_spec(run_dir: Path) -> dict[str, Any]:
    direct = load_json(run_dir / "integration_spec.json")
    if direct:
        return direct
    for path in sorted((run_dir / "_shards").glob("shard_*/integration_spec.json")):
        payload = load_json(path)
        if payload:
            return payload
    return {}


def row_label(row: dict[str, Any]) -> int | None:
    if "mixed_eval_label" in row:
        return int(row["mixed_eval_label"])
    if "graph_label" in row:
        return int(row["graph_label"])
    attack = str(row.get("attack_name", ""))
    if attack:
        return 0 if attack == "none" else 1
    return None


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
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

    total = len(rows)
    accepted = sum(1 for row in rows if str(row.get("defense_decision", "")).lower() == "accept")
    rejected = sum(1 for row in rows if str(row.get("defense_decision", "")).lower() == "reject")
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    accuracy = (tp + tn) / labeled if labeled else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    labels = Counter(row_label(row) for row in rows)
    labels.pop(None, None)
    decisions = Counter(str(row.get("defense_decision", "")) for row in rows)
    return {
        "total_rows": total,
        "labeled_rows": labeled,
        "label_counts": {str(k): v for k, v in sorted(labels.items())},
        "accepted": accepted,
        "rejected": rejected,
        "decision_counts": dict(decisions),
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


def compact_config(config: dict[str, Any]) -> dict[str, Any]:
    llm = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    return {
        "plugin": config.get("plugin"),
        "threshold": config.get("threshold"),
        "reject_threshold": config.get("reject_threshold"),
        "decision_policy": config.get("decision_policy"),
        "gnn_model_path": config.get("gnn_model_path") or config.get("model_path"),
        "embedding_model_name": config.get("embedding_model_name"),
        "embedding_pooling": config.get("embedding_pooling"),
        "link_similarity_threshold": config.get("link_similarity_threshold"),
        "link_topk_per_subtask": config.get("link_topk_per_subtask"),
        "prompt_parser": (config.get("parsers") or {}).get("prompt") if isinstance(config.get("parsers"), dict) else None,
        "patch_parser": (config.get("parsers") or {}).get("patch") if isinstance(config.get("parsers"), dict) else None,
        "linker": (config.get("parsers") or {}).get("linking") if isinstance(config.get("parsers"), dict) else None,
        "llm_provider": llm.get("provider") or config.get("provider"),
        "llm_model": llm.get("model") or config.get("model"),
        "llm_temperature": llm.get("temperature") if "temperature" in llm else config.get("temperature"),
    }


def summarize_run(results_path: Path, root: Path) -> dict[str, Any]:
    run_dir = results_path.parent
    rows = load_jsonl(results_path)
    manifest = load_json(run_dir / "shard_manifest.json")
    spec = first_integration_spec(run_dir)
    selected = spec.get("selected_configs") if isinstance(spec.get("selected_configs"), dict) else {}
    baseline_cfg_entry = selected.get("baseline") if isinstance(selected.get("baseline"), dict) else {}
    baseline_config = baseline_cfg_entry.get("config") if isinstance(baseline_cfg_entry.get("config"), dict) else {}
    first = rows[0] if rows else {}

    attacks = Counter(str(row.get("mixed_eval_source_attack_name") or row.get("attack_name", "")) for row in rows)
    attacks.pop("", None)
    mixed_conditions = Counter(str(row.get("mixed_eval_condition", "")) for row in rows)
    mixed_conditions.pop("", None)

    summary = {
        "run_dir": str(run_dir),
        "results_path": str(results_path),
        "relative_run_dir": str(run_dir.relative_to(root)) if run_dir.is_relative_to(root) else str(run_dir),
        "manifest": manifest,
        "integration_spec": spec,
        "dataset": first.get("dataset"),
        "split": first.get("split"),
        "agent_name": first.get("agent_name"),
        "attack_name": first.get("attack_name"),
        "mixed_eval_conditions": dict(mixed_conditions),
        "mixed_eval_source_attack_counts": dict(attacks),
        "baseline_name": first.get("baseline_name") or baseline_cfg_entry.get("name"),
        "baseline_config_hash": first.get("baseline_config_hash") or baseline_cfg_entry.get("hash"),
        "attack_config_hashes": sorted({str(row.get("attack_config_hash", "")) for row in rows if row.get("attack_config_hash")}),
        "agent_config_hashes": sorted({str(row.get("agent_config_hash", "")) for row in rows if row.get("agent_config_hash")}),
        "baseline_config": baseline_config,
        "baseline_config_compact": compact_config(baseline_config),
        "metrics": metric_summary(rows),
    }
    return summary


def write_outputs(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_run_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    csv_path = out_dir / "baseline_run_summary.csv"
    fields = [
        "run_dir",
        "baseline_name",
        "baseline_config_hash",
        "dataset",
        "split",
        "agent_name",
        "attack_name",
        "mixed_eval_conditions",
        "total_rows",
        "label_counts",
        "accepted",
        "rejected",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "fpr",
        "f1",
        "status",
        "completed_instances",
        "total_instances",
        "threshold",
        "reject_threshold",
        "plugin",
        "llm_provider",
        "llm_model",
        "prompt_parser",
        "patch_parser",
        "linker",
        "gnn_model_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            metrics = summary["metrics"]
            manifest = summary.get("manifest") or {}
            compact = summary.get("baseline_config_compact") or {}
            writer.writerow(
                {
                    "run_dir": summary["run_dir"],
                    "baseline_name": summary.get("baseline_name"),
                    "baseline_config_hash": summary.get("baseline_config_hash"),
                    "dataset": summary.get("dataset"),
                    "split": summary.get("split"),
                    "agent_name": summary.get("agent_name"),
                    "attack_name": summary.get("attack_name"),
                    "mixed_eval_conditions": json.dumps(summary.get("mixed_eval_conditions", {}), sort_keys=True),
                    "total_rows": metrics.get("total_rows"),
                    "label_counts": json.dumps(metrics.get("label_counts", {}), sort_keys=True),
                    "accepted": metrics.get("accepted"),
                    "rejected": metrics.get("rejected"),
                    "tp": metrics.get("tp"),
                    "fp": metrics.get("fp"),
                    "tn": metrics.get("tn"),
                    "fn": metrics.get("fn"),
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "fpr": metrics.get("fpr"),
                    "f1": metrics.get("f1"),
                    "status": manifest.get("status"),
                    "completed_instances": manifest.get("completed_instances"),
                    "total_instances": manifest.get("total_instances"),
                    "threshold": compact.get("threshold"),
                    "reject_threshold": compact.get("reject_threshold"),
                    "plugin": compact.get("plugin"),
                    "llm_provider": compact.get("llm_provider"),
                    "llm_model": compact.get("llm_model"),
                    "prompt_parser": compact.get("prompt_parser"),
                    "patch_parser": compact.get("patch_parser"),
                    "linker": compact.get("linker"),
                    "gnn_model_path": compact.get("gnn_model_path"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Baseline output root to scan")
    parser.add_argument("--out-dir", help="Directory for summary files. Defaults to <root>/_summary")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "_summary"
    summaries = [
        summarize_run(path, root)
        for path in sorted(root.glob("*/*/results.jsonl"))
    ]
    write_outputs(summaries, out_dir)
    print(f"wrote {len(summaries)} run summaries to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
