#!/usr/bin/env python3
"""Filter structural-misalignment eval results with bad prompt parsing diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.baseline.structural_misalignment.grounding.schemas import (
    analyze_subtask_requirement_retention,
    parse_subtasks_with_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]
PROBLEM_ROOT = ROOT / "outputs/diagnostics/structural_misalignment/problematic_eval_ids"
OUT_ROOT = ROOT / "outputs/diagnostics/structural_misalignment/filtered_clean_eval_results"


DATASETS = {
    "gemini_fcv": {
        "results": ROOT
        / "outputs/baselines/livecodebench/gemini3_flash/structural_misalignment_livecodebench_gemini/fcv_cwe94/results.jsonl",
        "known_bad": PROBLEM_ROOT / "gemini_fcv.txt",
    },
    "gemini_swexploit": {
        "results": ROOT
        / "outputs/baselines/livecodebench/gemini3_flash/structural_misalignment_livecodebench_gemini/swexploit_gemini_vertex/results.jsonl",
        "known_bad": PROBLEM_ROOT / "gemini_swexploit.txt",
    },
    "claude_fcv": {
        "results": ROOT
        / "outputs/baselines/livecodebench/claude37_sonnet_sweagent/structural_misalignment_livecodebench_claude/fcv_cwe94/results.jsonl",
        "known_bad": PROBLEM_ROOT / "claude_fcv.txt",
    },
    "claude_swexploit": {
        "results": ROOT
        / "outputs/baselines/livecodebench/claude37_sonnet_sweagent/structural_misalignment_livecodebench_claude/swexploit_gemini_vertex/results.jsonl",
        "known_bad": PROBLEM_ROOT / "claude_swexploit.txt",
    },
}


BAD_PARSE_MODES = {
    "empty",
    "line_fallback_non_json",
    "strict_json_nonarray",
    "embedded_json_nonarray",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def read_id_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def write_id_file(path: Path, ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{instance_id}\n" for instance_id in sorted(ids)))


def path_from_row(row: dict[str, Any], *keys: str) -> Path | None:
    current: Any = row
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if not current:
        return None
    path = Path(str(current))
    return path if path.is_absolute() else ROOT / path


def read_graph(row: dict[str, Any]) -> dict[str, Any]:
    graph_path = path_from_row(row, "defense_signals", "artifact_paths", "graph_json")
    if not graph_path or not graph_path.exists():
        return {}
    graph = json.loads(graph_path.read_text(errors="replace"))
    return graph if isinstance(graph, dict) else {}


def read_response_text(row: dict[str, Any]) -> str:
    graph_path = path_from_row(row, "defense_signals", "artifact_paths", "graph_json")
    if not graph_path:
        return ""
    subtask_dir = graph_path.parent.parent / "subtasks"
    metadata_path = subtask_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(errors="replace"))
            text = metadata.get("text")
            if isinstance(text, str):
                return text
        except json.JSONDecodeError:
            pass
    response_path = subtask_dir / "response.txt"
    if response_path.exists():
        return response_path.read_text(errors="replace")
    return ""


def diagnose_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt_path = path_from_row(row, "adv_agent_metadata", "prompt_path")
    prompt = prompt_path.read_text(errors="replace") if prompt_path and prompt_path.exists() else ""
    graph = read_graph(row)
    subtasks = graph.get("subtasks", []) if isinstance(graph, dict) else []
    code_nodes = graph.get("code_nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", {}) if isinstance(graph, dict) else {}
    subtask_to_code = edges.get("subtask_to_code", []) if isinstance(edges, dict) else []
    response_text = read_response_text(row)
    _parsed_subtasks, parse_diagnostics = parse_subtasks_with_diagnostics(response_text)
    retention = analyze_subtask_requirement_retention(prompt, subtasks)

    reasons: list[str] = []
    parse_mode = parse_diagnostics.get("parse_mode", "empty")
    if parse_mode in BAD_PARSE_MODES:
        reasons.append(f"parse:{parse_mode}")
    if retention.get("prompt_has_suspicious_terms") and not retention.get("subtasks_retain_any_suspicious_terms"):
        reasons.append("missing_injection_terms")
    if not subtasks:
        reasons.append("empty_subtasks")
    if not code_nodes:
        reasons.append("empty_code_nodes")
    if not subtask_to_code:
        reasons.append("empty_subtask_to_code_edges")

    return {
        "instance_id": row.get("instance_id", ""),
        "parse_mode": parse_mode,
        "subtask_count": len(subtasks),
        "code_node_count": len(code_nodes),
        "subtask_to_code_edges": len(subtask_to_code),
        "strict_json_valid": parse_diagnostics.get("strict_json_valid", False),
        "embedded_json_valid": parse_diagnostics.get("embedded_json_valid", False),
        "prompt_suspicious_terms": retention.get("prompt_suspicious_terms", []),
        "subtask_suspicious_terms": retention.get("subtask_suspicious_terms", []),
        "missing_suspicious_terms": retention.get("missing_suspicious_terms", []),
        "suspicious_term_retention_fraction": retention.get("suspicious_term_retention_fraction"),
        "reasons": reasons,
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for name, cfg in DATASETS.items():
        rows = read_jsonl(cfg["results"])
        known_bad = read_id_file(cfg["known_bad"])
        clean_rows: list[dict[str, Any]] = []
        removed_rows: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        removed_ids: set[str] = set()
        kept_ids: set[str] = set()
        fresh_problem_ids: set[str] = set()
        kept_with_fresh_problems: list[str] = []

        for row in rows:
            instance_id = str(row.get("instance_id", ""))
            diag = diagnose_row(row)
            diagnostics.append(diag)
            if diag["reasons"]:
                fresh_problem_ids.add(instance_id)
            should_remove = instance_id in known_bad or bool(diag["reasons"])
            if should_remove:
                removed_rows.append(row)
                removed_ids.add(instance_id)
            else:
                clean_rows.append(row)
                kept_ids.add(instance_id)

        for diag in diagnostics:
            instance_id = str(diag.get("instance_id", ""))
            if instance_id in kept_ids and diag.get("reasons"):
                kept_with_fresh_problems.append(instance_id)

        dataset_out = OUT_ROOT / name
        write_jsonl(dataset_out / "results.clean.jsonl", clean_rows)
        write_jsonl(dataset_out / "results.removed_problematic.jsonl", removed_rows)
        write_id_file(dataset_out / "kept_ids.txt", kept_ids)
        write_id_file(dataset_out / "removed_problematic_ids.txt", removed_ids)
        (dataset_out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

        by_reason: dict[str, int] = {}
        for diag in diagnostics:
            for reason in diag.get("reasons", []):
                by_reason[reason] = by_reason.get(reason, 0) + 1

        summary[name] = {
            "input_results": str(cfg["results"].relative_to(ROOT)),
            "known_bad_ids": len(known_bad),
            "fresh_problem_ids": len(fresh_problem_ids),
            "original_rows": len(rows),
            "clean_rows": len(clean_rows),
            "removed_rows": len(removed_rows),
            "kept_with_fresh_problems": kept_with_fresh_problems,
            "by_reason": dict(sorted(by_reason.items())),
            "outputs": {
                "clean_results": str((dataset_out / "results.clean.jsonl").relative_to(ROOT)),
                "removed_results": str((dataset_out / "results.removed_problematic.jsonl").relative_to(ROOT)),
                "kept_ids": str((dataset_out / "kept_ids.txt").relative_to(ROOT)),
                "removed_ids": str((dataset_out / "removed_problematic_ids.txt").relative_to(ROOT)),
                "diagnostics": str((dataset_out / "diagnostics.json").relative_to(ROOT)),
            },
        }

    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
