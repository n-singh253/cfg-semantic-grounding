#!/usr/bin/env python3
"""Export paper-oriented cost summaries for selected baseline runs.

The exporter intentionally keys off ``selection_audit.csv`` from
``scripts/export_paper_table.py`` so the cost table uses the same runs as the
accuracy table.  It reports directly observed quantities from result rows and
LLM cache files.  Dollar estimates are optional because model pricing changes;
pass ``--rates-json`` to pin the rates used in a paper draft.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


TOKEN_INPUT_KEYS = ("input_tokens", "prompt_tokens", "prompt_token_count")
TOKEN_OUTPUT_KEYS = ("output_tokens", "completion_tokens", "candidates_token_count")
TOKEN_TOTAL_KEYS = ("total_tokens", "total_token_count")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def first_number(payload: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def normalized_usage(usage: Any) -> dict[str, float]:
    if not isinstance(usage, dict):
        return {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
    input_tokens = first_number(usage, TOKEN_INPUT_KEYS)
    output_tokens = first_number(usage, TOKEN_OUTPUT_KEYS)
    total_tokens = first_number(usage, TOKEN_TOTAL_KEYS)
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def add_usage(dst: dict[str, float], usage: dict[str, float], prefix: str = "") -> None:
    for key, value in usage.items():
        dst[f"{prefix}{key}"] += float(value)


def find_result_tokens(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate token usage carried directly in results.jsonl rows."""
    totals: dict[str, float] = defaultdict(float)
    calls = 0
    billable_calls = 0
    models: set[str] = set()
    providers: set[str] = set()
    cache_hits = 0
    cache_misses = 0

    for row in rows:
        signals = row.get("defense_signals")
        if not isinstance(signals, dict):
            continue
        usage = normalized_usage(signals.get("token_usage"))
        if not any(usage.values()):
            continue
        calls += 1
        add_usage(totals, usage)
        cache_hit = bool(signals.get("cache_hit", False))
        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1
            billable_calls += 1
            add_usage(totals, usage, "billable_")
        if signals.get("provider"):
            providers.add(str(signals["provider"]))
        if signals.get("model"):
            models.add(str(signals["model"]))

    return {
        "source": "results_rows" if calls else "",
        "llm_calls": calls,
        "billable_llm_calls": billable_calls,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "providers": sorted(providers),
        "models": sorted(models),
        **totals,
    }


def find_cache_tokens(run_dir: Path) -> dict[str, Any]:
    """Aggregate token usage from artifacts/llm_cache JSON files in a run."""
    totals: dict[str, float] = defaultdict(float)
    calls = 0
    billable_calls = 0
    models: set[str] = set()
    providers: set[str] = set()
    cache_hits = 0
    cache_misses = 0
    seen: set[tuple[str, str, str]] = set()

    for path in sorted(run_dir.glob("**/artifacts/llm_cache/**/*.json")):
        payload = load_json(path)
        usage = normalized_usage(payload.get("token_usage"))
        if not any(usage.values()):
            continue
        dedupe_key = (
            str(payload.get("provider", "")),
            str(payload.get("model", "")),
            str(payload.get("cache_key") or path),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        calls += int(payload.get("call_count") or 1)
        add_usage(totals, usage)
        cache_hit = bool(payload.get("cache_hit", False))
        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1
            billable_calls += int(payload.get("call_count") or 1)
            add_usage(totals, usage, "billable_")
        if payload.get("provider"):
            providers.add(str(payload["provider"]))
        if payload.get("model"):
            models.add(str(payload["model"]))

    return {
        "source": "llm_cache" if calls else "",
        "llm_calls": calls,
        "billable_llm_calls": billable_calls,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "providers": sorted(providers),
        "models": sorted(models),
        **totals,
    }


def _graph_root_from_graph_json(path_value: Any) -> Path | None:
    if not path_value:
        return None
    graph_json = Path(str(path_value))
    if graph_json.name != "graph.json":
        return None
    if graph_json.parent.name != "graph":
        return None
    return graph_json.parent.parent


def find_prebuilt_graph_tokens(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Attribute token usage from prebuilt structural graph artifacts.

    Some selected SWEGuard/Ours evaluation runs reuse graphs built during model
    training.  The evaluation rows then correctly have ``graph_cache_hit=True``
    and no per-row LLM cache, but a paper cost table should attribute the LLM
    calls that built those referenced graphs.  This function follows each
    result row's ``defense_signals.artifact_paths.graph_json`` to the graph root
    and aggregates token usage from the graph's subtask chunk metadata.
    """
    totals: dict[str, float] = defaultdict(float)
    calls = 0
    billable_calls = 0
    models: set[str] = set()
    providers: set[str] = set()
    cache_hits = 0
    cache_misses = 0
    graph_roots: set[Path] = set()
    seen: set[tuple[str, str, str]] = set()

    for row in rows:
        signals = row.get("defense_signals")
        if not isinstance(signals, dict):
            continue
        artifacts = signals.get("artifact_paths")
        if not isinstance(artifacts, dict):
            continue
        graph_root = _graph_root_from_graph_json(artifacts.get("graph_json"))
        if graph_root is not None and graph_root.exists():
            graph_roots.add(graph_root)

    for graph_root in sorted(graph_roots):
        for path in sorted((graph_root / "subtasks").glob("**/metadata.json")):
            payload = load_json(path)
            usage = normalized_usage(payload.get("token_usage"))
            if not any(usage.values()):
                continue
            dedupe_key = (
                str(payload.get("provider", "")),
                str(payload.get("model", "")),
                str(payload.get("cache_key") or path),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            call_count = int(payload.get("call_count") or 1)
            calls += call_count
            add_usage(totals, usage)
            cache_hit = bool(payload.get("cache_hit", False))
            if cache_hit:
                cache_hits += call_count
            else:
                cache_misses += call_count
                billable_calls += call_count
                add_usage(totals, usage, "billable_")
            if payload.get("provider"):
                providers.add(str(payload["provider"]))
            if payload.get("model"):
                models.add(str(payload["model"]))

    return {
        "source": "prebuilt_graph_artifacts" if calls else "",
        "llm_calls": calls,
        "billable_llm_calls": billable_calls,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "providers": sorted(providers),
        "models": sorted(models),
        "attributed_graph_count": len(graph_roots),
        **totals,
    }


def runtime_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    defense = [
        float(row["defense_runtime_sec"])
        for row in rows
        if isinstance(row.get("defense_runtime_sec"), (int, float))
    ]
    total = [
        float(row["runtime_sec"])
        for row in rows
        if isinstance(row.get("runtime_sec"), (int, float))
    ]
    return {
        "defense_runtime_sec_total": sum(defense),
        "defense_runtime_sec_mean": mean(defense) if defense else 0.0,
        "row_runtime_sec_total": sum(total),
        "row_runtime_sec_mean": mean(total) if total else 0.0,
    }


def rates_key(provider: str, model: str) -> str:
    return f"{provider}/{model}".strip("/")


def load_rates(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    payload = load_json(path)
    rates: dict[str, dict[str, float]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        rates[str(key)] = {
            "input_per_million": float(value.get("input_per_million", 0.0) or 0.0),
            "output_per_million": float(value.get("output_per_million", 0.0) or 0.0),
        }
    return rates


def estimate_usd(
    providers: list[str],
    models: list[str],
    input_tokens: float,
    output_tokens: float,
    rates: dict[str, dict[str, float]],
) -> tuple[float | None, str]:
    if len(providers) != 1 or len(models) != 1:
        return None, ""
    key = rates_key(providers[0], models[0])
    rate = rates.get(key)
    if not rate:
        return None, key
    usd = (
        input_tokens * rate.get("input_per_million", 0.0)
        + output_tokens * rate.get("output_per_million", 0.0)
    ) / 1_000_000.0
    return usd, key


def selected_runs(selection_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for selection_path in selection_paths:
        if not selection_path.exists():
            continue
        with selection_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("selected", "")).lower() == "true":
                    row = dict(row)
                    row["selection_file"] = str(selection_path)
                    rows.append(row)
    return rows


def summarize_run(selection: dict[str, str], rates: dict[str, dict[str, float]]) -> dict[str, Any]:
    run_dir = Path(selection["run_dir"])
    result_path = run_dir / "results.jsonl"
    rows = load_jsonl(result_path)

    token_summary = find_result_tokens(rows)
    if not token_summary.get("llm_calls"):
        token_summary = find_cache_tokens(run_dir)
    if not token_summary.get("llm_calls"):
        token_summary = find_prebuilt_graph_tokens(rows)

    providers = token_summary.get("providers") or []
    models = token_summary.get("models") or []
    logical_usd, logical_rate_key = estimate_usd(
        providers,
        models,
        float(token_summary.get("input_tokens", 0.0) or 0.0),
        float(token_summary.get("output_tokens", 0.0) or 0.0),
        rates,
    )
    billable_usd, billable_rate_key = estimate_usd(
        providers,
        models,
        float(token_summary.get("billable_input_tokens", 0.0) or 0.0),
        float(token_summary.get("billable_output_tokens", 0.0) or 0.0),
        rates,
    )
    runtime = runtime_summary(rows)

    return {
        "attack": selection.get("attack", ""),
        "agent": selection.get("agent", ""),
        "method": selection.get("method", ""),
        "dataset": selection.get("dataset", ""),
        "rows": len(rows),
        "selection_total_rows": selection.get("total_rows", ""),
        "llm_usage_source": token_summary.get("source", ""),
        "providers": ";".join(providers),
        "models": ";".join(models),
        "llm_calls": int(token_summary.get("llm_calls", 0) or 0),
        "cache_hits": int(token_summary.get("cache_hits", 0) or 0),
        "cache_misses": int(token_summary.get("cache_misses", 0) or 0),
        "billable_llm_calls": int(token_summary.get("billable_llm_calls", 0) or 0),
        "input_tokens": int(token_summary.get("input_tokens", 0) or 0),
        "output_tokens": int(token_summary.get("output_tokens", 0) or 0),
        "total_tokens_reported": int(token_summary.get("total_tokens", 0) or 0),
        "billable_input_tokens": int(token_summary.get("billable_input_tokens", 0) or 0),
        "billable_output_tokens": int(token_summary.get("billable_output_tokens", 0) or 0),
        "billable_total_tokens_reported": int(token_summary.get("billable_total_tokens", 0) or 0),
        "estimated_logical_usd": "" if logical_usd is None else f"{logical_usd:.6f}",
        "estimated_billable_usd": "" if billable_usd is None else f"{billable_usd:.6f}",
        "pricing_key": logical_rate_key or billable_rate_key,
        **{k: f"{v:.6f}" for k, v in runtime.items()},
        "run_dir": str(run_dir),
        "results_path": str(result_path),
        "selection_file": selection.get("selection_file", ""),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "attack",
        "agent",
        "method",
        "dataset",
        "rows",
        "llm_usage_source",
        "providers",
        "models",
        "llm_calls",
        "cache_hits",
        "cache_misses",
        "billable_llm_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens_reported",
        "billable_input_tokens",
        "billable_output_tokens",
        "billable_total_tokens_reported",
        "estimated_logical_usd",
        "estimated_billable_usd",
        "pricing_key",
        "defense_runtime_sec_total",
        "defense_runtime_sec_mean",
        "row_runtime_sec_total",
        "row_runtime_sec_mean",
        "run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        ("Attack", "attack"),
        ("Agent", "agent"),
        ("Method", "method"),
        ("Dataset", "dataset"),
        ("Rows", "rows"),
        ("LLM calls", "llm_calls"),
        ("Input tok.", "input_tokens"),
        ("Output tok.", "output_tokens"),
        ("Billable calls", "billable_llm_calls"),
        ("Billable USD", "estimated_billable_usd"),
        ("Mean defense s", "defense_runtime_sec_mean"),
    ]
    lines = []
    lines.append("| " + " | ".join(label for label, _ in fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for _, key in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)

    out: list[dict[str, Any]] = []
    for group_key, group_rows in grouped.items():
        total_rows = sum(int(row.get("rows", 0) or 0) for row in group_rows)
        input_tokens = sum(int(row.get("input_tokens", 0) or 0) for row in group_rows)
        output_tokens = sum(int(row.get("output_tokens", 0) or 0) for row in group_rows)
        total_tokens = sum(int(row.get("total_tokens_reported", 0) or 0) for row in group_rows)
        billable_input = sum(int(row.get("billable_input_tokens", 0) or 0) for row in group_rows)
        billable_output = sum(int(row.get("billable_output_tokens", 0) or 0) for row in group_rows)
        billable_total = sum(int(row.get("billable_total_tokens_reported", 0) or 0) for row in group_rows)
        llm_calls = sum(int(row.get("llm_calls", 0) or 0) for row in group_rows)
        billable_calls = sum(int(row.get("billable_llm_calls", 0) or 0) for row in group_rows)
        defense_runtime = sum(float(row.get("defense_runtime_sec_total", 0.0) or 0.0) for row in group_rows)
        estimated_values: list[float] = []
        all_have_usd = True
        for row in group_rows:
            estimated = str(row.get("estimated_billable_usd", "")).strip()
            if estimated:
                estimated_values.append(float(estimated))
                continue
            has_llm_cost_signal = bool(str(row.get("pricing_key", "")).strip()) or bool(
                int(row.get("billable_input_tokens", 0) or 0)
                or int(row.get("billable_output_tokens", 0) or 0)
            )
            if has_llm_cost_signal:
                all_have_usd = False
            else:
                estimated_values.append(0.0)
        item = {
            "runs": len(group_rows),
            "rows": total_rows,
            "llm_calls": llm_calls,
            "billable_llm_calls": billable_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens_reported": total_tokens,
            "billable_input_tokens": billable_input,
            "billable_output_tokens": billable_output,
            "billable_total_tokens_reported": billable_total,
            "input_tokens_per_row": input_tokens / total_rows if total_rows else 0.0,
            "output_tokens_per_row": output_tokens / total_rows if total_rows else 0.0,
            "llm_calls_per_row": llm_calls / total_rows if total_rows else 0.0,
            "billable_llm_calls_per_row": billable_calls / total_rows if total_rows else 0.0,
            "defense_runtime_sec_total": defense_runtime,
            "defense_runtime_sec_per_row": defense_runtime / total_rows if total_rows else 0.0,
            "estimated_billable_usd": f"{sum(estimated_values):.6f}" if all_have_usd else "",
            "estimated_billable_usd_per_row": (
                f"{sum(estimated_values) / total_rows:.6f}" if all_have_usd and total_rows else ""
            ),
        }
        for key_name, value in zip(keys, group_key):
            item[key_name] = value
        out.append(item)
    out.sort(key=lambda row: tuple(str(row.get(key, "")) for key in keys))
    return out


def write_aggregate_csv(rows: list[dict[str, Any]], path: Path, keys: list[str]) -> None:
    fields = [
        *keys,
        "runs",
        "rows",
        "llm_calls",
        "llm_calls_per_row",
        "input_tokens",
        "input_tokens_per_row",
        "output_tokens",
        "output_tokens_per_row",
        "billable_llm_calls",
        "billable_llm_calls_per_row",
        "billable_input_tokens",
        "billable_output_tokens",
        "estimated_billable_usd",
        "estimated_billable_usd_per_row",
        "defense_runtime_sec_total",
        "defense_runtime_sec_per_row",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        f"{row[field]:.6f}"
                        if isinstance(row.get(field), float)
                        else row.get(field, "")
                    )
                    for field in fields
                }
            )


def write_aggregate_markdown(rows: list[dict[str, Any]], path: Path, keys: list[str]) -> None:
    fields = [
        *[(key.replace("_", " ").title(), key) for key in keys],
        ("Rows", "rows"),
        ("LLM calls / row", "llm_calls_per_row"),
        ("Input tok. / row", "input_tokens_per_row"),
        ("Output tok. / row", "output_tokens_per_row"),
        ("Billable USD / row", "estimated_billable_usd_per_row"),
        ("Defense s / row", "defense_runtime_sec_per_row"),
    ]
    lines = []
    lines.append("| " + " | ".join(label for label, _ in fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        rendered = []
        for _, key in fields:
            value = row.get(key, "")
            if isinstance(value, float):
                if "tokens" in key or "calls" in key:
                    rendered.append(f"{value:.2f}")
                else:
                    rendered.append(f"{value:.3f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-audit",
        action="append",
        default=[],
        help="Path to selection_audit.csv. Can be provided multiple times.",
    )
    parser.add_argument("--rates-json", help="Optional model pricing JSON with rates per million tokens.")
    parser.add_argument("--out-dir", default="outputs/baselines/_cost_table")
    args = parser.parse_args()

    selection_paths = [Path(p) for p in args.selection_audit] or [
        Path("outputs/baselines/_paper_table/selection_audit.csv"),
        Path("outputs/baselines/_paper_table_obfuscated/selection_audit.csv"),
    ]
    rates = load_rates(Path(args.rates_json) if args.rates_json else None)
    out_dir = Path(args.out_dir)
    rows = [summarize_run(row, rates) for row in selected_runs(selection_paths)]
    rows.sort(key=lambda r: (r["attack"], r["agent"], r["method"], r["dataset"], r["run_dir"]))
    write_csv(rows, out_dir / "paper_cost_table.csv")
    write_markdown(rows, out_dir / "paper_cost_table.md")
    by_method = aggregate_rows(rows, ["method"])
    by_method_dataset = aggregate_rows(rows, ["method", "dataset"])
    write_aggregate_csv(by_method, out_dir / "paper_cost_by_method.csv", ["method"])
    write_aggregate_markdown(by_method, out_dir / "paper_cost_by_method.md", ["method"])
    write_aggregate_csv(by_method_dataset, out_dir / "paper_cost_by_method_dataset.csv", ["method", "dataset"])
    write_aggregate_markdown(
        by_method_dataset,
        out_dir / "paper_cost_by_method_dataset.md",
        ["method", "dataset"],
    )
    (out_dir / "paper_cost_table.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} selected-run cost summaries to {out_dir}")
    if not rates:
        print("note: no --rates-json supplied, so USD estimate columns are blank unless rates are later provided")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
