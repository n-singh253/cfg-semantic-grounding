"""Train and evaluate structural misalignment models from graph-build outputs."""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.baseline.structural_misalignment.graph.build import build_pyg_heterodata
from src.baseline.structural_misalignment.models.tabular import (
    feature_columns,
    train_feature_models,
)
from src.common.artifact_store import atomic_write_json
from src.common.config import config_hash, load_yaml
from src.common.hashing import sha256_text
from src.common.rows import label_counts, load_rows


GRAPH_BUILD_REQUIRED_STAGES = (
    "cfg_ready",
    "subtasks_ready",
    "grounding_ready",
    "embeddings_ready",
    "graph_ready",
    "features_ready",
    "training_example_ready",
)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip().lstrip("\x00")
            if line:
                rows.append(json.loads(line))
    return rows


def _artifact_payload(path: Path) -> Dict[str, Any]:
    raw = _read_json(path)
    if isinstance(raw, dict) and isinstance(raw.get("payload"), dict):
        return raw["payload"]
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"artifact is not a JSON object: {path}")


def _path_or_empty(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text).expanduser()


def _resolve_artifact_path(path: Path | None, graph_dir: Path) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return graph_dir / path


def _row_identity(row: Dict[str, Any], row_index: int) -> Dict[str, Any]:
    return {
        "row_index": row_index,
        "code_base": row["code_base"],
        "label": int(row["label"]),
    }


def _load_graph_examples(
    *,
    rows_path: Path | None,
    graph_dir: Path,
    source_name: str,
    min_candidate_nodes: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    input_rows = load_rows(rows_path) if rows_path is not None else []
    input_by_index = {
        idx: _row_identity(row, idx)
        for idx, row in enumerate(input_rows)
    }

    result_path = graph_dir / "results.jsonl"
    if not result_path.exists():
        raise FileNotFoundError(f"Graph-build results not found: {result_path}")

    examples: List[Dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    result_rows = _read_jsonl(result_path)
    for result in result_rows:
        if str(result.get("status", "")) != "success":
            dropped["result_not_success"] += 1
            continue

        row_index = int(result.get("row_index", -1))
        row_info = (
            input_by_index.get(row_index)
            if rows_path is not None
            else {"code_base": result.get("code_base", ""), "label": result.get("label", -1)}
        )
        if row_info is None:
            dropped["row_index_not_in_input"] += 1
            continue

        signals = result.get("defense_signals", {})
        if not isinstance(signals, dict):
            dropped["missing_signals"] += 1
            continue
        artifact_paths = signals.get("artifact_paths", {})
        if not isinstance(artifact_paths, dict):
            dropped["missing_artifact_paths"] += 1
            continue

        training_path = _resolve_artifact_path(
            _path_or_empty(artifact_paths.get("training_example")),
            graph_dir,
        )
        if training_path is None or not training_path.exists():
            dropped["missing_training_example"] += 1
            continue

        try:
            training = _artifact_payload(training_path)
        except Exception:
            dropped["unreadable_training_example"] += 1
            continue

        stage_completed = training.get("stage_completed", {})
        if not isinstance(stage_completed, dict) or not all(
            bool(stage_completed.get(stage)) for stage in GRAPH_BUILD_REQUIRED_STAGES
        ):
            dropped["incomplete_graph_build"] += 1
            continue

        counts = training.get("counts", {})
        if not isinstance(counts, dict):
            counts = {}
        if int(counts.get("num_candidate_nodes", 0) or 0) < min_candidate_nodes:
            dropped["too_few_candidate_nodes"] += 1
            continue

        features = training.get("features", {})
        if not isinstance(features, dict) or not features:
            dropped["missing_features"] += 1
            continue

        graph_info = training.get("graph", {})
        graph_artifacts = graph_info.get("artifacts", {}) if isinstance(graph_info, dict) else {}
        if not isinstance(graph_artifacts, dict):
            graph_artifacts = {}
        graph_json = _resolve_artifact_path(
            _path_or_empty(graph_artifacts.get("graph_json") or artifact_paths.get("graph_json")),
            graph_dir,
        )
        graph_pt = _resolve_artifact_path(
            _path_or_empty(graph_artifacts.get("graph_pt") or artifact_paths.get("graph_pt")),
            graph_dir,
        )
        if graph_json is None or not graph_json.exists():
            dropped["missing_graph_json"] += 1
            continue

        label = int(training.get("label", result.get("label", row_info["label"])))
        if label != row_info["label"]:
            dropped["label_mismatch"] += 1
            continue

        examples.append(
            {
                "source": source_name,
                "row_index": row_index,
                "artifact_instance_id": result.get("artifact_instance_id", ""),
                "code_base": row_info["code_base"],
                "label": label,
                "features": {
                    str(key): float(value or 0.0)
                    for key, value in features.items()
                    if isinstance(value, (int, float))
                },
                "counts": counts,
                "training_example_path": str(training_path),
                "graph_json_path": str(graph_json),
                "graph_pt_path": str(graph_pt) if graph_pt is not None and graph_pt.exists() else "",
            }
        )

    report = {
        "source": source_name,
        "rows_path": str(rows_path or ""),
        "graph_dir": str(graph_dir),
        "input_rows": len(input_rows) if rows_path is not None else len(result_rows),
        "raw_result_rows": len(result_rows),
        "loaded_examples": len(examples),
        "loaded_code_bases": len({example["code_base"] for example in examples}),
        "label_counts": dict(Counter(example["label"] for example in examples)),
        "dropped_reasons": dict(dropped),
    }
    return examples, report


def _split_code_bases(
    rows: Sequence[Dict[str, Any]],
    *,
    train_ratio: float,
    seed: int,
) -> Tuple[set[str], set[str], Dict[str, Any]]:
    return _split_code_base_ids(
        sorted({str(row["code_base"]) for row in rows}),
        train_ratio=train_ratio,
        seed=seed,
    )


def _split_code_base_ids(
    code_bases: Sequence[str],
    *,
    train_ratio: float,
    seed: int,
) -> Tuple[set[str], set[str], Dict[str, Any]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    code_bases = sorted({str(code_base) for code_base in code_bases})
    if len(code_bases) <= 1:
        raise ValueError("Structural evaluation requires at least two code_base groups.")

    rng = random.Random(seed)
    rng.shuffle(code_bases)
    split_at = int(round(len(code_bases) * train_ratio))
    split_at = min(max(1, split_at), len(code_bases) - 1)
    train_ids = set(code_bases[:split_at])
    test_ids = set(code_bases[split_at:])
    return train_ids, test_ids, {
        "split_key": "code_base",
        "train_ratio": train_ratio,
        "seed": seed,
        "code_base_count": len(code_bases),
        "train_code_base_count": len(train_ids),
        "test_code_base_count": len(test_ids),
        "train_code_bases": sorted(train_ids),
        "test_code_bases": sorted(test_ids),
    }


def _validate_examples(train: Sequence[Dict[str, Any]], test: Sequence[Dict[str, Any]]) -> None:
    if not train:
        raise ValueError("No synthetic training examples survived loading/filtering.")
    if not test:
        raise ValueError("No real test examples survived loading/filtering.")
    train_labels = {int(example["label"]) for example in train}
    test_labels = {int(example["label"]) for example in test}
    if train_labels != {0, 1}:
        raise ValueError(f"Training examples must contain both labels, got {sorted(train_labels)}")
    if test_labels != {0, 1}:
        raise ValueError(f"Test examples must contain both labels, got {sorted(test_labels)}")
    overlap = {example["code_base"] for example in train}.intersection(
        {example["code_base"] for example in test}
    )
    if overlap:
        raise ValueError(f"code_base leakage detected: {sorted(overlap)[:10]}")


def _append_results(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _config_path_value(config: Dict[str, Any], *names: str) -> Any:
    inputs = config.get("inputs")
    for name in names:
        if config.get(name) is not None and config.get(name) != "":
            return config[name]
        if isinstance(inputs, dict) and inputs.get(name) is not None and inputs.get(name) != "":
            return inputs[name]
    return None


def _resolve_required_path(
    explicit: Path | None,
    config: Dict[str, Any],
    *names: str,
    base_dir: Path,
) -> Path:
    value = explicit if explicit is not None else _config_path_value(config, *names)
    if value is None or value == "":
        joined = ", ".join(names)
        raise ValueError(f"Missing required structural eval path in config: {joined}")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _result_key(
    *,
    baseline_config_hash: str,
    method: str,
    model_name: str,
    data_rows_path: Path,
    data_graph_dir: Path,
    synthetic_graph_dir: Path,
    split: Dict[str, Any],
) -> str:
    return sha256_text(
        json.dumps(
            {
                "baseline_config_hash": baseline_config_hash,
                "method": method,
                "model_name": model_name,
                "data_rows_path": str(data_rows_path),
                "synthetic_results_path": str(synthetic_graph_dir / "results.jsonl"),
                "data_graph_dir": str(data_graph_dir),
                "synthetic_graph_dir": str(synthetic_graph_dir),
                "split_key": split.get("split_key"),
                "train_code_bases": split.get("train_code_bases", []),
                "test_code_bases": split.get("test_code_bases", []),
            },
            sort_keys=True,
        )
    )


def _load_graph_for_gnn(example: Dict[str, Any]):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("GNN evaluation requires torch and torch-geometric.") from exc

    graph_pt = _path_or_empty(example.get("graph_pt_path"))
    if graph_pt is not None and graph_pt.exists():
        graph = torch.load(graph_pt, map_location="cpu", weights_only=False)
    else:
        graph_json = _path_or_empty(example.get("graph_json_path"))
        if graph_json is None or not graph_json.exists():
            raise FileNotFoundError(f"graph_json missing for {example.get('artifact_instance_id', '')}")
        payload = _artifact_payload(graph_json)
        payload["graph_label"] = int(example["label"])
        graph = build_pyg_heterodata(payload)

    relation = ("subtask", "grounds", "code")
    if graph[relation].edge_index.numel() == 0:
        training_path = _path_or_empty(example.get("training_example_path"))
        graph_json = _path_or_empty(example.get("graph_json_path"))
        if training_path is not None and graph_json is not None:
            training = _artifact_payload(training_path)
            grounding_path = _path_or_empty(training.get("artifact_paths", {}).get("grounding"))
            if grounding_path is not None:
                if not grounding_path.is_absolute():
                    grounding_path = training_path.parent / grounding_path
                if grounding_path.exists():
                    nodes = _artifact_payload(graph_json).get("code_nodes", [])
                    node_indices = {str(node.get("node_id")): index for index, node in enumerate(nodes)}
                    links = _artifact_payload(grounding_path).get("links", [])
                    pairs = []
                    for link in links:
                        index = link.get("subtask_index")
                        if not isinstance(index, int) or not 0 <= index < graph["subtask"].num_nodes:
                            continue
                        for node_id in link.get("node_ids", []):
                            target = node_indices.get(str(node_id))
                            if target is not None:
                                pairs.append([index, target])
                    if pairs:
                        graph[relation].edge_index = torch.tensor(pairs, dtype=torch.long).T.contiguous()

    graph.y = torch.tensor([int(example["label"])], dtype=torch.long)
    return graph


def _run_feature_eval(
    *,
    train: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    seed: int,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    columns = feature_columns([*train, *test])
    atomic_write_json(
        out_dir / "artifacts" / "feature_columns.json",
        {"feature_columns": columns, "feature_column_count": len(columns)},
    )
    metrics_by_model = train_feature_models(train, test, columns, seed)
    rows: List[Dict[str, Any]] = []
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "method": "feature_classifiers",
                "model_name": model_name,
                "status": "error" if "error" in metrics else "success",
                "metrics": metrics,
                "feature_column_count": len(columns),
                "timestamp": _utc_now(),
            }
        )
    return rows


def _run_gnn_eval(
    *,
    train: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    out_dir: Path,
) -> List[Dict[str, Any]]:
    from src.baseline.structural_misalignment.embeddings import clear_embedding_encoder_cache
    from src.baseline.structural_misalignment.models.train import train_graph_model

    gnn_config = config.get("gnn") if isinstance(config.get("gnn"), dict) else {}
    model_dir = out_dir / "artifacts" / str(gnn_config.get("output_subdir", "gnn_model"))
    train_graphs = [_load_graph_for_gnn(example) for example in train]
    test_graphs = [_load_graph_for_gnn(example) for example in test]
    clear_embedding_encoder_cache()
    metadata = train_graph_model(
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        output_dir=model_dir,
        hidden_dim=int(gnn_config.get("hidden_dim", 128)),
        dropout=float(gnn_config.get("dropout", 0.1)),
        learning_rate=float(gnn_config.get("learning_rate", 1e-3)),
        epochs=int(gnn_config.get("epochs", 10)),
        batch_size=int(gnn_config.get("batch_size", 8)),
        seed=int(gnn_config.get("seed", config.get("seed", 42))),
        embedding_model_name=str(gnn_config.get("embedding_model_name", "microsoft/codebert-base")),
        embedding_pooling=str(gnn_config.get("embedding_pooling", "mean")),
        device=str(gnn_config.get("device", "auto")),
    )
    return [
        {
            "method": "gnn",
            "model_name": str(metadata.get("gnn_model_type", "hetero_sage")),
            "status": "success",
            "metrics": metadata.get("metrics", {}),
            "model_dir": str(model_dir),
            "timestamp": _utc_now(),
        }
    ]


def run_structural_misalignment_eval(
    *,
    data_rows_path: Path | None = None,
    data_graph_dir: Path | None = None,
    synthetic_graph_dir: Path | None = None,
    out_dir: Path,
    config_path: Path | None = None,
    config: Dict[str, Any] | None = None,
    baseline_name: str = "",
    baseline_plugin: str = "structural_misalignment_eval",
    baseline_config_hash: str | None = None,
    train_ratio: float | None = None,
    seed: int | None = None,
    min_candidate_nodes: int | None = None,
) -> List[Dict[str, Any]]:
    t0 = time.time()
    out_dir = out_dir.expanduser().resolve()
    if config_path is not None:
        config_path = config_path.expanduser().resolve()
    if config is None:
        if config_path is None:
            raise ValueError("run_structural_misalignment_eval requires config or config_path")
        config = load_yaml(config_path)
    else:
        config = dict(config)
    if config_path is None and config.get("_config_path"):
        config_path = Path(str(config["_config_path"])).expanduser().resolve()
    config.pop("_config_path", None)
    config_base_dir = config_path.parent if config_path is not None else Path.cwd()
    data_rows_path = _resolve_required_path(
        data_rows_path,
        config,
        "data_rows",
        "data_rows_path",
        base_dir=config_base_dir,
    )
    data_graph_dir = _resolve_required_path(
        data_graph_dir,
        config,
        "data_graph_dir",
        "data_graphs",
        base_dir=config_base_dir,
    )
    synthetic_graph_dir = _resolve_required_path(
        synthetic_graph_dir,
        config,
        "synthetic_graph_dir",
        "synthetic_graphs",
        base_dir=config_base_dir,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_hash = baseline_config_hash or config_hash(config)
    resolved_baseline_name = baseline_name or str(config.get("name", "structural_misalignment_eval"))
    method = str(config.get("method", "feature_classifiers")).strip()
    resolved_train_ratio = float(train_ratio if train_ratio is not None else config.get("train_ratio", 0.7))
    resolved_seed = int(seed if seed is not None else config.get("seed", 42))
    resolved_min_nodes = int(
        min_candidate_nodes
        if min_candidate_nodes is not None
        else config.get("min_candidate_nodes", 1)
    )

    data_examples, data_report = _load_graph_examples(
        rows_path=data_rows_path,
        graph_dir=data_graph_dir,
        source_name="real",
        min_candidate_nodes=resolved_min_nodes,
    )
    synthetic_examples, synthetic_report = _load_graph_examples(
        rows_path=None,
        graph_dir=synthetic_graph_dir,
        source_name="synthetic",
        min_candidate_nodes=resolved_min_nodes,
    )

    data_rows = load_rows(data_rows_path)
    synthetic_results = _read_jsonl(synthetic_graph_dir / "results.jsonl")
    data_graph_code_bases = {example["code_base"] for example in data_examples}
    synthetic_graph_code_bases = {example["code_base"] for example in synthetic_examples}
    synthetic_paired_code_bases = (
        {example["code_base"] for example in synthetic_examples if example["label"] == 0}
        & {example["code_base"] for example in synthetic_examples if example["label"] == 1}
    )
    eligible_code_bases = sorted(data_graph_code_bases & synthetic_paired_code_bases)
    if len(eligible_code_bases) <= 1:
        raise ValueError(
            "Structural evaluation requires graph examples for at least two shared "
            f"code_base groups, got {len(eligible_code_bases)}."
        )
    train_ids, test_ids, split = _split_code_base_ids(
        eligible_code_bases,
        train_ratio=resolved_train_ratio,
        seed=resolved_seed,
    )
    train = [example for example in synthetic_examples if example["code_base"] in train_ids]
    test = [example for example in data_examples if example["code_base"] in test_ids]
    _validate_examples(train, test)

    split_manifest = {
        **split,
        "data_row_count": len(data_rows),
        "synthetic_row_count": len(synthetic_results),
        "data_row_code_base_count": len({str(row["code_base"]) for row in data_rows}),
        "synthetic_row_code_base_count": len({str(row["code_base"]) for row in synthetic_results}),
        "data_graph_code_base_count": len(data_graph_code_bases),
        "synthetic_graph_code_base_count": len(synthetic_graph_code_bases),
        "synthetic_paired_code_base_count": len(synthetic_paired_code_bases),
        "eligible_code_base_count": len(eligible_code_bases),
        "eligible_code_bases": eligible_code_bases,
        "excluded_data_graph_only_code_bases": sorted(data_graph_code_bases - synthetic_graph_code_bases),
        "excluded_synthetic_graph_only_code_bases": sorted(synthetic_graph_code_bases - data_graph_code_bases),
        "train_example_count": len(train),
        "test_example_count": len(test),
        "train_label_counts": dict(Counter(example["label"] for example in train)),
        "test_label_counts": dict(Counter(example["label"] for example in test)),
        "data_row_label_counts": label_counts(data_rows),
        "synthetic_row_label_counts": dict(Counter(int(row["label"]) for row in synthetic_results if row.get("label") in {0, 1})),
    }
    atomic_write_json(out_dir / "artifacts" / "split_manifest.json", split_manifest)
    atomic_write_json(out_dir / "artifacts" / "load_report.json", {"data": data_report, "synthetic": synthetic_report})
    atomic_write_json(
        out_dir / "integration_spec.json",
        {
            "schema_version": "structural_misalignment_eval_v1",
            "baseline_name": resolved_baseline_name,
            "baseline_plugin": baseline_plugin,
            "config_path": str(config_path or ""),
            "config_hash": cfg_hash,
            "baseline_config_hash": cfg_hash,
            "config": config,
            "method": method,
            "data_rows_path": str(data_rows_path),
            "synthetic_results_path": str(synthetic_graph_dir / "results.jsonl"),
            "data_graph_dir": str(data_graph_dir),
            "synthetic_graph_dir": str(synthetic_graph_dir),
            "train_ratio": resolved_train_ratio,
            "seed": resolved_seed,
            "min_candidate_nodes": resolved_min_nodes,
            "split_manifest_path": str(out_dir / "artifacts" / "split_manifest.json"),
        },
    )

    print(
        f"[structural_misalignment_eval] method={method} train={len(train)} "
        f"test={len(test)} out={out_dir}",
        flush=True,
    )

    result_rows: List[Dict[str, Any]] = []
    if method in {"feature_classifiers", "both"}:
        result_rows.extend(
            _run_feature_eval(
                train=train,
                test=test,
                seed=resolved_seed,
                out_dir=out_dir,
            )
        )
    if method in {"gnn", "both"}:
        result_rows.extend(
            _run_gnn_eval(
                train=train,
                test=test,
                config={**config, "seed": resolved_seed},
                out_dir=out_dir,
            )
        )
    if method not in {"feature_classifiers", "gnn", "both"}:
        raise ValueError(f"Unsupported structural misalignment eval method: {method}")

    common = {
        "baseline_name": resolved_baseline_name,
        "baseline_plugin": baseline_plugin,
        "baseline_config_hash": cfg_hash,
        "config_hash": cfg_hash,
        "train_example_count": len(train),
        "test_example_count": len(test),
        "train_code_base_count": split_manifest["train_code_base_count"],
        "test_code_base_count": split_manifest["test_code_base_count"],
        "train_label_counts": split_manifest["train_label_counts"],
        "test_label_counts": split_manifest["test_label_counts"],
        "runtime_sec": round(time.time() - t0, 6),
    }
    final_rows = []
    for row in result_rows:
        model_name = str(row.get("model_name", method))
        merged = {
            **common,
            **row,
            "row_index": -1,
            "artifact_instance_id": f"structural_misalignment_eval__{method}__{model_name}",
            "code_base": "__dataset__",
            "label": None,
        }
        merged["row_key"] = _result_key(
            baseline_config_hash=cfg_hash,
            method=method,
            model_name=model_name,
            data_rows_path=data_rows_path,
            data_graph_dir=data_graph_dir,
            synthetic_graph_dir=synthetic_graph_dir,
            split=split,
        )
        final_rows.append(merged)
    _append_results(out_dir / "results.jsonl", final_rows)

    print(
        f"[structural_misalignment_eval] complete rows={len(final_rows)} "
        f"runtime_sec={round(time.time() - t0, 6)}",
        flush=True,
    )
    return final_rows
