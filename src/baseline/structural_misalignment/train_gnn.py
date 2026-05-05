"""Train the structural misalignment hetero GNN from finalized attack datasets."""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from src.common.config import config_hash, load_yaml
from src.common.llm import LLMClient
from src.eval.attack_finalize import require_finalized_attack_rows
from src.eval.report import load_jsonl_rows
from src.baseline.structural_misalignment.graph.cache import structural_graph_key
from src.baseline.structural_misalignment.models.train import train_graph_model
from src.baseline.structural_misalignment.pipeline import build_structural_graph

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional progress dependency.
    tqdm = None


def _load_prompt(row: Dict[str, Any]) -> str:
    artifact_path = Path(str(row.get("attack_artifact_path", "")))
    for name in ("adv_prompt.txt", "final_adv_prompt.txt", "original_prompt.txt"):
        candidate = artifact_path / name
        if candidate.exists():
            prompt = candidate.read_text(encoding="utf-8").strip()
            if prompt:
                return prompt
    raise FileNotFoundError(
        "No nonempty attack prompt artifact found for "
        f"{row.get('instance_id', 'unknown')} under {artifact_path}"
    )


def _load_patch(row: Dict[str, Any]) -> str:
    patch_artifacts = row.get("patch_artifacts", {})
    if not isinstance(patch_artifacts, dict):
        patch_artifacts = {}
    for key in ("final_patch_path", "adv_patch_path", "ori_patch_path"):
        candidate = Path(str(patch_artifacts.get(key, "")))
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="replace").strip()
    validation = row.get("attack_validation", {})
    if isinstance(validation, dict):
        apply_details = validation.get("apply_details", {})
        if isinstance(apply_details, dict):
            diff = str(apply_details.get("sanitized_diff", ""))
            if diff.strip():
                return diff.strip()
    raise FileNotFoundError(
        "No nonempty patch artifact found for "
        f"{row.get('instance_id', 'unknown')} in finalized attack row"
    )


def _build_graphs(
    rows: List[Dict[str, Any]],
    *,
    artifact_root: Path,
    config: Dict[str, Any],
    llm_client: LLMClient,
    fidelity_mode: str,
    workers: int = 1,
) -> tuple[List[Any], List[Dict[str, Any]]]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    graph_config_hash = config_hash(config)

    def build_one(row: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
        prompt = _load_prompt(row)
        patch_text = _load_patch(row)
        graph_key = structural_graph_key(row)
        graph_payload, hetero_graph, metadata = build_structural_graph(
            prompt=prompt,
            patch_text=patch_text,
            repo_code={
                "instance_id": row.get("instance_id", "unknown"),
                "path": row.get("repo_path", ""),
                "dataset": row.get("dataset", ""),
                "split": row.get("split", ""),
            },
            config=config,
            artifact_root=artifact_root / graph_key,
            llm_client=llm_client,
            module_name="structural_misalignment_train",
            module_config_hash=graph_config_hash,
            fidelity_mode=fidelity_mode,
            graph_label=int(row.get("graph_label", 0)),
        )
        manifest_row = {
            "instance_id": row.get("instance_id", "unknown"),
            "split": row.get("split", ""),
            "graph_label": int(row.get("graph_label", 0)),
            "attack_name": row.get("attack_name", ""),
            "patch_hash": row.get("patch_hash", ""),
            "source_attack_dataset_path": row.get("source_attack_dataset_path", ""),
            "graph_key": graph_key,
            "graph_artifacts": metadata.get("artifact_paths", {}),
        }
        return hetero_graph, manifest_row

    graphs: List[Any] = []
    manifest: List[Dict[str, Any]] = []
    worker_count = max(1, int(workers))
    if worker_count == 1 or len(rows) <= 1:
        iterator = rows
        if tqdm is not None:
            iterator = tqdm(rows, desc=f"build-graphs:{artifact_root.name}", unit="graph")
        for row in iterator:
            hetero_graph, manifest_row = build_one(row)
            graphs.append(hetero_graph)
            manifest.append(manifest_row)
    else:
        futures = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(build_one, row) for row in rows]
            iterator = as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc=f"build-graphs:{artifact_root.name}", unit="graph")
            for future in iterator:
                hetero_graph, manifest_row = future.result()
                graphs.append(hetero_graph)
                manifest.append(manifest_row)
    (artifact_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return graphs, manifest


def _split_rows(rows: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get("split", "")).strip().lower() == split_name]


def _load_rows_from_paths(paths: List[Path], *, graph_label: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Finalized attack dataset not found: {path}")
        loaded = load_jsonl_rows(path)
        require_finalized_attack_rows(loaded, path)
        loaded = [{**row, "source_attack_dataset_path": str(path)} for row in loaded]
        if graph_label is not None:
            loaded = [{**row, "graph_label": int(graph_label)} for row in loaded]
        rows.extend(loaded)
    return rows


def _path_list(value: Any) -> List[Path]:
    if isinstance(value, list):
        return [Path(str(item)) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [Path(value)]
    return []


def _split_rows_by_instance(
    rows: List[Dict[str, Any]],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not rows:
        return [], []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("instance_id", "unknown")), []).append(row)
    instance_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(instance_ids)
    test_count = max(1, int(round(len(instance_ids) * test_fraction)))
    if len(instance_ids) > 1:
        test_count = min(test_count, len(instance_ids) - 1)
    test_ids = set(instance_ids[:test_count])
    train_rows = [row for iid in instance_ids if iid not in test_ids for row in grouped[iid]]
    test_rows = [row for iid in instance_ids if iid in test_ids for row in grouped[iid]]
    return train_rows, test_rows


def _resolve_split_rows(data_config: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_benign_paths = _path_list(data_config.get("train_benign_attack_dataset_paths"))
    train_malicious_paths = _path_list(data_config.get("train_malicious_attack_dataset_paths"))
    test_benign_paths = _path_list(data_config.get("test_benign_attack_dataset_paths"))
    test_malicious_paths = _path_list(data_config.get("test_malicious_attack_dataset_paths"))

    if train_benign_paths or train_malicious_paths or test_benign_paths or test_malicious_paths:
        if not (train_benign_paths and train_malicious_paths and test_benign_paths and test_malicious_paths):
            raise ValueError(
                "Split-specific training requires all four path groups: "
                "train/test x benign/malicious attack dataset paths."
            )
        train_rows = _load_rows_from_paths(train_benign_paths, graph_label=0) + _load_rows_from_paths(
            train_malicious_paths, graph_label=1
        )
        test_rows = _load_rows_from_paths(test_benign_paths, graph_label=0) + _load_rows_from_paths(
            test_malicious_paths, graph_label=1
        )
        return train_rows, test_rows

    benign_paths = _path_list(data_config.get("benign_attack_dataset_paths"))
    malicious_paths = _path_list(data_config.get("malicious_attack_dataset_paths"))
    if not benign_paths:
        benign_paths = _path_list(data_config.get("benign_attack_dataset_path"))
    if not malicious_paths:
        malicious_paths = _path_list(data_config.get("malicious_attack_dataset_path"))
    if not benign_paths or not malicious_paths:
        raise FileNotFoundError(
            "Training requires either split-specific finalized dataset paths or combined benign/malicious finalized dataset paths."
        )

    benign_rows = _load_rows_from_paths(benign_paths, graph_label=0)
    malicious_rows = _load_rows_from_paths(malicious_paths, graph_label=1)
    train_rows = _split_rows(benign_rows, "train") + _split_rows(malicious_rows, "train")
    test_rows = _split_rows(benign_rows, "test") + _split_rows(malicious_rows, "test")
    if not train_rows or not test_rows:
        test_fraction = float(data_config.get("test_fraction", 0.2))
        if not (0.0 < test_fraction < 1.0):
            raise ValueError("data_preparation.test_fraction must be between 0 and 1")
        split_seed = int(data_config.get("split_seed", data_config.get("seed", 42)))
        train_rows, test_rows = _split_rows_by_instance(
            benign_rows + malicious_rows,
            test_fraction=test_fraction,
            seed=split_seed,
        )
    return train_rows, test_rows


def _counts_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "") or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _write_split_manifest(output_dir: Path, train_rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]]) -> None:
    train_ids = sorted({str(row.get("instance_id", "")) for row in train_rows if row.get("instance_id")})
    test_ids = sorted({str(row.get("instance_id", "")) for row in test_rows if row.get("instance_id")})
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"Train/test instance-id leakage detected: {overlap[:10]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_instance_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (output_dir / "heldout_instance_ids.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    manifest = {
        "split_strategy": "by_instance",
        "train_row_count": len(train_rows),
        "heldout_row_count": len(test_rows),
        "train_instance_count": len(train_ids),
        "heldout_instance_count": len(test_ids),
        "train_label_counts": _counts_by(train_rows, "graph_label"),
        "heldout_label_counts": _counts_by(test_rows, "graph_label"),
        "train_attack_counts": _counts_by(train_rows, "attack_name"),
        "heldout_attack_counts": _counts_by(test_rows, "attack_name"),
        "train_source_counts": _counts_by(train_rows, "source_attack_dataset_path"),
        "heldout_source_counts": _counts_by(test_rows, "source_attack_dataset_path"),
        "train_instance_ids_path": str(output_dir / "train_instance_ids.txt"),
        "heldout_instance_ids_path": str(output_dir / "heldout_instance_ids.txt"),
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_graph_cache_index(
    output_dir: Path,
    train_manifest: List[Dict[str, Any]],
    test_manifest: List[Dict[str, Any]],
) -> None:
    index: Dict[str, Dict[str, Any]] = {}
    for split_name, rows in (("train", train_manifest), ("test", test_manifest)):
        for row in rows:
            graph_key = str(row.get("graph_key", ""))
            if not graph_key:
                continue
            index[graph_key] = {
                **row,
                "model_split": split_name,
            }
    (output_dir / "graph_cache_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train structural misalignment hetero GNN")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument(
        "--graph-workers",
        type=int,
        default=None,
        help="Parallel workers for graph construction. Defaults to training.graph_workers or 1.",
    )
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    data_config = config.get("data_preparation", {}) if isinstance(config.get("data_preparation"), dict) else {}
    train_config = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    train_rows, test_rows = _resolve_split_rows(data_config)
    if not train_rows or not test_rows:
        raise ValueError("Training requires both original train and test split rows; no re-splitting is allowed.")

    output_dir = Path(str(train_config.get("output_dir", "data/models/structural_misalignment/hetero_gnn")))
    _write_split_manifest(output_dir, train_rows, test_rows)
    llm_client = LLMClient(output_dir / "llm_cache")
    graph_config = dict(config.get("graph_pipeline", {})) if isinstance(config.get("graph_pipeline"), dict) else {}
    if "embedding_model_name" in train_config:
        graph_config["embedding_model_name"] = train_config["embedding_model_name"]
    if "embedding_pooling" in train_config:
        graph_config["embedding_pooling"] = train_config["embedding_pooling"]
    if "link_similarity_threshold" in train_config:
        graph_config["link_similarity_threshold"] = train_config["link_similarity_threshold"]
    if "link_topk_fallback" in train_config:
        graph_config["link_topk_fallback"] = train_config["link_topk_fallback"]

    fidelity_mode = str(config.get("fidelity_mode", "llm"))
    graph_workers = int(args.graph_workers if args.graph_workers is not None else train_config.get("graph_workers", 1))
    train_graphs, train_manifest = _build_graphs(
        train_rows,
        artifact_root=output_dir / "graphs" / "train",
        config=graph_config,
        llm_client=llm_client,
        fidelity_mode=fidelity_mode,
        workers=graph_workers,
    )
    test_graphs, test_manifest = _build_graphs(
        test_rows,
        artifact_root=output_dir / "graphs" / "test",
        config=graph_config,
        llm_client=llm_client,
        fidelity_mode=fidelity_mode,
        workers=graph_workers,
    )
    _write_graph_cache_index(output_dir, train_manifest, test_manifest)
    metrics = train_graph_model(
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        output_dir=output_dir,
        hidden_dim=int(train_config.get("hidden_dim", 128)),
        dropout=float(train_config.get("dropout", 0.1)),
        learning_rate=float(train_config.get("learning_rate", 1e-3)),
        epochs=int(train_config.get("epochs", 10)),
        batch_size=int(train_config.get("batch_size", 8)),
        seed=int(train_config.get("seed", 42)),
        embedding_model_name=str(graph_config.get("embedding_model_name", "microsoft/codebert-base")),
        embedding_pooling=str(graph_config.get("embedding_pooling", "mean")),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
