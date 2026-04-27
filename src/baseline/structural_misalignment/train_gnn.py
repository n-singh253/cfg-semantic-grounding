"""Train the structural misalignment hetero GNN from finalized attack datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.common.config import load_yaml
from src.common.llm import LLMClient
from src.eval.attack_finalize import require_finalized_attack_rows
from src.eval.report import load_jsonl_rows
from src.baseline.structural_misalignment.models.train import train_graph_model
from src.baseline.structural_misalignment.pipeline import build_structural_graph


def _load_prompt(row: Dict[str, Any]) -> str:
    artifact_path = Path(str(row.get("attack_artifact_path", "")))
    for name in ("adv_prompt.txt", "final_adv_prompt.txt", "original_prompt.txt"):
        candidate = artifact_path / name
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
    return ""


def _build_graphs(
    rows: List[Dict[str, Any]],
    *,
    artifact_root: Path,
    config: Dict[str, Any],
    llm_client: LLMClient,
) -> List[Any]:
    graphs: List[Any] = []
    manifest: List[Dict[str, Any]] = []
    artifact_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        prompt = _load_prompt(row)
        adv_patch_path = Path(str(row.get("patch_artifacts", {}).get("adv_patch_path", "")))
        patch_text = adv_patch_path.read_text(encoding="utf-8").strip() if adv_patch_path.exists() else ""
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
            artifact_root=artifact_root / str(row.get("instance_id", "unknown")),
            llm_client=llm_client,
            module_name="structural_misalignment_train",
            module_config_hash="train",
            fidelity_mode="surrogate_debug",
            graph_label=int(row.get("graph_label", 0)),
        )
        graphs.append(hetero_graph)
        manifest.append(
            {
                "instance_id": row.get("instance_id", "unknown"),
                "split": row.get("split", ""),
                "graph_label": int(row.get("graph_label", 0)),
                "graph_artifacts": metadata.get("artifact_paths", {}),
            }
        )
    (artifact_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return graphs


def _split_rows(rows: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get("split", "")).strip().lower() == split_name]


def _load_rows_from_paths(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Finalized attack dataset not found: {path}")
        loaded = load_jsonl_rows(path)
        require_finalized_attack_rows(loaded, path)
        rows.extend(loaded)
    return rows


def _path_list(value: Any) -> List[Path]:
    if isinstance(value, list):
        return [Path(str(item)) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [Path(value)]
    return []


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
        train_rows = _load_rows_from_paths(train_benign_paths) + _load_rows_from_paths(train_malicious_paths)
        test_rows = _load_rows_from_paths(test_benign_paths) + _load_rows_from_paths(test_malicious_paths)
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

    benign_rows = _load_rows_from_paths(benign_paths)
    malicious_rows = _load_rows_from_paths(malicious_paths)
    train_rows = _split_rows(benign_rows, "train") + _split_rows(malicious_rows, "train")
    test_rows = _split_rows(benign_rows, "test") + _split_rows(malicious_rows, "test")
    return train_rows, test_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train structural misalignment hetero GNN")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    data_config = config.get("data_preparation", {}) if isinstance(config.get("data_preparation"), dict) else {}
    train_config = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    train_rows, test_rows = _resolve_split_rows(data_config)
    if not train_rows or not test_rows:
        raise ValueError("Training requires both original train and test split rows; no re-splitting is allowed.")

    output_dir = Path(str(train_config.get("output_dir", "data/models/structural_misalignment/hetero_gnn")))
    llm_client = LLMClient(output_dir / "llm_cache_unused")
    graph_config = dict(config.get("graph_pipeline", {})) if isinstance(config.get("graph_pipeline"), dict) else {}
    if "embedding_model_name" in train_config:
        graph_config["embedding_model_name"] = train_config["embedding_model_name"]
    if "embedding_pooling" in train_config:
        graph_config["embedding_pooling"] = train_config["embedding_pooling"]
    if "link_similarity_threshold" in train_config:
        graph_config["link_similarity_threshold"] = train_config["link_similarity_threshold"]
    if "link_topk_fallback" in train_config:
        graph_config["link_topk_fallback"] = train_config["link_topk_fallback"]

    train_graphs = _build_graphs(train_rows, artifact_root=output_dir / "graphs" / "train", config=graph_config, llm_client=llm_client)
    test_graphs = _build_graphs(test_rows, artifact_root=output_dir / "graphs" / "test", config=graph_config, llm_client=llm_client)
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
