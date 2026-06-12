"""Relink completed structural graph caches with a new embedding threshold.

This utility reuses cached subtasks, code nodes, embeddings, and CFG edges from
an existing structural-misalignment graph cache. It does not call an LLM or
rerun the patch parser.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.baseline.structural_misalignment.embeddings import cosine_similarity_matrix
from src.baseline.structural_misalignment.graph.build import write_graph_artifacts

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional progress dependency.
    tqdm = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", required=True, type=Path)
    parser.add_argument("--output-model-dir", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--topk-per-subtask", type=int, default=3)
    parser.add_argument("--topk-fallback", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional graph limit per split for smoke tests.")
    parser.add_argument(
        "--fail-on-fallback-subtasks",
        action="store_true",
        help="Fail if any source graph used deterministic subtask fallback.",
    )
    return parser.parse_args()


def _subtask_to_subtask_edges(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    subtask_id_to_index = {str(subtask.get("subtask_id", "")): idx for idx, subtask in enumerate(subtasks)}
    edges: List[Dict[str, Any]] = []
    for subtask in subtasks:
        dst = subtask_id_to_index.get(str(subtask.get("subtask_id", "")))
        if dst is None:
            continue
        for dependency in subtask.get("depends_on", []):
            src = subtask_id_to_index.get(str(dependency))
            if src is None:
                continue
            edges.append({"src": src, "dst": dst, "kind": "depends_on"})
    return edges


def _relink_graph(
    graph: Dict[str, Any],
    *,
    threshold: float,
    topk_per_subtask: int,
    topk_fallback: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    subtasks = list(graph.get("subtasks", []))
    code_nodes = list(graph.get("code_nodes", []))
    subtask_features = np.asarray(graph.get("subtask_features", []), dtype=np.float32)
    code_features = np.asarray(graph.get("code_features", []), dtype=np.float32)
    node_ids = [str(node.get("node_id", "")) for node in code_nodes]
    similarity = cosine_similarity_matrix(subtask_features, code_features)

    cross_edges: List[Dict[str, Any]] = []
    total_links = 0
    fallback_subtasks = 0
    zero_before_fallback = 0
    for subtask_idx in range(len(subtasks)):
        ranked = sorted(
            [
                (float(similarity[subtask_idx, node_idx]), node_idx, node_ids[node_idx])
                for node_idx in range(len(code_nodes))
            ],
            key=lambda item: (-item[0], item[2]),
        )
        selected = [(score, node_idx, node_id) for score, node_idx, node_id in ranked if score >= threshold]
        if not selected:
            zero_before_fallback += 1
        if topk_per_subtask > 0:
            selected = selected[:topk_per_subtask]
        fallback_used = False
        if not selected and ranked and topk_fallback > 0:
            selected = ranked[:topk_fallback]
            fallback_used = True
            fallback_subtasks += 1
        for score, node_idx, _node_id in selected:
            cross_edges.append(
                {
                    "src": subtask_idx,
                    "dst": node_idx,
                    "score": score,
                    "fallback_used": fallback_used,
                }
            )
        total_links += len(selected)

    relinked = {
        **graph,
        "edges": {
            "subtask_to_subtask": _subtask_to_subtask_edges(subtasks),
            "code_to_code": list(graph.get("edges", {}).get("code_to_code", [])),
            "subtask_to_code": cross_edges,
        },
    }
    metadata = {
        "similarity_threshold": threshold,
        "link_topk_per_subtask": topk_per_subtask,
        "link_topk_fallback": topk_fallback,
        "subtask_count": len(subtasks),
        "node_count": len(code_nodes),
        "link_count": total_links,
        "fallback_subtasks": fallback_subtasks,
        "zero_link_subtasks_before_fallback": zero_before_fallback,
    }
    return relinked, metadata


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _process_split(
    *,
    split: str,
    source_model_dir: Path,
    output_model_dir: Path,
    threshold: float,
    topk_per_subtask: int,
    topk_fallback: int,
    fail_on_fallback_subtasks: bool,
    limit: int | None,
) -> Dict[str, Any]:
    source_split = source_model_dir / "graphs" / split
    output_split = output_model_dir / "graphs" / split
    graph_paths = sorted(source_split.glob("*/graph/graph.json"))
    if limit is not None:
        graph_paths = graph_paths[: max(0, int(limit))]
    iterator = graph_paths
    if tqdm is not None:
        iterator = tqdm(graph_paths, desc=f"relink:{split}", unit="graph")

    manifest: List[Dict[str, Any]] = []
    fallback_graphs = 0
    total_links = 0
    total_fallback_subtasks = 0
    total_zero_before_fallback = 0
    for graph_path in iterator:
        graph_key = graph_path.parents[1].name
        source_graph_root = graph_path.parents[1]
        output_graph_root = output_split / graph_key
        source_fallback_metadata = source_graph_root / "subtasks" / "fallback_metadata.json"
        used_subtask_fallback = source_fallback_metadata.exists()
        if used_subtask_fallback:
            fallback_graphs += 1
            if fail_on_fallback_subtasks:
                raise RuntimeError(f"Source graph used deterministic subtask fallback: {source_graph_root}")

        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        relinked, link_metadata = _relink_graph(
            graph,
            threshold=threshold,
            topk_per_subtask=topk_per_subtask,
            topk_fallback=topk_fallback,
        )
        graph_artifacts = write_graph_artifacts(output_graph_root / "graph", relinked)
        grounding_dir = output_graph_root / "grounding"
        grounding_dir.mkdir(parents=True, exist_ok=True)
        (grounding_dir / "relink_metadata.json").write_text(
            json.dumps(link_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _copy_if_exists(source_fallback_metadata, output_graph_root / "subtasks" / "fallback_metadata.json")
        _copy_if_exists(source_graph_root / "subtasks" / "metadata.json", output_graph_root / "subtasks" / "metadata.json")
        _copy_if_exists(source_graph_root / "subtasks" / "response.txt", output_graph_root / "subtasks" / "response.txt")
        _copy_if_exists(source_graph_root / "subtasks" / "prompt.txt", output_graph_root / "subtasks" / "prompt.txt")

        manifest.append(
            {
                "graph_key": graph_key,
                "instance_id": graph.get("instance_id", "unknown"),
                "graph_label": int(graph.get("graph_label", 0)),
                "graph_artifacts": graph_artifacts,
                "source_graph_json": str(graph_path),
                "used_subtask_fallback": used_subtask_fallback,
                "relink_metadata": str(grounding_dir / "relink_metadata.json"),
            }
        )
        total_links += int(link_metadata["link_count"])
        total_fallback_subtasks += int(link_metadata["fallback_subtasks"])
        total_zero_before_fallback += int(link_metadata["zero_link_subtasks_before_fallback"])

    output_split.mkdir(parents=True, exist_ok=True)
    (output_split / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "split": split,
        "graph_count": len(manifest),
        "source_subtask_fallback_graph_count": fallback_graphs,
        "link_count": total_links,
        "fallback_subtasks": total_fallback_subtasks,
        "zero_link_subtasks_before_fallback": total_zero_before_fallback,
    }


def main() -> int:
    args = _parse_args()
    args.output_model_dir.mkdir(parents=True, exist_ok=True)
    split_summaries = [
        _process_split(
            split=split,
            source_model_dir=args.source_model_dir,
            output_model_dir=args.output_model_dir,
            threshold=args.threshold,
            topk_per_subtask=args.topk_per_subtask,
            topk_fallback=args.topk_fallback,
            fail_on_fallback_subtasks=bool(args.fail_on_fallback_subtasks),
            limit=args.limit,
        )
        for split in ("train", "test")
    ]
    summary = {
        "source_model_dir": str(args.source_model_dir),
        "output_model_dir": str(args.output_model_dir),
        "threshold": args.threshold,
        "topk_per_subtask": args.topk_per_subtask,
        "topk_fallback": args.topk_fallback,
        "splits": split_summaries,
    }
    (args.output_model_dir / "relink_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
