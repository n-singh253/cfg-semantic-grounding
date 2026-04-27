"""Structural misalignment defense baseline backed by a graph neural network."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.baseline.base import BaseDefense
from src.baseline.registry import register_baseline
from src.baseline.structural_misalignment.models.infer import decide_from_policy, predict_graph_probability
from src.baseline.structural_misalignment.models.load import load_graph_model_bundle
from src.baseline.structural_misalignment.pipeline import build_structural_graph


class StructuralMisalignmentDefense(BaseDefense):
    name = "structural_misalignment"

    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ):
        del all_tests
        decision_policy = str(self.config.get("decision_policy", "reject_if_score_ge_threshold")).strip()
        threshold = float(self.config.get("threshold", 0.5))
        model_path = str(self.config.get("gnn_model_path") or self.config.get("model_path") or "").strip()
        if not model_path:
            self.last_signals = {
                "error": "Missing required config: gnn_model_path",
                "failure_flags": {"graph_build_fail": False, "model_missing": True, "inference_fail": False},
            }
            return False

        instance_id = str(repo_code.get("instance_id", "unknown"))
        defense_root = self.run_root / "artifacts" / "defenses" / instance_id / self.name
        try:
            graph_payload, hetero_graph, graph_meta = build_structural_graph(
                prompt=prompt,
                patch_text=str(code_or_patch or ""),
                repo_code=repo_code,
                config=self.config,
                artifact_root=defense_root,
                llm_client=self.llm_client,
                module_name=self.name,
                module_config_hash=self.baseline_config_hash,
                fidelity_mode=self.fidelity_mode,
                graph_label=0,
            )
            bundle = load_graph_model_bundle(model_path)
            inference = predict_graph_probability(bundle, hetero_graph)
            accepted = decide_from_policy(inference.injection_probability, threshold, decision_policy)
            self.last_signals = {
                "threshold": threshold,
                "decision_policy": decision_policy,
                "graph_label_placeholder": 0,
                "injection_probability": inference.injection_probability,
                "predicted_label": inference.prediction,
                "accepted": bool(accepted),
                "graph_summary": {
                    "subtask_count": len(graph_payload.get("subtasks", [])),
                    "code_node_count": len(graph_payload.get("code_nodes", [])),
                    "subtask_dependency_edges": len(graph_payload.get("edges", {}).get("subtask_to_subtask", [])),
                    "code_cfg_edges": len(graph_payload.get("edges", {}).get("code_to_code", [])),
                    "subtask_code_edges": len(graph_payload.get("edges", {}).get("subtask_to_code", [])),
                },
                "artifact_paths": graph_meta.get("artifact_paths", {}),
                "embedding_model_name": graph_meta.get("embedding_model_name", ""),
                "embedding_pooling": graph_meta.get("embedding_pooling", ""),
                "parsers": {
                    "prompt_parser": graph_meta.get("prompt_parser", ""),
                    "patch_parser": graph_meta.get("patch_parser", ""),
                    "linker": graph_meta.get("linker", ""),
                },
                "model_metadata": bundle.metadata,
                "failure_flags": {"graph_build_fail": False, "model_missing": False, "inference_fail": False},
            }
            return True if accepted else False
        except FileNotFoundError as exc:
            self.last_signals = {
                "error": f"{type(exc).__name__}: {exc}",
                "failure_flags": {"graph_build_fail": False, "model_missing": True, "inference_fail": False},
            }
            return False
        except Exception as exc:
            self.last_signals = {
                "error": f"{type(exc).__name__}: {exc}",
                "failure_flags": {"graph_build_fail": True, "model_missing": False, "inference_fail": True},
                "artifact_root": str(Path(defense_root)),
            }
            return False


register_baseline("structural_misalignment")(StructuralMisalignmentDefense)
