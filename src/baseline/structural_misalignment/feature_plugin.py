"""Extract-only structural misalignment plugin.

This plugin is for offline training-data generation, not final defense inference.

Execution model:
1. Receive one prompt + one patch from run_defense.
2. Extract changed code nodes/chunks from the patch.
3. Decompose the prompt into subtasks.
4. Link subtasks to changed code nodes/chunks.
5. Extract structural features.
6. Write all intermediate artifacts plus a compact training_example.json.
7. Return True iff extraction succeeded.

Important:
- This plugin does NOT load a trained model.
- This plugin does NOT run inference.
- True means "feature extraction succeeded", not "patch is safe".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# Trigger parser registrations.
import src.baseline.structural_misalignment.parsers.prompt.llm_subtasks  # noqa: F401
import src.baseline.structural_misalignment.parsers.patch.cfg_ast  # noqa: F401
import src.baseline.structural_misalignment.parsers.patch.cfg_ast_scoped  # noqa: F401
import src.baseline.structural_misalignment.parsers.patch.llm_chunks  # noqa: F401
import src.baseline.structural_misalignment.parsers.linking.llm_grounding  # noqa: F401
import src.baseline.structural_misalignment.parsers.linking.llm_grounding_iterative  # noqa: F401
import src.baseline.structural_misalignment.parsers.linking.embedding_similarity  # noqa: F401

from src.baseline.base import BaseDefense
from src.baseline.registry import register_baseline
from src.baseline.structural_misalignment.cfg.stats import compute_cfg_stats
from src.baseline.structural_misalignment.features.schema import (
    FEATURE_SCHEMA_VERSION,
    STRUCTURAL_FAMILY_MODES,
    feature_set_name,
    normalize_mode,
)
from src.baseline.structural_misalignment.features.structural_features import (
    SIMILARITY_ONLY_FEATURES,
    STRUCTURAL_COMBINED_FEATURES,
    STRUCTURAL_ONLY_FEATURES,
    compute_structural_feature_row,
)
from src.baseline.structural_misalignment.grounding.subtasks import DEFAULT_SYSTEM_PROMPT
from src.baseline.structural_misalignment.parsers.registry import (
    get_linker,
    get_patch_parser,
    get_prompt_parser,
)
from src.common.artifact_store import atomic_write_json
from src.common.artifacts import write_hashed_json_artifact
from src.common.hashing import sha256_text


class StructuralMisalignmentFeatureDefense(BaseDefense):
    """Extract subtasks, changed nodes, grounding links, and feature rows."""

    name = "structural_misalignment_features"

    def _failure_flags(self, failed_stage: str) -> Dict[str, bool]:
        return {
            "cfg_fail": failed_stage == "cfg",
            "subtasks_fail": failed_stage == "subtasks",
            "grounding_fail": failed_stage == "grounding",
            "features_fail": failed_stage == "features",
        }

    def _selected_columns_for_mode(self, mode: str) -> List[str]:
        if mode == "structural_only":
            return list(STRUCTURAL_ONLY_FEATURES)
        if mode == "similarity_only":
            return list(SIMILARITY_ONLY_FEATURES)
        if mode == "structural_combined":
            return list(STRUCTURAL_COMBINED_FEATURES)
        raise ValueError(f"Unsupported structural feature mode: {mode}")

    def _safe_prompt_type(self, repo_code: Dict[str, Any], patch_hash: str) -> str:
        prompt_type = str(repo_code.get("prompt_type", "")).strip()
        if prompt_type in {"ori", "adv"}:
            return prompt_type
        return f"unknown_{patch_hash[:12]}"

    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ):
        _ = all_tests

        raw_mode = str(self.config.get("mode", "structural_only")).strip()
        try:
            mode = normalize_mode(raw_mode)
        except Exception as exc:
            self.last_signals = {
                "extract_only": True,
                "mode": raw_mode,
                "error": f"Invalid mode: {type(exc).__name__}: {exc}",
                "failure_flags": self._failure_flags("features"),
            }
            return False

        if mode not in STRUCTURAL_FAMILY_MODES:
            self.last_signals = {
                "extract_only": True,
                "mode": mode,
                "error": (
                    "structural_misalignment_features currently supports only "
                    f"structural-family modes, got {mode}"
                ),
                "failure_flags": self._failure_flags("features"),
            }
            return False

        # Similarity modes require a train-fitted vectorizer. This extract-only
        # plugin intentionally starts with structural_only so the generated
        # examples can later be split into train/test before vectorizer fitting.
        include_similarity = mode in {"similarity_only", "structural_combined"}
        if include_similarity and not bool(self.config.get("allow_unfitted_similarity", False)):
            self.last_signals = {
                "extract_only": True,
                "mode": mode,
                "error": (
                    "Similarity modes require a train-fitted vectorizer. "
                    "Use mode: structural_only for the initial extraction pass, "
                    "or set allow_unfitted_similarity: true only for debugging."
                ),
                "failure_flags": self._failure_flags("features"),
            }
            return False

        parser_config = self.config.get("parsers", {})
        if not isinstance(parser_config, dict):
            parser_config = {}

        prompt_parser_name = str(parser_config.get("prompt", "llm_subtasks")).strip()
        patch_parser_name = str(parser_config.get("patch", "cfg_ast")).strip()
        linker_name = str(parser_config.get("linking", "llm_grounding")).strip()

        try:
            prompt_parser = get_prompt_parser(prompt_parser_name)
        except KeyError as exc:
            self.last_signals = {
                "extract_only": True,
                "mode": mode,
                "error": f"Unknown prompt parser: {exc}",
                "failure_flags": self._failure_flags("subtasks"),
            }
            return False

        try:
            patch_parser = get_patch_parser(patch_parser_name)
        except KeyError as exc:
            self.last_signals = {
                "extract_only": True,
                "mode": mode,
                "error": f"Unknown patch parser: {exc}",
                "failure_flags": self._failure_flags("cfg"),
            }
            return False

        try:
            linker = get_linker(linker_name)
        except KeyError as exc:
            self.last_signals = {
                "extract_only": True,
                "mode": mode,
                "error": f"Unknown linker: {exc}",
                "failure_flags": self._failure_flags("grounding"),
            }
            return False

        instance_id = str(repo_code.get("instance_id", "unknown"))
        dataset_name = str(repo_code.get("dataset", "")).strip().lower()
        repo_id = str(repo_code.get("repo_id", ""))
        base_commit = str(repo_code.get("base_commit", ""))

        patch_text = str(code_or_patch or "")
        patch_hash = str(repo_code.get("patch_hash", "") or sha256_text(patch_text))
        prompt_hash = str(repo_code.get("prompt_hash", "") or sha256_text(prompt or ""))
        prompt_type = self._safe_prompt_type(repo_code, patch_hash)
        label_hint_raw: Optional[Any] = repo_code.get("label_hint")
        label_hint = None
        if label_hint_raw in {0, 1, "0", "1"}:
            label_hint = int(label_hint_raw)
        elif prompt_type == "ori":
            label_hint = 0
        elif prompt_type == "adv":
            label_hint = 1

        defense_root = (
            self.run_root
            / "artifacts"
            / "defenses"
            / instance_id
            / prompt_type
            / self.name
        )
        defense_root.mkdir(parents=True, exist_ok=True)

        artifact_paths: Dict[str, str] = {}
        stage_status = {
            "cfg_ready": False,
            "subtasks_ready": False,
            "grounding_ready": False,
            "features_ready": False,
            "training_example_ready": False,
        }

        stage_status_path = defense_root / "stage_status.json"
        artifact_paths["stage_status"] = str(stage_status_path)

        current_stage = "init"

        def persist_stage_status(extra: Optional[Dict[str, Any]] = None) -> None:
            payload = {
                "stage_completed": stage_status,
                "mode": mode,
                "config_hash": self.baseline_config_hash,
                "instance_id": instance_id,
                "prompt_type": prompt_type,
                "label_hint": label_hint,
                "patch_hash": patch_hash,
                "prompt_hash": prompt_hash,
            }
            if extra:
                payload.update(extra)
            atomic_write_json(stage_status_path, payload)

        persist_stage_status()

        try:
            if not patch_text.strip():
                raise ValueError("empty_patch")

            if mode not in STRUCTURAL_FAMILY_MODES:
                raise ValueError(f"Unsupported extract-only mode: {mode}")

            selected_cols = self._selected_columns_for_mode(mode)
            canonical_feature_set = feature_set_name(mode)

            # ----- Stage 1: patch/code-node extraction -----
            current_stage = "cfg"

            allow_hunk_fallback = bool(self.config.get("allow_hunk_fallback", True))
            if dataset_name and dataset_name != "toy":
                allow_hunk_fallback = bool(
                    self.config.get(
                        "allow_hunk_fallback_non_toy",
                        allow_hunk_fallback,
                    )
                )

            base_repo = Path(str(repo_code.get("path", ""))).resolve()

            cfg_diff, candidate_nodes, cfg_diagnostics = patch_parser(
                patch_text,
                base_repo=base_repo if base_repo.exists() else None,
                allow_hunk_fallback=allow_hunk_fallback,
                config=self.config,
                artifact_dir=defense_root,
                llm_client=self.llm_client,
                instance_id=instance_id,
                module_name=self.name,
                module_config_hash=self.baseline_config_hash,
                fidelity_mode=self.fidelity_mode,
            )

            if cfg_diagnostics.get("fallback_used") and not allow_hunk_fallback:
                raise RuntimeError(
                    "CFG fallback disallowed but required: "
                    f"{cfg_diagnostics.get('fallback_reason', 'unknown')}"
                )

            cfg_stats = compute_cfg_stats(candidate_nodes)
            cfg_payload: Dict[str, Any] = {
                "cfg_stats": cfg_stats,
                "cfg_diff_summary": cfg_diff.get("summary", {}),
                "cfg_diagnostics": cfg_diagnostics,
                "candidate_node_count": len(candidate_nodes),
            }

            if cfg_diagnostics.get("risk_features"):
                cfg_payload["risk_features"] = cfg_diagnostics["risk_features"]
            if cfg_diagnostics.get("risk_analysis"):
                cfg_payload["risk_analysis_summary"] = cfg_diagnostics["risk_analysis"]

            cfg_stats_path = write_hashed_json_artifact(
                defense_root / "cfg_stats.json",
                cfg_payload,
                config_hash=self.baseline_config_hash,
                refs={
                    "instance_id": instance_id,
                    "prompt_type": prompt_type,
                    "patch_hash": patch_hash,
                },
            )
            artifact_paths["cfg_stats"] = str(cfg_stats_path)

            candidate_nodes_path = write_hashed_json_artifact(
                defense_root / "candidate_nodes.json",
                {
                    "candidate_nodes": candidate_nodes,
                    "candidate_node_count": len(candidate_nodes),
                },
                config_hash=self.baseline_config_hash,
                refs={
                    "cfg_stats": str(cfg_stats_path),
                    "patch_hash": patch_hash,
                },
            )
            artifact_paths["candidate_nodes"] = str(candidate_nodes_path)

            stage_status["cfg_ready"] = True
            persist_stage_status()

            # ----- Stage 2: prompt subtask extraction -----
            current_stage = "subtasks"

            llm_cfg = self.config.get("llm") if isinstance(self.config.get("llm"), dict) else {}
            provider = str(llm_cfg.get("provider", self.config.get("provider", "openai")))
            model = str(llm_cfg.get("model", self.config.get("model", "gpt-4o-mini")))
            temperature = float(llm_cfg.get("temperature", self.config.get("temperature", 0.2)))
            seed = llm_cfg.get("seed", self.config.get("seed"))
            max_retries = int(llm_cfg.get("max_retries", self.config.get("max_retries", 2)))
            backoff_sec = float(llm_cfg.get("backoff_sec", self.config.get("backoff_sec", 1.0)))
            allow_provider_fallback = bool(
                llm_cfg.get(
                    "allow_provider_fallback",
                    self.config.get("allow_provider_fallback", False),
                )
            )
            if dataset_name == "toy" and bool(self.config.get("allow_provider_fallback_on_toy", True)):
                allow_provider_fallback = True

            subtasks, subtasks_meta = prompt_parser(
                llm_client=self.llm_client,
                instance_id=instance_id,
                module_name=self.name,
                module_config_hash=self.baseline_config_hash,
                fidelity_mode=self.fidelity_mode,
                provider=provider,
                model=model,
                problem_statement=prompt,
                artifact_dir=defense_root / "llm" / "subtasks",
                temperature=temperature,
                seed=seed,
                max_retries=max_retries,
                backoff_sec=backoff_sec,
                allow_provider_fallback=allow_provider_fallback,
                system_prompt=str(
                    llm_cfg.get("subtasks_system_prompt", DEFAULT_SYSTEM_PROMPT)
                ),
                config=self.config,
            )

            subtasks_path = write_hashed_json_artifact(
                defense_root / "subtasks.json",
                {
                    "subtasks": subtasks,
                    "llm_metadata": subtasks_meta,
                    "prompt_hash": prompt_hash,
                },
                config_hash=self.baseline_config_hash,
                refs={
                    "cfg_stats": str(cfg_stats_path),
                    "prompt_hash": prompt_hash,
                },
            )
            artifact_paths["subtasks"] = str(subtasks_path)

            stage_status["subtasks_ready"] = True
            persist_stage_status()

            # ----- Stage 3: subtask-to-code grounding -----
            current_stage = "grounding"

            links, links_meta = linker(
                llm_client=self.llm_client,
                instance_id=instance_id,
                module_name=self.name,
                module_config_hash=self.baseline_config_hash,
                fidelity_mode=self.fidelity_mode,
                provider=provider,
                model=model,
                problem_statement=prompt,
                subtasks=subtasks,
                candidate_nodes=candidate_nodes,
                artifact_dir=defense_root / "llm" / "grounding",
                temperature=temperature,
                seed=seed,
                max_retries=max_retries,
                backoff_sec=backoff_sec,
                allow_provider_fallback=allow_provider_fallback,
                config=self.config,
            )

            grounding_path = write_hashed_json_artifact(
                defense_root / "grounding.json",
                {
                    "links": links,
                    "llm_metadata": links_meta,
                    "candidate_node_ids": [
                        node.get("node_id", "") for node in candidate_nodes
                    ],
                    "subtask_count": len(subtasks),
                    "candidate_node_count": len(candidate_nodes),
                },
                config_hash=self.baseline_config_hash,
                refs={
                    "cfg_stats": str(cfg_stats_path),
                    "subtasks": str(subtasks_path),
                },
            )
            artifact_paths["grounding"] = str(grounding_path)

            stage_status["grounding_ready"] = True
            persist_stage_status()

            # ----- Stage 4: feature extraction -----
            current_stage = "features"

            node_id_to_snippet = {
                str(node.get("node_id", "")): str(node.get("code_snippet", ""))
                for node in candidate_nodes
                if node.get("node_id")
            }

            # Intentionally no fitted vectorizer here. This first extraction pass
            # should be split into train/test before similarity vectorizer fitting.
            full_row = compute_structural_feature_row(
                subtasks=subtasks,
                links=links,
                node_id_to_snippet=node_id_to_snippet,
                vectorizer=None,
                include_similarity=False,
                mask_tokens=bool(self.config.get("mask_tokens", True)),
            )

            feature_row = {
                col: float(full_row.get(col, 0.0))
                for col in selected_cols
                if col in full_row or col in STRUCTURAL_ONLY_FEATURES
            }

            # Ensure every selected column exists.
            for col in selected_cols:
                feature_row.setdefault(col, 0.0)

            # Risk features are optional extras produced by scoped parsers.
            risk_features = cfg_diagnostics.get("risk_features")
            if isinstance(risk_features, dict):
                for col, val in risk_features.items():
                    try:
                        feature_row.setdefault(col, float(val))
                    except Exception:
                        continue

            features_payload = {
                "mode": mode,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_set_name": canonical_feature_set,
                "selected_columns": selected_cols,
                "feature_vector": feature_row,
                "full_feature_vector": {
                    k: float(v)
                    for k, v in full_row.items()
                    if isinstance(v, (int, float))
                },
                "patch_hash": patch_hash,
                "prompt_hash": prompt_hash,
                "label_hint": label_hint,
            }

            features_path = write_hashed_json_artifact(
                defense_root / "features.json",
                features_payload,
                config_hash=self.baseline_config_hash,
                refs={
                    "cfg_stats": str(cfg_stats_path),
                    "grounding": str(grounding_path),
                },
            )
            artifact_paths["features"] = str(features_path)

            stage_status["features_ready"] = True
            persist_stage_status()

            # ----- Stage 5: compact training example -----
            linked_node_ids = {
                nid
                for link in links
                for nid in link.get("node_ids", [])
                if isinstance(nid, str)
            }
            unmatched_nodes = max(0, len(candidate_nodes) - len(linked_node_ids))
            unmatched_subtasks = sum(1 for link in links if not link.get("node_ids"))

            # Mark this before writing training_example.json so the artifact itself
            # records the final completed state.
            stage_status["training_example_ready"] = True

            training_example = {
                "instance_id": instance_id,
                "prompt_type": prompt_type,
                "label": label_hint,
                "dataset": repo_code.get("dataset", ""),
                "split": repo_code.get("split", ""),
                "repo_id": repo_id,
                "base_commit": base_commit,
                "patch_hash": patch_hash,
                "prompt_hash": prompt_hash,
                "patch_artifact_path": repo_code.get("patch_artifact_path", ""),
                "mode": mode,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_set_name": canonical_feature_set,
                "selected_columns": selected_cols,
                "features": feature_row,
                "counts": {
                    "num_subtasks": len(subtasks),
                    "num_candidate_nodes": len(candidate_nodes),
                    "num_links_total": sum(
                        len(link.get("node_ids", [])) for link in links
                    ),
                    "unmatched_nodes": unmatched_nodes,
                    "unmatched_subtasks": unmatched_subtasks,
                },
                "key_metrics": {
                    "subtask_coverage": float(feature_row.get("subtask_coverage", 0.0)),
                    "node_justification": float(feature_row.get("node_justification", 0.0)),
                    "justification_gap": float(feature_row.get("justification_gap", 0.0)),
                    "unjustified_node_rate": float(
                        feature_row.get("unjustified_node_rate", 0.0)
                    ),
                    "node_link_entropy": float(feature_row.get("node_link_entropy", 0.0)),
                    "link_entropy_over_subtasks": float(
                        feature_row.get("link_entropy_over_subtasks", 0.0)
                    ),
                },
                # Use dict(...) so the JSON gets a stable snapshot, not a live reference.
                "stage_completed": dict(stage_status),
                "parsers": {
                    "prompt_parser": prompt_parser_name,
                    "patch_parser": patch_parser_name,
                    "linker": linker_name,
                },
                "llm_metadata": {
                    "subtasks": {
                        "prompt_hash": subtasks_meta.get("prompt_hash", ""),
                        "response_hash": subtasks_meta.get("response_hash", ""),
                        "cache_hit": bool(subtasks_meta.get("cache_hit", False)),
                        "cache_key": subtasks_meta.get("cache_key", ""),
                        "token_usage": subtasks_meta.get("token_usage", {}),
                        "provider_fallback": bool(
                            subtasks_meta.get("provider_fallback", False)
                        ),
                    },
                    "grounding": {
                        "prompt_hash": links_meta.get("prompt_hash", ""),
                        "response_hash": links_meta.get("response_hash", ""),
                        "cache_hit": bool(links_meta.get("cache_hit", False)),
                        "cache_key": links_meta.get("cache_key", ""),
                        "token_usage": links_meta.get("token_usage", {}),
                        "provider_fallback": bool(
                            links_meta.get("provider_fallback", False)
                        ),
                    },
                },
                "cfg_diagnostics": cfg_diagnostics,
                "artifact_paths": dict(artifact_paths),
            }

            training_example_path = write_hashed_json_artifact(
                defense_root / "training_example.json",
                training_example,
                config_hash=self.baseline_config_hash,
                refs={
                    "features": str(features_path),
                    "grounding": str(grounding_path),
                },
            )
            artifact_paths["training_example"] = str(training_example_path)

            # Now rewrite stage_status.json with the final artifact path included.
            persist_stage_status(
                {
                    "artifact_paths": dict(artifact_paths),
                }
            )

            self.last_signals = {
                "extract_only": True,
                "mode": mode,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_set_name": canonical_feature_set,
                "instance_id": instance_id,
                "prompt_type": prompt_type,
                "label_hint": label_hint,
                "patch_hash": patch_hash,
                "prompt_hash": prompt_hash,
                "parsers": {
                    "prompt_parser": prompt_parser_name,
                    "patch_parser": patch_parser_name,
                    "linker": linker_name,
                },
                "provider": provider,
                "model": model,
                "num_subtasks": len(subtasks),
                "num_candidate_nodes": len(candidate_nodes),
                "num_links_total": sum(len(link.get("node_ids", [])) for link in links),
                "key_metrics": training_example["key_metrics"],
                "stage_completed": stage_status,
                "failure_flags": {
                    "cfg_fail": False,
                    "subtasks_fail": False,
                    "grounding_fail": False,
                    "features_fail": False,
                },
                "subtasks_cache_hit": bool(subtasks_meta.get("cache_hit", False)),
                "subtasks_cache_key": subtasks_meta.get("cache_key", ""),
                "subtasks_token_usage": subtasks_meta.get("token_usage", {}),
                "grounding_cache_hit": bool(links_meta.get("cache_hit", False)),
                "grounding_cache_key": links_meta.get("cache_key", ""),
                "grounding_token_usage": links_meta.get("token_usage", {}),
                "artifact_paths": artifact_paths,
            }

            # True means extraction succeeded, not that the patch is safe.
            return True

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            persist_stage_status(
                {
                    "stage_failed": current_stage,
                    "error": error_text,
                    "artifact_paths": artifact_paths,
                }
            )

            self.last_signals = {
                "extract_only": True,
                "mode": mode,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_set_name": (
                    feature_set_name(mode)
                    if mode in STRUCTURAL_FAMILY_MODES
                    else "unknown"
                ),
                "instance_id": instance_id,
                "prompt_type": prompt_type,
                "label_hint": label_hint,
                "patch_hash": patch_hash,
                "prompt_hash": prompt_hash,
                "error": error_text,
                "stage_failed": current_stage,
                "stage_completed": stage_status,
                "failure_flags": self._failure_flags(current_stage),
                "artifact_paths": artifact_paths,
            }

            return False


register_baseline("structural_misalignment_features")(StructuralMisalignmentFeatureDefense)
