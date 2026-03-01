"""Structural misalignment defense baseline (CFG diff + grounding + ML inference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.baseline.base import BaseDefense
from src.baseline.registry import register_baseline
from src.common.artifacts import write_hashed_json_artifact
from src.common.artifact_store import atomic_write_json
from src.baseline.structural_misalignment.cfg.stats import compute_cfg_stats
from src.baseline.structural_misalignment.features.schema import (
    FEATURE_SCHEMA_VERSION,
    STRUCTURAL_FAMILY_MODES,
    UNIVERSAL_FAMILY_MODES,
    feature_set_name,
    normalize_mode,
)
from src.baseline.structural_misalignment.features.structural_features import (
    SIMILARITY_ONLY_FEATURES,
    STRUCTURAL_COMBINED_FEATURES,
    STRUCTURAL_ONLY_FEATURES,
    compute_structural_feature_row,
)
from src.baseline.structural_misalignment.features.universal_features import (
    columns_for_universal_mode,
    extract_universal_features,
    select_feature_subset,
)
from src.baseline.structural_misalignment.grounding.subtasks import DEFAULT_SYSTEM_PROMPT
from src.common.hashing import sha256_text
from src.baseline.structural_misalignment.models.infer import (
    decide_from_policy,
    predict_reject_score,
)
from src.baseline.structural_misalignment.models.load import load_model_bundle
from src.baseline.structural_misalignment.security.severity import analyze_cfg_diff_nodes

# Import parser registries and default implementations (triggers registration)
from src.baseline.structural_misalignment.parsers.registry import (
    get_linker,
    get_patch_parser,
    get_prompt_parser,
)

# Import default parsers to register them
import src.baseline.structural_misalignment.parsers.prompt.llm_subtasks  # noqa: F401
import src.baseline.structural_misalignment.parsers.patch.cfg_ast  # noqa: F401
import src.baseline.structural_misalignment.parsers.patch.llm_chunks  # noqa: F401
import src.baseline.structural_misalignment.parsers.linking.llm_grounding  # noqa: F401
import src.baseline.structural_misalignment.parsers.linking.embedding_similarity  # noqa: F401


class StructuralMisalignmentDefense(BaseDefense):
    name = "structural_misalignment"

    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ):
        _ = all_tests
        mode = str(self.config.get("mode", "structural_combined")).strip()
        try:
            mode = normalize_mode(mode)
        except Exception as exc:
            self.last_signals = {
                "mode": "unknown",
                "decision_policy": str(self.config.get("decision_policy", "reject_if_score_ge_threshold")).strip(),
                "threshold": float(self.config.get("threshold", 0.5)),
                "failure_flags": {
                    "cfg_fail": False,
                    "subtasks_fail": False,
                    "grounding_fail": False,
                    "features_fail": False,
                    "model_missing": False,
                    "inference_fail": False,
                },
                "error": f"Invalid mode: {exc}",
            }
            return False

        decision_policy = str(self.config.get("decision_policy", "reject_if_score_ge_threshold")).strip()
        threshold = float(self.config.get("threshold", 0.5))

        # Load parser configurations (with defaults)
        parser_config = self.config.get("parsers", {})
        if not isinstance(parser_config, dict):
            parser_config = {}
        prompt_parser_name = str(parser_config.get("prompt", "llm_subtasks")).strip()
        patch_parser_name = str(parser_config.get("patch", "cfg_ast")).strip()
        linker_name = str(parser_config.get("linking", "llm_grounding")).strip()

        # Get parser implementations from registries
        try:
            prompt_parser = get_prompt_parser(prompt_parser_name)
        except KeyError as exc:
            self.last_signals = {
                "mode": mode,
                "decision_policy": decision_policy,
                "threshold": threshold,
                "failure_flags": {
                    "cfg_fail": False,
                    "subtasks_fail": True,
                    "grounding_fail": False,
                    "features_fail": False,
                    "model_missing": False,
                    "inference_fail": False,
                },
                "error": f"Unknown prompt parser: {exc}",
            }
            return False

        try:
            patch_parser = get_patch_parser(patch_parser_name)
        except KeyError as exc:
            self.last_signals = {
                "mode": mode,
                "decision_policy": decision_policy,
                "threshold": threshold,
                "failure_flags": {
                    "cfg_fail": True,
                    "subtasks_fail": False,
                    "grounding_fail": False,
                    "features_fail": False,
                    "model_missing": False,
                    "inference_fail": False,
                },
                "error": f"Unknown patch parser: {exc}",
            }
            return False

        try:
            linker = get_linker(linker_name)
        except KeyError as exc:
            self.last_signals = {
                "mode": mode,
                "decision_policy": decision_policy,
                "threshold": threshold,
                "failure_flags": {
                    "cfg_fail": False,
                    "subtasks_fail": False,
                    "grounding_fail": True,
                    "features_fail": False,
                    "model_missing": False,
                    "inference_fail": False,
                },
                "error": f"Unknown linker: {exc}",
            }
            return False

        model_path = ""
        path_map = self.config.get("model_paths")
        if isinstance(path_map, dict):
            if mode in path_map:
                model_path = str(path_map.get(mode, "")).strip()
        if not model_path:
            model_path = str(self.config.get("model_path", "")).strip()

        severity_mode = str(self.config.get("severity_mode", "universal")).strip().lower()
        if not model_path:
            self.last_signals = {
                "mode": mode,
                "decision_policy": decision_policy,
                "threshold": threshold,
                "failure_flags": {
                    "cfg_fail": False,
                    "subtasks_fail": False,
                    "grounding_fail": False,
                    "features_fail": False,
                    "model_missing": True,
                    "inference_fail": False,
                },
                "error": "Missing required config: model_path",
            }
            return False

        instance_id = str(repo_code.get("instance_id", "unknown"))
        dataset_name = str(repo_code.get("dataset", "")).strip().lower()
        defense_root = self.run_root / "artifacts" / "defenses" / instance_id / self.name
        defense_root.mkdir(parents=True, exist_ok=True)

        artifact_paths: Dict[str, str] = {}
        stage_status = {
            "cfg_ready": False,
            "subtasks_ready": False,
            "grounding_ready": False,
            "features_ready": False,
            "model_loaded": False,
            "inference_done": False,
        }
        stage_status_path = defense_root / "stage_status.json"
        artifact_paths["stage_status"] = str(stage_status_path)
        current_stage = "init"

        def persist_stage_status() -> None:
            atomic_write_json(
                stage_status_path,
                {
                    "stage_completed": stage_status,
                    "mode": mode,
                    "config_hash": self.baseline_config_hash,
                },
            )

        persist_stage_status()
        try:
            if mode not in STRUCTURAL_FAMILY_MODES and mode not in UNIVERSAL_FAMILY_MODES:
                raise ValueError(f"Unsupported structural_misalignment mode: {mode}")
            if severity_mode in {"markers", "dataset_markers", "hybrid"} and not bool(
                self.config.get("enable_marker_debug_mode", False)
            ):
                raise ValueError(
                    "severity_mode with dataset markers requires enable_marker_debug_mode=true"
                )

            patch_text = str(code_or_patch or "")
            if not patch_text.strip():
                raise ValueError("Defense received empty patch")

            allow_hunk_fallback = bool(self.config.get("allow_hunk_fallback", True))
            if dataset_name and dataset_name != "toy":
                allow_hunk_fallback = bool(
                    self.config.get("allow_hunk_fallback_non_toy", allow_hunk_fallback)
                )

            current_stage = "cfg"
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
                    f"CFG fallback disallowed but required: {cfg_diagnostics.get('fallback_reason', 'unknown')}"
                )

            cfg_stats = compute_cfg_stats(candidate_nodes)
            cfg_payload = {
                "cfg_stats": cfg_stats,
                "cfg_diff_summary": cfg_diff.get("summary", {}),
                "cfg_diagnostics": cfg_diagnostics,
                "candidate_node_count": len(candidate_nodes),
            }
            cfg_stats_path = write_hashed_json_artifact(
                defense_root / "cfg_stats.json",
                cfg_payload,
                config_hash=self.baseline_config_hash,
                refs={
                    "instance_id": instance_id,
                    "patch_hash": sha256_text(patch_text),
                },
            )
            artifact_paths["cfg_stats"] = str(cfg_stats_path)
            stage_status["cfg_ready"] = True
            persist_stage_status()

            llm_cfg = self.config.get("llm") if isinstance(self.config.get("llm"), dict) else {}
            provider = str(llm_cfg.get("provider", self.config.get("provider", "openai")))
            model = str(llm_cfg.get("model", self.config.get("model", "gpt-4o-mini")))
            temperature = float(llm_cfg.get("temperature", self.config.get("temperature", 0.2)))
            seed = llm_cfg.get("seed", self.config.get("seed"))
            max_retries = int(llm_cfg.get("max_retries", self.config.get("max_retries", 2)))
            backoff_sec = float(llm_cfg.get("backoff_sec", self.config.get("backoff_sec", 1.0)))
            allow_provider_fallback = bool(llm_cfg.get("allow_provider_fallback", self.config.get("allow_provider_fallback", False)))
            if dataset_name == "toy" and bool(self.config.get("allow_provider_fallback_on_toy", True)):
                allow_provider_fallback = True

            current_stage = "subtasks"
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
                system_prompt=str(llm_cfg.get("subtasks_system_prompt", DEFAULT_SYSTEM_PROMPT)),
                config=self.config,
            )
            subtasks_path = write_hashed_json_artifact(
                defense_root / "subtasks.json",
                {
                    "subtasks": subtasks,
                    "llm_metadata": subtasks_meta,
                },
                config_hash=self.baseline_config_hash,
                refs={"cfg_stats": str(cfg_stats_path)},
            )
            artifact_paths["subtasks"] = str(subtasks_path)
            stage_status["subtasks_ready"] = True
            persist_stage_status()

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
                    "candidate_node_ids": [node.get("node_id", "") for node in candidate_nodes],
                },
                config_hash=self.baseline_config_hash,
                refs={
                    "cfg_stats": str(cfg_stats_path),
                    "subtasks": str(subtasks_path),
                },
            )
            artifact_paths["grounding"] = str(grounding_path)

            include_similarity = mode in {"similarity_only", "structural_combined"}
            require_vectorizer = include_similarity
            current_stage = "model_load"
            bundle = load_model_bundle(model_path, require_vectorizer=require_vectorizer)
            stage_status["model_loaded"] = True
            persist_stage_status()

            feature_row: Dict[str, float]
            severity_payload: Dict[str, Any] = {}
            analyzed_nodes = candidate_nodes
            stage_status["grounding_ready"] = True
            persist_stage_status()

            current_stage = "features"
            if mode in STRUCTURAL_FAMILY_MODES:
                node_id_to_snippet = {
                    str(node.get("node_id", "")): str(node.get("code_snippet", ""))
                    for node in candidate_nodes
                    if node.get("node_id")
                }
                full_row = compute_structural_feature_row(
                    subtasks=subtasks,
                    links=links,
                    node_id_to_snippet=node_id_to_snippet,
                    vectorizer=bundle.vectorizer,
                    include_similarity=include_similarity,
                    mask_tokens=bool(self.config.get("mask_tokens", True)),
                )
                if mode == "structural_only":
                    selected_cols = STRUCTURAL_ONLY_FEATURES
                elif mode == "similarity_only":
                    selected_cols = SIMILARITY_ONLY_FEATURES
                else:
                    selected_cols = STRUCTURAL_COMBINED_FEATURES
                feature_row = {col: float(full_row.get(col, 0.0)) for col in selected_cols}
            else:
                analyzed = analyze_cfg_diff_nodes(
                    candidate_nodes,
                    severity_mode=severity_mode,
                    comment_stripping_mode=str(self.config.get("comment_stripping_mode", "python_tokenize")),
                )
                analyzed_nodes = analyzed["nodes"]
                universal_full = extract_universal_features(
                    {
                        "nodes": analyzed_nodes,
                        "links": links,
                        "subtasks": subtasks,
                    },
                    mode="adv_subtask",
                )
                selected_cols = columns_for_universal_mode(mode)
                feature_row, resolved_cols = select_feature_subset(universal_full, selected_cols)
                selected_cols = resolved_cols
                severity_payload = {
                    "summary": analyzed.get("summary", {}),
                    "severity_mode": severity_mode,
                    "marker_debug_enabled": bool(self.config.get("enable_marker_debug_mode", False)),
                    "universal_features_all": universal_full,
                }

            canonical_feature_set = feature_set_name(mode)
            features_payload = {
                "mode": mode,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_set_name": canonical_feature_set,
                "selected_columns": selected_cols,
                "feature_vector": feature_row,
                "model_feature_list": bundle.feature_list,
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

            if mode in UNIVERSAL_FAMILY_MODES:
                severity_path = write_hashed_json_artifact(
                    defense_root / "severity.json",
                    severity_payload,
                    config_hash=self.baseline_config_hash,
                    refs={
                        "grounding": str(grounding_path),
                        "features": str(features_path),
                    },
                )
                artifact_paths["severity"] = str(severity_path)

            current_stage = "inference"
            inference = predict_reject_score(bundle, feature_row)
            accepted = decide_from_policy(inference.score, threshold, decision_policy)
            stage_status["inference_done"] = True
            persist_stage_status()

            coverage = float(feature_row.get("subtask_coverage", 0.0))
            justification = float(feature_row.get("node_justification", 0.0))
            link_entropy = float(
                feature_row.get(
                    "node_link_entropy",
                    feature_row.get("link_entropy_over_subtasks", 0.0),
                )
            )
            unmatched_nodes = max(
                0,
                int(round(float(feature_row.get("num_candidate_nodes", len(candidate_nodes)))))
                - len({nid for link in links for nid in link.get("node_ids", [])}),
            )
            unmatched_subtasks = sum(1 for link in links if not link.get("node_ids"))

            model_output_path = write_hashed_json_artifact(
                defense_root / "model_output.json",
                {
                    "score": float(inference.score),
                    "threshold": threshold,
                    "decision_policy": decision_policy,
                    "accepted": bool(accepted),
                    "mode": mode,
                    "feature_set_name": canonical_feature_set,
                    "missing_feature_columns_filled_zero": inference.missing_columns_filled_zero,
                    "model_metadata": bundle.metadata,
                    "model_dir": str(bundle.model_dir),
                },
                config_hash=self.baseline_config_hash,
                refs={
                    "features": str(features_path),
                },
            )
            artifact_paths["model_output"] = str(model_output_path)

            self.last_signals = {
                "mode": mode,
                "decision_policy": decision_policy,
                "model_score": float(inference.score),
                "threshold": threshold,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_set_name": canonical_feature_set,
                "eval_only_no_refit_guard": True,
                "missing_feature_columns_filled_zero": inference.missing_columns_filled_zero,
                "severity_mode": severity_mode,
                "model_path": str(model_path),
                "provider": provider,
                "model": model,
                "subtasks_prompt_hash": subtasks_meta.get("prompt_hash", ""),
                "subtasks_response_hash": subtasks_meta.get("response_hash", ""),
                "subtasks_cache_hit": bool(subtasks_meta.get("cache_hit", False)),
                "subtasks_cache_key": subtasks_meta.get("cache_key", ""),
                "subtasks_token_usage": subtasks_meta.get("token_usage", {}),
                "grounding_prompt_hash": links_meta.get("prompt_hash", ""),
                "grounding_response_hash": links_meta.get("response_hash", ""),
                "grounding_cache_hit": bool(links_meta.get("cache_hit", False)),
                "grounding_cache_key": links_meta.get("cache_key", ""),
                "grounding_token_usage": links_meta.get("token_usage", {}),
                "key_metrics": {
                    "coverage": coverage,
                    "justification_gap": coverage - justification,
                    "entropy": link_entropy,
                    "unmatched_nodes": unmatched_nodes,
                    "unmatched_subtasks": unmatched_subtasks,
                },
                "stage_completed": stage_status,
                "failure_flags": {
                    "cfg_fail": False,
                    "subtasks_fail": False,
                    "grounding_fail": False,
                    "features_fail": False,
                    "model_missing": False,
                    "inference_fail": False,
                },
                "artifact_paths": artifact_paths,
            }
            return True if accepted else False

        except Exception as exc:
            failure_flags = {
                "cfg_fail": current_stage == "cfg",
                "subtasks_fail": current_stage == "subtasks",
                "grounding_fail": current_stage == "grounding",
                "features_fail": current_stage == "features",
                "model_missing": current_stage == "model_load",
                "inference_fail": current_stage == "inference",
            }
            error_text = f"{type(exc).__name__}: {exc}"
            lowered_error = error_text.lower()
            if "model path not found" in lowered_error or "model bundle incomplete" in lowered_error:
                failure_flags["model_missing"] = True
            persist_stage_status()
            self.last_signals = {
                "mode": mode,
                "decision_policy": decision_policy,
                "threshold": threshold,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_set_name": feature_set_name(mode)
                if mode in STRUCTURAL_FAMILY_MODES.union(UNIVERSAL_FAMILY_MODES)
                else "unknown",
                "error": error_text,
                "stage_failed": current_stage,
                "stage_completed": stage_status,
                "failure_flags": failure_flags,
                "artifact_paths": artifact_paths,
            }
            return False


register_baseline("structural_misalignment")(StructuralMisalignmentDefense)
