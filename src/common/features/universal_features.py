"""Universal feature extraction (structural + optional universal severity)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

from src.common.features.schema import normalize_structural_mode

# Stable feature order from old pipeline contract.
FEATURE_COLUMNS = [
    "num_subtasks",
    "num_candidate_nodes",
    "num_links_total",
    "num_suspicious_nodes",
    "frac_suspicious_nodes",
    "subtask_coverage",
    "node_justification",
    "avg_nodes_per_subtask",
    "avg_subtasks_per_node",
    "suspicious_acceptance",
    "suspicious_unlinked",
    "suspicious_misassignment",
    "max_severity",
    "mean_severity",
    "top3_severity_sum",
    "max_severity_universal",
    "mean_severity_universal",
    "top3_severity_sum_universal",
    "num_suspicious_nodes_universal",
    "frac_suspicious_nodes_universal",
    "cnt_code_execution",
    "cnt_command_execution",
    "cnt_shell_injection",
    "cnt_network_io",
    "cnt_deserialization",
    "cnt_file_access",
    "cnt_adversarial_marker",
    "link_entropy_over_subtasks",
    "max_subtask_link_share",
]

SECURITY_FEATURE_PREFIXES = ["suspicious_", "cnt_"]
SECURITY_FEATURE_NAMES = [
    "max_severity",
    "mean_severity",
    "top3_severity_sum",
    "num_suspicious_nodes",
    "frac_suspicious_nodes",
    "max_severity_universal",
    "mean_severity_universal",
    "top3_severity_sum_universal",
    "num_suspicious_nodes_universal",
    "frac_suspicious_nodes_universal",
]

SEVERITY_ONLY_FEATURES_UNIVERSAL = [
    "max_severity_universal",
    "mean_severity_universal",
    "top3_severity_sum_universal",
    "num_suspicious_nodes_universal",
    "frac_suspicious_nodes_universal",
]


def compute_link_entropy(links_per_subtask: List[int]) -> float:
    total = sum(links_per_subtask)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in links_per_subtask:
        if count <= 0:
            continue
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy


def compute_suspicious_misassignment(
    links: List[Dict[str, Any]],
    suspicious_node_ids: set[str],
    num_subtasks: int,
    mode: str,
) -> float:
    # Old behavior: only defined in adv_subtask mode.
    if mode != "adv_subtask" or num_subtasks < 2:
        return 0.0
    if not suspicious_node_ids:
        return 0.0
    adv_subtask_idx = num_subtasks - 1
    suspicious_to_non_adv = set()
    for idx, link in enumerate(links):
        if idx == adv_subtask_idx:
            continue
        suspicious_to_non_adv.update(set(link.get("node_ids", [])) & suspicious_node_ids)
    return len(suspicious_to_non_adv) / len(suspicious_node_ids)


def count_suspicious_categories(nodes: List[Dict[str, Any]]) -> Dict[str, float]:
    mapping = {
        "code_execution": "cnt_code_execution",
        "command_execution": "cnt_command_execution",
        "shell_injection": "cnt_shell_injection",
        "network": "cnt_network_io",
        "deserialization": "cnt_deserialization",
        "file_access": "cnt_file_access",
        "file_deletion": "cnt_file_access",
        "adversarial_marker": "cnt_adversarial_marker",
    }
    counts = {
        "cnt_code_execution": 0,
        "cnt_command_execution": 0,
        "cnt_shell_injection": 0,
        "cnt_network_io": 0,
        "cnt_deserialization": 0,
        "cnt_file_access": 0,
        "cnt_adversarial_marker": 0,
    }

    for node in nodes:
        patterns = node.get("suspicious_patterns", [])
        for pattern in patterns:
            category = ""
            if isinstance(pattern, (list, tuple)) and len(pattern) >= 2:
                category = str(pattern[1])
            elif isinstance(pattern, str):
                category = pattern
            feature = mapping.get(category)
            if feature:
                counts[feature] += 1

    return {name: float(value) for name, value in counts.items()}


def extract_universal_features(record: Dict[str, Any], mode: str) -> Dict[str, float]:
    nodes = record.get("nodes", [])
    links = record.get("links", [])
    subtasks = record.get("subtasks", [])

    num_subtasks = len(subtasks)
    num_nodes = len(nodes)

    suspicious_nodes = [node for node in nodes if node.get("is_suspicious", False)]
    suspicious_ids = {str(node.get("node_id", "")) for node in suspicious_nodes if node.get("node_id")}
    suspicious_nodes_universal = [
        node for node in nodes if node.get("is_suspicious_universal", node.get("is_suspicious", False))
    ]

    all_linked = set()
    links_per_subtask: List[int] = []
    for link in links:
        ids = [str(node_id) for node_id in link.get("node_ids", [])]
        all_linked.update(ids)
        links_per_subtask.append(len(ids))

    num_links_total = sum(links_per_subtask)
    linked_subtasks = sum(1 for link in links if link.get("node_ids"))
    linked_nodes = len(all_linked)

    features: Dict[str, float] = {
        "num_subtasks": float(num_subtasks),
        "num_candidate_nodes": float(num_nodes),
        "num_links_total": float(num_links_total),
        "num_suspicious_nodes": float(len(suspicious_nodes)),
        "frac_suspicious_nodes": len(suspicious_nodes) / max(1, num_nodes),
        "num_suspicious_nodes_universal": float(len(suspicious_nodes_universal)),
        "frac_suspicious_nodes_universal": len(suspicious_nodes_universal) / max(1, num_nodes),
        "subtask_coverage": linked_subtasks / max(1, num_subtasks),
        "node_justification": linked_nodes / max(1, num_nodes),
        "avg_nodes_per_subtask": num_links_total / max(1, num_subtasks),
    }

    node_to_subtask = Counter()
    for link in links:
        for node_id in link.get("node_ids", []):
            node_to_subtask[str(node_id)] += 1

    features["avg_subtasks_per_node"] = (
        sum(node_to_subtask.values()) / linked_nodes if linked_nodes > 0 else 0.0
    )

    suspicious_linked = suspicious_ids & all_linked
    suspicious_unlinked = suspicious_ids - all_linked
    features["suspicious_acceptance"] = len(suspicious_linked) / max(1, len(suspicious_ids))
    features["suspicious_unlinked"] = len(suspicious_unlinked) / max(1, len(suspicious_ids))
    features["suspicious_misassignment"] = compute_suspicious_misassignment(
        links,
        suspicious_ids,
        num_subtasks,
        mode,
    )

    sev = [float(node.get("severity_score", 0.0) or 0.0) for node in suspicious_nodes]
    if sev:
        features["max_severity"] = max(sev)
        features["mean_severity"] = sum(sev) / len(sev)
        features["top3_severity_sum"] = sum(sorted(sev, reverse=True)[:3])
    else:
        features["max_severity"] = 0.0
        features["mean_severity"] = 0.0
        features["top3_severity_sum"] = 0.0

    sev_u: List[float] = []
    for node in suspicious_nodes_universal:
        raw = node.get("severity_score_universal", node.get("severity_score", 0.0))
        try:
            sev_u.append(float(raw))
        except (TypeError, ValueError):
            sev_u.append(0.0)

    if sev_u:
        features["max_severity_universal"] = max(sev_u)
        features["mean_severity_universal"] = sum(sev_u) / len(sev_u)
        features["top3_severity_sum_universal"] = sum(sorted(sev_u, reverse=True)[:3])
    else:
        features["max_severity_universal"] = 0.0
        features["mean_severity_universal"] = 0.0
        features["top3_severity_sum_universal"] = 0.0

    features.update(count_suspicious_categories(nodes))
    features["link_entropy_over_subtasks"] = compute_link_entropy(links_per_subtask)
    features["max_subtask_link_share"] = (
        max(links_per_subtask) / max(1, num_links_total) if links_per_subtask else 0.0
    )

    # Fill any missing expected columns with 0.0 for stable schema.
    for col in FEATURE_COLUMNS:
        if col not in features:
            features[col] = 0.0

    return {name: float(features[name]) for name in FEATURE_COLUMNS}


def is_security_feature(col: str) -> bool:
    if col in SECURITY_FEATURE_NAMES:
        return True
    return any(col.startswith(prefix) for prefix in SECURITY_FEATURE_PREFIXES)


def columns_for_universal_mode(mode: str) -> List[str]:
    canonical_mode = normalize_structural_mode(mode)
    if canonical_mode == "severity_only_universal":
        return SEVERITY_ONLY_FEATURES_UNIVERSAL.copy()
    if canonical_mode == "no_security":
        return [col for col in FEATURE_COLUMNS if not is_security_feature(col)]
    if canonical_mode == "full_universal":
        return FEATURE_COLUMNS.copy()
    raise ValueError(f"Unsupported universal mode: {mode}")


def select_feature_subset(feature_row: Dict[str, float], columns: List[str]) -> Tuple[Dict[str, float], List[str]]:
    resolved: List[str] = []
    selected: Dict[str, float] = {}
    for col in columns:
        resolved.append(col)
        selected[col] = float(feature_row.get(col, 0.0))
    return selected, resolved


# Backward-compatible aliases for legacy imports.
extract_task7_features = extract_universal_features
columns_for_task7_mode = columns_for_universal_mode
