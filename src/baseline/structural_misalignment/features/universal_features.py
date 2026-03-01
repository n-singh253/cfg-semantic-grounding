"""Universal feature extraction (structural + optional universal severity)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

from src.baseline.structural_misalignment.features.schema import normalize_mode

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
    # NEW IN V2: ADDITIONAL 18 FEATURES BELOW
    # Link distribution statistics
    "max_links_per_subtask",
    "min_links_per_subtask",
    "std_links_per_subtask",
    "max_links_per_node",
    "min_links_per_node",
    "std_links_per_node",
    # Density and coverage
    "link_density",
    # Pattern features
    "diagonal_links",
    "diagonal_ratio",
    "one_to_one_subtasks",
    "one_to_many_subtasks",
    "one_to_one_nodes",
    "many_to_one_nodes",
    # Ratios
    "subtask_node_ratio",
    # Distribution inequality
    "gini_subtasks",
    "gini_nodes",
    # Connectivity features
    "is_connected",
    "all_min_2_connections",
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


def compute_gini_coefficient(values: List[int]) -> float:
    """Calculate Gini coefficient for link distribution."""
    if not values or sum(values) == 0:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    cumsum = 0.0
    for i, val in enumerate(sorted_values):
        cumsum += (2 * (i + 1) - n - 1) * val
    return cumsum / (n * sum(sorted_values))


def compute_std(values: List[int]) -> float:
    """Calculate standard deviation."""
    if not values or len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def check_connectivity(links: List[Dict[str, Any]], num_subtasks: int, num_nodes: int) -> bool:
    """Check if the bipartite graph is connected using BFS."""
    if num_subtasks == 0 or num_nodes == 0:
        return False
    
    # Build adjacency list
    graph: Dict[str, List[str]] = {}
    for idx, link in enumerate(links):
        subtask_key = f"s_{idx}"
        if subtask_key not in graph:
            graph[subtask_key] = []
        for node_id in link.get("node_ids", []):
            node_key = f"n_{node_id}"
            if node_key not in graph:
                graph[node_key] = []
            graph[subtask_key].append(node_key)
            graph[node_key].append(subtask_key)
    
    if not graph:
        return False
    
    # BFS from first node
    visited = set()
    queue = [next(iter(graph.keys()))]
    visited.add(queue[0])
    
    while queue:
        current = queue.pop(0)
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    # Check if all nodes with links are visited
    return len(visited) == len(graph)


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

    # === New Features: Link Distribution Statistics ===
    # Per subtask statistics
    if links_per_subtask:
        features["max_links_per_subtask"] = float(max(links_per_subtask))
        features["min_links_per_subtask"] = float(min(links_per_subtask))
        features["std_links_per_subtask"] = compute_std(links_per_subtask)
    else:
        features["max_links_per_subtask"] = 0.0
        features["min_links_per_subtask"] = 0.0
        features["std_links_per_subtask"] = 0.0
    
    # Per node statistics
    links_per_node = list(node_to_subtask.values()) if node_to_subtask else []
    if links_per_node:
        features["max_links_per_node"] = float(max(links_per_node))
        features["min_links_per_node"] = float(min(links_per_node))
        features["std_links_per_node"] = compute_std(links_per_node)
    else:
        features["max_links_per_node"] = 0.0
        features["min_links_per_node"] = 0.0
        features["std_links_per_node"] = 0.0
    
    # === New Features: Density and Coverage ===
    features["link_density"] = (
        num_links_total / max(1, linked_subtasks * linked_nodes) if linked_subtasks and linked_nodes else 0.0
    )
    
    # === New Features: Pattern Features ===
    # Diagonal links (where subtask index equals node index, if applicable)
    diagonal_count = 0
    for idx, link in enumerate(links):
        node_ids = link.get("node_ids", [])
        # Check if any node_id contains the subtask index pattern
        for node_id in node_ids:
            node_str = str(node_id)
            # Try to extract numeric suffix from node_id
            if f"::n{idx}" in node_str or node_str.endswith(f"_{idx}"):
                diagonal_count += 1
                break
    
    features["diagonal_links"] = float(diagonal_count)
    features["diagonal_ratio"] = diagonal_count / max(1, len(links)) if links else 0.0
    
    # One-to-one and one-to-many patterns
    subtask_link_counts = [len(link.get("node_ids", [])) for link in links]
    features["one_to_one_subtasks"] = float(sum(1 for count in subtask_link_counts if count == 1))
    features["one_to_many_subtasks"] = float(sum(1 for count in subtask_link_counts if count > 1))
    
    node_link_counts = list(node_to_subtask.values())
    features["one_to_one_nodes"] = float(sum(1 for count in node_link_counts if count == 1))
    features["many_to_one_nodes"] = float(sum(1 for count in node_link_counts if count > 1))
    
    # === New Features: Ratios ===
    features["subtask_node_ratio"] = linked_subtasks / max(1, linked_nodes) if linked_nodes else 0.0
    
    # === New Features: Distribution Inequality (Gini Coefficients) ===
    features["gini_subtasks"] = compute_gini_coefficient(links_per_subtask)
    features["gini_nodes"] = compute_gini_coefficient(links_per_node)
    
    # === New Features: Connectivity ===
    features["is_connected"] = float(check_connectivity(links, num_subtasks, num_nodes))
    
    # Check if all nodes have at least 2 connections
    min_connections = 2
    all_have_min = True
    if linked_subtasks < num_subtasks or linked_nodes < num_nodes:
        all_have_min = False  # Some nodes have 0 connections
    elif subtask_link_counts:
        if any(count < min_connections for count in subtask_link_counts if count > 0):
            all_have_min = False
    if node_link_counts and all_have_min:
        if any(count < min_connections for count in node_link_counts):
            all_have_min = False
    
    features["all_min_2_connections"] = float(all_have_min)

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
    canonical_mode = normalize_mode(mode)
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
