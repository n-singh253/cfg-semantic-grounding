"""Task8 misalignment features (structural + optional similarity)."""

from __future__ import annotations

import io
import json
import math
import re
import tokenize
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_FEATURES = [
    "link_sim_mean",
    "link_sim_median",
    "link_sim_p10",
    "link_sim_min",
    "link_sim_frac_below_tau_0_2",
    "link_sim_frac_below_tau_0_3",
]

STRUCTURAL_FEATURES = [
    "subtask_coverage",
    "node_justification",
    "justification_gap",
    "unjustified_node_rate",
    "avg_nodes_per_subtask",
    "avg_subtasks_per_node",
    "max_subtask_link_share",
    "node_link_entropy",
    "link_entropy_over_subtasks",
]

TASK8_STRUCTURE_ONLY_FEATURES = [
    "subtask_coverage",
    "node_justification",
    "justification_gap",
    "unjustified_node_rate",
    "avg_nodes_per_subtask",
    "avg_subtasks_per_node",
    "max_subtask_link_share",
    "node_link_entropy",
    "num_subtasks",
    "num_candidate_nodes",
    "num_links_total",
    "link_entropy_over_subtasks",
]
TASK8_SIMILARITY_ONLY_FEATURES = SIMILARITY_FEATURES.copy()
TASK8_COMBINED_FEATURES = TASK8_STRUCTURE_ONLY_FEATURES + TASK8_SIMILARITY_ONLY_FEATURES


def validate_links_schema(links: List[Dict[str, Any]]) -> None:
    if not isinstance(links, list):
        raise ValueError(f"links must be list, got {type(links)}")
    for idx, item in enumerate(links):
        if not isinstance(item, dict):
            raise ValueError(f"links[{idx}] must be dict")
        if "subtask_index" not in item or not isinstance(item["subtask_index"], int):
            raise ValueError(f"links[{idx}] missing/invalid subtask_index")
        node_ids = item.get("node_ids")
        if not isinstance(node_ids, list):
            raise ValueError(f"links[{idx}] node_ids must be list")
        for nidx, node_id in enumerate(node_ids):
            if not isinstance(node_id, str):
                raise ValueError(f"links[{idx}].node_ids[{nidx}] must be str")


def parse_json_column(value: str, column_name: str) -> Any:
    if not isinstance(value, str):
        raise ValueError(f"{column_name} must be JSON string")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{column_name} has invalid JSON: {exc}") from exc


def mask_code_tokens(code: str) -> str:
    if not code or not code.strip():
        return code
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except Exception:
        return _fallback_mask(code)

    import keyword

    keywords = set(keyword.kwlist)
    builtins = {
        "True",
        "False",
        "None",
        "self",
        "cls",
        "print",
        "len",
        "range",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "bool",
        "open",
        "type",
        "isinstance",
        "hasattr",
        "getattr",
        "setattr",
        "super",
        "property",
        "staticmethod",
        "classmethod",
    }

    out: List[str] = []
    prev_end = (1, 0)
    for tok_type, tok_string, tok_start, tok_end, _ in tokens:
        if tok_start[0] > prev_end[0]:
            out.append("\n" * (tok_start[0] - prev_end[0]))
            out.append(" " * tok_start[1])
        elif tok_start[1] > prev_end[1]:
            out.append(" " * (tok_start[1] - prev_end[1]))

        if tok_type == tokenize.NAME:
            out.append(tok_string if tok_string in keywords or tok_string in builtins else "VAR")
        elif tok_type == tokenize.STRING:
            out.append("STR")
        elif tok_type == tokenize.NUMBER:
            out.append("NUM")
        elif tok_type != tokenize.ENDMARKER:
            out.append(tok_string)
        prev_end = tok_end

    return "".join(out)


def _fallback_mask(code: str) -> str:
    code = re.sub(r'""".*?"""', "STR", code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", "STR", code, flags=re.DOTALL)
    code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', "STR", code)
    code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "STR", code)
    code = re.sub(r"\b\d+\.?\d*\b", "NUM", code)
    return code


def _compute_entropy(counts: List[int]) -> float:
    if not counts:
        return 0.0
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy


def compute_structural_features(
    subtasks: List[str],
    node_id_to_snippet: Dict[str, str],
    links: List[Dict[str, Any]],
) -> Dict[str, float]:
    validate_links_schema(links)
    num_subtasks = len(subtasks)
    num_nodes = len(node_id_to_snippet)
    if num_subtasks == 0:
        return {name: 0.0 for name in STRUCTURAL_FEATURES}

    link_counts: List[int] = []
    all_linked = set()
    node_to_subtasks = Counter()
    for item in links:
        node_ids = item.get("node_ids", [])
        link_counts.append(len(node_ids))
        all_linked.update(node_ids)
        for node_id in node_ids:
            node_to_subtasks[node_id] += 1

    while len(link_counts) < num_subtasks:
        link_counts.append(0)

    total_links = sum(link_counts)
    linked_subtasks = sum(1 for c in link_counts if c > 0)
    linked_nodes = len(all_linked)

    subtask_coverage = linked_subtasks / max(1, num_subtasks)
    node_justification = linked_nodes / max(1, num_nodes)
    justification_gap = subtask_coverage - node_justification
    unjustified_node_rate = 1.0 - node_justification

    avg_nodes_per_subtask = total_links / max(1, num_subtasks)
    avg_subtasks_per_node = (
        sum(node_to_subtasks.values()) / max(1, linked_nodes) if linked_nodes > 0 else 0.0
    )
    max_share = max(link_counts) / max(1, total_links) if total_links > 0 else 0.0

    return {
        "subtask_coverage": float(subtask_coverage),
        "node_justification": float(node_justification),
        "justification_gap": float(justification_gap),
        "unjustified_node_rate": float(unjustified_node_rate),
        "avg_nodes_per_subtask": float(avg_nodes_per_subtask),
        "avg_subtasks_per_node": float(avg_subtasks_per_node),
        "max_subtask_link_share": float(max_share),
        "node_link_entropy": float(_compute_entropy(list(node_to_subtasks.values()))),
        "link_entropy_over_subtasks": float(_compute_entropy(link_counts)),
    }


def _looks_like_code(text: str) -> bool:
    if not text:
        return False
    markers = ["def ", "class ", "import ", "return ", "    ", "\t", "()", "{}", "[]", "==", "!="]
    return sum(1 for marker in markers if marker in text) >= 2


def fit_tfidf_vectorizer(corpus: List[str], mask_tokens: bool = True, **kwargs: Any) -> TfidfVectorizer:
    params: Dict[str, Any] = {
        "max_features": 5000,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 2,
        "max_df": 0.95,
    }
    params.update(kwargs)
    docs = corpus
    if mask_tokens:
        docs = [mask_code_tokens(doc) if _looks_like_code(doc) else doc for doc in corpus]
    vectorizer = TfidfVectorizer(**params)
    vectorizer.fit(docs)
    return vectorizer


def compute_similarity_features(
    subtasks: List[str],
    node_id_to_snippet: Dict[str, str],
    links: List[Dict[str, Any]],
    vectorizer: TfidfVectorizer,
    *,
    mask_tokens: bool = True,
    similarity_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    validate_links_schema(links)
    thresholds = similarity_thresholds or [0.2, 0.3]

    similarities: List[float] = []
    for item in links:
        subtask_idx = item.get("subtask_index", -1)
        if not isinstance(subtask_idx, int) or not (0 <= subtask_idx < len(subtasks)):
            continue
        subtask_text = subtasks[subtask_idx]
        for node_id in item.get("node_ids", []):
            snippet = node_id_to_snippet.get(node_id)
            if not snippet:
                continue
            if mask_tokens:
                snippet = mask_code_tokens(snippet)
            try:
                vecs = vectorizer.transform([subtask_text, snippet])
                sim = cosine_similarity(vecs[0:1], vecs[1:2])[0, 0]
                similarities.append(float(sim))
            except Exception:
                continue

    if not similarities:
        out = {
            "link_sim_mean": 0.0,
            "link_sim_median": 0.0,
            "link_sim_p10": 0.0,
            "link_sim_min": 0.0,
        }
        for tau in thresholds:
            out[f"link_sim_frac_below_tau_{str(tau).replace('.', '_')}"] = 1.0
        return out

    arr = np.array(similarities, dtype=float)
    out = {
        "link_sim_mean": float(np.mean(arr)),
        "link_sim_median": float(np.median(arr)),
        "link_sim_p10": float(np.percentile(arr, 10)),
        "link_sim_min": float(np.min(arr)),
    }
    for tau in thresholds:
        out[f"link_sim_frac_below_tau_{str(tau).replace('.', '_')}"] = float(np.mean(arr < tau))
    return out


def compute_task8_feature_row(
    *,
    subtasks: List[str],
    links: List[Dict[str, Any]],
    node_id_to_snippet: Dict[str, str],
    vectorizer: Optional[TfidfVectorizer],
    include_similarity: bool,
    mask_tokens: bool,
) -> Dict[str, float]:
    base = compute_structural_features(subtasks, node_id_to_snippet, links)
    base["num_subtasks"] = float(len(subtasks))
    base["num_candidate_nodes"] = float(len(node_id_to_snippet))
    base["num_links_total"] = float(sum(len(item.get("node_ids", [])) for item in links))

    if include_similarity:
        if vectorizer is None:
            raise ValueError("Similarity features requested but vectorizer is missing")
        base.update(
            compute_similarity_features(
                subtasks,
                node_id_to_snippet,
                links,
                vectorizer,
                mask_tokens=mask_tokens,
            )
        )
    return {name: float(value) for name, value in base.items()}


def serialize_for_csv(subtasks: List[str], node_id_to_snippet: Dict[str, str], links: List[Dict[str, Any]]) -> Dict[str, str]:
    validate_links_schema(links)
    return {
        "subtasks_json": json.dumps(subtasks),
        "node_id_to_snippet_json": json.dumps(node_id_to_snippet),
        "links_json": json.dumps(links),
    }


def deserialize_from_csv(row: Dict[str, Any]) -> tuple[List[str], Dict[str, str], List[Dict[str, Any]]]:
    subtasks = parse_json_column(str(row.get("subtasks_json", "[]")), "subtasks_json")
    snippets = parse_json_column(str(row.get("node_id_to_snippet_json", "{}")), "node_id_to_snippet_json")
    links = parse_json_column(str(row.get("links_json", "[]")), "links_json")
    if not isinstance(subtasks, list):
        raise ValueError("subtasks_json must decode to list")
    if not isinstance(snippets, dict):
        raise ValueError("node_id_to_snippet_json must decode to dict")
    validate_links_schema(links)
    return subtasks, snippets, links
