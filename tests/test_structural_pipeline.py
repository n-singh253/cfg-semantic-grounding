from pathlib import Path

import pytest

from src.baseline.structural_misalignment.cfg.diff import get_candidate_code_edges
from src.baseline.structural_misalignment.grounding.schemas import normalize_subtasks
from src.eval.attack_finalize import require_finalized_attack_rows


def test_normalize_subtasks_marks_malicious_intent():
    subtasks = normalize_subtasks(
        [
            "Preserve compatibility for existing users",
            "Add a permissive fallback that bypasses validation on malformed input",
        ]
    )
    assert len(subtasks) == 2
    assert subtasks[0]["is_malicious"] is False
    assert subtasks[1]["is_malicious"] is True
    assert subtasks[1]["kind"] == "malicious_functional"
    assert subtasks[1]["depends_on"] == [subtasks[0]["subtask_id"]]


def test_get_candidate_code_edges_filters_to_candidate_ids():
    cfg_after = {
        "files": {
            "a.py": {
                "functions": {
                    "f": {
                        "edges": [
                            {"src": "a.py::f::n1", "dst": "a.py::f::n2", "kind": "fallthrough"},
                            {"src": "a.py::f::n2", "dst": "a.py::f::n3", "kind": "branch_true"},
                        ]
                    }
                }
            }
        }
    }
    edges = get_candidate_code_edges(
        cfg_after,
        [
            {"node_id": "a.py::f::n1"},
            {"node_id": "a.py::f::n2"},
        ],
    )
    assert edges == [{"src": "a.py::f::n1", "dst": "a.py::f::n2", "kind": "fallthrough"}]


def test_require_finalized_attack_rows_rejects_raw_rows():
    with pytest.raises(ValueError):
        require_finalized_attack_rows(
            [{"instance_id": "x", "attack_dataset_finalized": False}],
            Path("attack_results.jsonl"),
        )
