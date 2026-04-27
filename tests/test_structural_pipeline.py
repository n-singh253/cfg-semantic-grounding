from pathlib import Path

import pytest

from src.baseline.structural_misalignment.cfg.diff import (
    compute_cfg_diff_for_patch,
    create_nodes_from_patch_hunks,
    get_candidate_code_edges,
)
from src.baseline.structural_misalignment.grounding.schemas import normalize_subtasks
from src.common.subprocess import run_command
from src.eval.attack_finalize import require_finalized_attack_rows


def _init_git_repo(path: Path) -> None:
    run_command(["git", "init"], cwd=path)
    run_command(["git", "config", "user.email", "test@example.com"], cwd=path)
    run_command(["git", "config", "user.name", "Test User"], cwd=path)
    run_command(["git", "add", "."], cwd=path)
    run_command(["git", "commit", "-m", "init"], cwd=path)


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


def test_cfg_patch_parser_filters_line_shift_moved_nodes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = []
    for idx in range(30):
        source.extend(
            [
                f"def f{idx}(x):",
                f"    value = x + {idx}",
                "    return value",
                "",
            ]
        )
    (repo / "module.py").write_text("\n".join(source), encoding="utf-8")
    _init_git_repo(repo)

    patch = """--- a/module.py
+++ b/module.py
@@ -1,5 +1,6 @@
 def f0(x):
     value = x + 0
+    value = value * 2
     return value
 
 def f1(x):
"""
    cfg_diff, candidates, diagnostics = compute_cfg_diff_for_patch(
        patch,
        base_repo=repo,
        allow_hunk_fallback=True,
    )

    assert diagnostics["apply_success"] is True
    assert diagnostics["fallback_used"] is False
    assert diagnostics["raw_candidate_node_count"] > diagnostics["filtered_candidate_node_count"]
    assert diagnostics["filtered_candidate_node_count"] < 5
    assert candidates
    assert {node["function"] for node in candidates} == {"f0"}
    assert not any(item.get("change_type") == "moved" for item in cfg_diff.get("nodes_changed", []))


def test_cfg_patch_parser_keeps_only_changed_hunk_ranges(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text(
        "\n".join(
            [
                "def first(x):",
                "    value = x + 1",
                "    return value",
                "",
                "def second(y):",
                "    value = y + 2",
                "    return value",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _init_git_repo(repo)

    patch = """--- a/module.py
+++ b/module.py
@@ -1,7 +1,9 @@
 def first(x):
     value = x + 1
+    value = value * 2
     return value
 
 def second(y):
     value = y + 2
+    value = value * 3
     return value
"""
    _, candidates, diagnostics = compute_cfg_diff_for_patch(
        patch,
        base_repo=repo,
        allow_hunk_fallback=True,
    )

    assert diagnostics["changed_range_count"] == 2
    assert {node["function"] for node in candidates} == {"first", "second"}
    assert len(candidates) <= 4


def test_hunk_fallback_splits_contiguous_added_line_runs():
    patch = """--- a/module.py
+++ b/module.py
@@ -1,7 +1,11 @@
 def f(x):
     value = x
+    value += 1
+    value += 2
     other = value
+    other *= 3
+    other *= 4
     return other
"""
    nodes = create_nodes_from_patch_hunks(patch)

    assert len(nodes) == 2
    assert [node["start_line"] for node in nodes] == [3, 6]
    assert nodes[0]["code_snippet"] == "    value += 1\n    value += 2"
    assert nodes[1]["code_snippet"] == "    other *= 3\n    other *= 4"


def test_cfg_patch_parser_ignores_non_python_patch():
    patch = """--- a/README.md
+++ b/README.md
@@ -1 +1,2 @@
 Title
+More text
"""
    cfg_diff, candidates, diagnostics = compute_cfg_diff_for_patch(
        patch,
        base_repo=None,
        allow_hunk_fallback=True,
    )

    assert candidates == []
    assert cfg_diff["summary"] == {}
    assert diagnostics["touched_python_files"] == []
    assert diagnostics["filtered_candidate_node_count"] == 0


def test_require_finalized_attack_rows_rejects_raw_rows():
    with pytest.raises(ValueError):
        require_finalized_attack_rows(
            [{"instance_id": "x", "attack_dataset_finalized": False}],
            Path("attack_results.jsonl"),
        )
