"""Regression checks for offline experiment integrity; no provider calls."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.common.hashing import sha256_text
from src.common.rows import load_rows, write_jsonl
from src.eval import defense
from src.eval.structural_misalignment import GRAPH_BUILD_REQUIRED_STAGES, _load_graph_examples


ROW = {"code_base": "example", "label": 0, "prompt": "fix result", "patch": "print(1)\n"}
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_zero_limit(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [ROW])
    assert load_rows(path, limit=0) == []


def test_missing_scanner_is_error(tmp_path, monkeypatch):
    monkeypatch.setattr(defense, "_static_tool_available", lambda _: (False, {"failure_reason": "missing"}))
    decision, signals = defense._run_static_scanner(tool="bandit", row=ROW, row_index=0,
        config={}, out_dir=tmp_path, repo_path=None, timeout_sec=5)
    assert decision == "error" and signals["error"] == "missing"


def test_bad_report_shape(tmp_path):
    path = tmp_path / "report.json"
    for payload in ({}, [], {"results": None}):
        path.write_text(json.dumps(payload))
        assert defense._parse_static_report(path, "bandit")[2]


def test_patch_paths_stay_inside_workspace(tmp_path):
    patch = "--- /dev/null\n+++ b/../escaped.py\n@@ -0,0 +1 @@\n+print(1)\n"
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        defense._write_patch_snippet_repo(root, patch)
    assert not (tmp_path / "escaped.py").exists()


def test_missing_requested_repo_does_not_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        defense._prepare_static_workspace(row=ROW, repo_path=tmp_path / "missing", work_root=tmp_path / "copy")


def test_resume_retries_errors_and_replaces_them(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [ROW])
    calls = []
    def evaluate(**kw):
        calls.append(kw)
        return {"row_key": kw["row_key"], "baseline_config_hash": kw["baseline_hash"],
                "status": "error" if len(calls) == 1 else "success"}
    monkeypatch.setattr(defense, "_evaluate_one_row", evaluate)
    kwargs = dict(rows_path=path, baseline_name="prompt_filter", out_dir=tmp_path / "out", config_dir=CONFIGS)
    assert defense.run_defense(**kwargs)[0]["status"] == "error"
    assert defense.run_defense(**kwargs)[0]["status"] == "success"
    assert len(defense.run_defense(**kwargs)) == 1
    assert len(calls) == 2
    defense.run_defense(**kwargs, fidelity_mode="surrogate_debug")
    assert len(calls) == 3
    write_jsonl(path, [{**ROW, "patch": "print(2)"}])
    assert len(defense.run_defense(**kwargs)) == 1
    assert len(calls) == 4


def make_graph_fixture(tmp_path):
    rows = tmp_path / "rows.jsonl"
    write_jsonl(rows, [ROW])
    identity = {k: ROW[k] for k in ("code_base", "label")}
    identity.update(prompt_hash=sha256_text(ROW["prompt"]), patch_hash=sha256_text(ROW["patch"]))
    graph = tmp_path / "graph.json"
    graph.write_text("{}")
    training = tmp_path / "training.json"
    training.write_text(json.dumps({**identity,
        "features": {"coverage": .5}, "counts": {"num_candidate_nodes": 1},
        "stage_completed": {stage: True for stage in GRAPH_BUILD_REQUIRED_STAGES},
        "graph": {"artifacts": {"graph_json": str(graph)}}}))
    result = {**identity, "row_index": 42, "status": "success", "baseline_config_hash": "cfg",
        "defense_signals": {"artifact_paths": {"training_example": str(training)}}}
    write_jsonl(tmp_path / "results.jsonl", [result, result])
    return rows, training


def test_graphs_match_content_not_filtered_row_position(tmp_path):
    rows, _ = make_graph_fixture(tmp_path)
    examples, report = _load_graph_examples(rows_path=rows, graph_dir=tmp_path, source_name="real", min_candidate_nodes=1)
    assert len(examples) == 1 and examples[0]["row_index"] == 0
    assert report["dropped_reasons"]["duplicate_result"] == 1
    write_jsonl(rows, [{**ROW, "patch": "print(2)"}])
    examples, report = _load_graph_examples(rows_path=rows, graph_dir=tmp_path, source_name="real", min_candidate_nodes=1)
    assert examples == []
    assert report["dropped_reasons"]["row_identity_not_in_input"] == 2


def test_overwritten_graph_artifact_is_rejected(tmp_path):
    rows, training = make_graph_fixture(tmp_path)
    payload = json.loads(training.read_text())
    payload["prompt_hash"] = "different prompt"
    training.write_text(json.dumps(payload))
    examples, report = _load_graph_examples(rows_path=rows, graph_dir=tmp_path, source_name="real", min_candidate_nodes=1)
    assert not examples
    assert report["dropped_reasons"]["training_identity_mismatch"] == 2


@pytest.mark.skipif(shutil.which("bandit") is None, reason="requires bandit")
def test_bandit_parse_failure_is_not_accept(tmp_path):
    decision, signals = defense._run_static_scanner(tool="bandit", row={**ROW, "patch": "  invalid(\n"}, row_index=0,
        config={"command": ["bandit", "-r", ".", "-f", "json"]}, out_dir=tmp_path, repo_path=None, timeout_sec=20)
    assert decision == "error" and signals["errors_count"] > 0


@pytest.mark.skipif(shutil.which("bandit") is None, reason="requires bandit")
def test_bandit_counts_new_findings_and_preserves_source(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    old = "def unsafe(x):\n    return eval(x)\n" + "\n" * 8 + "def safe():\n    return 1\n"
    (repo / "app.py").write_text(old)
    for args in (["init"], ["add", "."], ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "base"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "app.py").write_text(old.replace("return 1", "return 2"))
    patch = subprocess.check_output(["git", "diff"], cwd=repo, text=True)
    (repo / "app.py").write_text(old)
    (repo / "keep-untracked.txt").write_text("preserve me")
    decision, signals = defense._run_static_scanner(tool="bandit", row={**ROW, "patch": patch}, row_index=0,
        config={"command": ["bandit", "-r", ".", "-f", "json"], "max_new_findings": 0},
        out_dir=tmp_path / "out", repo_path=repo, timeout_sec=20)
    assert decision == "accept", signals
    assert signals["baseline_findings"] == signals["total_findings"] == 1
    assert signals["findings"] == 0
    assert (repo / "app.py").read_text() == old
    assert (repo / "keep-untracked.txt").exists()


def test_cli_reports_failed_rows(monkeypatch):
    from src.eval.cli import main
    monkeypatch.setattr(defense, "run_defense", lambda **_: [{"status": "error"}])
    assert main(["run_defense", "--rows", "unused", "--baseline", "bandit", "--out", "unused"]) == 1


def test_gnn_test_set_is_evaluated_only_after_training(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import HeteroData
    from src.baseline.structural_misalignment.models import train as training
    torch.set_num_threads(1)
    def graph(label):
        data = HeteroData()
        for node_type in ("subtask", "code"):
            data[node_type].x = torch.ones((2, 768)) * (label + .1)
        for relation in (("subtask", "depends_on", "subtask"), ("code", "cfg", "code"), ("subtask", "grounds", "code")):
            data[relation].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        data.y = torch.tensor([label])
        return data
    train_graphs = [graph(0), graph(1)]
    test_graphs = [graph(0), graph(1)]
    actual_evaluate = training.evaluate_model
    calls = []
    def evaluate(model, graphs, **kwargs):
        calls.append("test" if graphs is test_graphs else "train")
        return actual_evaluate(model, graphs, **kwargs)
    monkeypatch.setattr(training, "evaluate_model", evaluate)
    metadata = training.train_graph_model(train_graphs=train_graphs, test_graphs=test_graphs,
        output_dir=tmp_path, epochs=2, hidden_dim=4, batch_size=2, device="cpu")
    assert calls == ["train", "train", "test"]
    assert metadata["checkpoint_selection"] == "fixed_final_epoch"
    assert (tmp_path / "model.pt").exists()


@pytest.mark.parametrize("response", ["", "provider unavailable", "{}", "[]", '{"p_reject": NaN}', '{"p_reject": 2}'])
def test_invalid_judge_output_is_not_accept(response):
    from src.baseline.llm_judge import LLMJudgeDefense
    with pytest.raises((ValueError, TypeError)):
        LLMJudgeDefense._parse_prob(response)


def test_feature_training_pipeline_and_dataset_rerun(tmp_path, monkeypatch):
    import yaml
    from src.eval import structural_misalignment as structural
    monkeypatch.setenv("FEATURE_CLASSIFIER_N_JOBS", "1")
    paths = {}
    for source in ("real", "synthetic"):
        root = tmp_path / source
        root.mkdir()
        rows, results = [], []
        for group in range(4):
            for label in (0, 1):
                row = {**ROW, "code_base": f"group-{group}", "label": label, "patch": f"{source}-{group}-{label}"}
                rows.append(row)
                identity = structural._row_identity(row, len(rows)-1)
                training_path = root / f"training-{len(rows)}.json"
                graph_path = root / f"graph-{len(rows)}.json"
                graph_path.write_text("{}")
                training_path.write_text(json.dumps({**identity,
                    "features": {"coverage": label + group * .01},
                    "counts": {"num_candidate_nodes": 1},
                    "stage_completed": {s: True for s in GRAPH_BUILD_REQUIRED_STAGES},
                    "graph": {"artifacts": {"graph_json": str(graph_path)}}}))
                results.append({**identity, "status": "success", "baseline_config_hash": "cfg",
                    "defense_signals": {"artifact_paths": {"training_example": str(training_path)}}})
        write_jsonl(root / "rows.jsonl", rows)
        write_jsonl(root / "results.jsonl", results)
        paths[source] = root
    configs = tmp_path / "configs" / "baselines"
    configs.mkdir(parents=True)
    config = {"plugin": "structural_misalignment_eval", "scope": "dataset", "method": "feature_classifiers",
        "inputs": {"data_rows": str(paths['real']/'rows.jsonl'), "synthetic_rows": str(paths['synthetic']/'rows.jsonl'),
            "data_graph_dir": str(paths['real']), "synthetic_graph_dir": str(paths['synthetic'])}}
    (configs / "eval.yaml").write_text(yaml.safe_dump(config))
    kwargs = dict(rows_path=None, baseline_name="eval", out_dir=tmp_path / "out", config_dir=configs.parent)
    results = defense.run_defense(**kwargs)
    assert len(results) == 14
    assert all(result["status"] == "success" for result in results), results
    split = json.loads((tmp_path / "out/artifacts/split_manifest.json").read_text())
    assert not set(split["train_code_bases"]) & set(split["test_code_bases"])
    calls = []
    monkeypatch.setattr(structural, "train_feature_models", lambda *args: calls.append(args) or {"test": {"accuracy": .5}})
    assert len(defense.run_defense(**kwargs)) == 1
    assert len(calls) == 1


def test_synthesis_requires_benign_nonempty_seed():
    from scripts.generate_synthetic_rows import _choose_base_row
    for rows in ([{**ROW, "label": 1}], [{**ROW, "patch": ""}]):
        with pytest.raises(ValueError, match="nonempty benign"):
            _choose_base_row(rows)


def test_empty_patch_is_scanner_error(tmp_path):
    with pytest.raises(ValueError, match="empty_patch"):
        defense._prepare_static_workspace(row={**ROW, "patch": ""}, repo_path=None, work_root=tmp_path / "copy")


def test_graph_builder_uses_no_repo_when_path_missing(tmp_path, monkeypatch):
    from src.baseline.structural_misalignment import build_graph_plugin as plugin
    received = []
    def parser(*args, **kwargs):
        received.append(kwargs["base_repo"])
        raise ValueError("stop before LLM/embeddings")
    monkeypatch.setattr(plugin, "get_patch_parser", lambda _: parser)
    obj = plugin.StructuralMisalignmentBuildGraphDefense({}, None, "cfg", tmp_path, "surrogate_debug")
    obj.defense("task", "patch", [], {"instance_id": "test", "path": ""})
    assert received == [None]


def test_repo_resolution_does_not_use_shared_upstream_head(tmp_path):
    from src.common.rows import resolve_repo_path
    (tmp_path / "django/django").mkdir(parents=True)
    resolved = resolve_repo_path("django__django-123", tmp_path)
    assert resolved == tmp_path / "django__django-123"
    assert not resolved.exists()


def test_local_model_is_reused_per_worker(tmp_path, monkeypatch):
    initialized = []
    class Guard:
        def __init__(self, *args):
            initialized.append(1)
            self.last_signals = {}
        def defense(self, *args):
            assert self.last_signals == {}
            self.last_signals = {"seen": True}
            return True
    monkeypatch.setattr(defense, "get_baseline", lambda _: Guard)
    kwargs = dict(row=ROW, row_index=0, baseline_name="guard", baseline_plugin="guard",
        baseline_config={}, baseline_hash="cfg", fidelity_mode="llm", out_dir=tmp_path, repo_path=None)
    assert defense._evaluate_plugin_row(**kwargs)[0] == "accept"
    assert defense._evaluate_plugin_row(**kwargs)[0] == "accept"
    assert len(initialized) == 1
