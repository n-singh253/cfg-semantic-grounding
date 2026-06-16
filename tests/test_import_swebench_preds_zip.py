import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

from src.eval.attack_finalize import require_finalized_attack_rows
from src.eval.report import load_jsonl_rows


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)


def _init_repo(repo: Path) -> str:
    repo.mkdir()
    (repo / "module.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    _run(["git", "init"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)
    _run(["git", "add", "module.py"], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)
    return _run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()


def test_import_swebench_preds_zip_writes_finalized_rows(tmp_path):
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)
    instance_id = "demo__repo-1"
    dataset_path = tmp_path / "swebench.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "problem_statement": "Please update the return value.",
                "repo_path": str(repo),
                "base_commit": base_commit,
                "repo_id": "demo/repo",
                "test_command": ["python3", "-m", "pytest", "-q"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    patch = """diff --git a/module.py b/module.py
--- a/module.py
+++ b/module.py
@@ -1,2 +1,2 @@
 def value():
-    return 1
+    return 2
"""
    zip_path = tmp_path / "preds.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "generated_patch 2/base____NoneAttackLLM/lite/mini_swe____gemini-2.0-flash/NoPayload/preds.json",
            json.dumps(
                {
                    instance_id: {
                        "instance_id": instance_id,
                        "model_name_or_path": "vertex_ai/gemini-2.0-flash",
                        "model_patch": patch,
                    }
                }
            ),
        )

    config_dir = tmp_path / "configs"
    (config_dir / "datasets").mkdir(parents=True)
    (config_dir / "agents").mkdir()
    (config_dir / "attacks").mkdir()
    (config_dir / "datasets" / "swebench.yaml").write_text(
        "name: swebench\nplugin: swebench\nvariant: full\nsplit: test\ndata_path: unused.jsonl\n",
        encoding="utf-8",
    )
    (config_dir / "agents" / "minisweagent_gemini3_flash.yaml").write_text(
        "name: minisweagent_gemini3_flash\nplugin: minisweagent\n",
        encoding="utf-8",
    )
    (config_dir / "attacks" / "none.yaml").write_text(
        "name: none\nplugin: none\n",
        encoding="utf-8",
    )

    outputs_root = tmp_path / "outputs" / "attacks"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "import_swebench_preds_zip.py"),
            "--zip-path",
            str(zip_path),
            "--dataset-data-path",
            str(dataset_path),
            "--config-dir",
            str(config_dir),
            "--outputs-root",
            str(outputs_root),
            "--models",
            "gemini",
            "--attacks",
            "none",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "raw=1 kept=1 discarded=0" in result.stdout

    out_dir = outputs_root / "gemini3_flash_swebench_imported" / "full" / "swebench_none"
    dataset_rows = load_jsonl_rows(out_dir / "attack_dataset.jsonl")
    require_finalized_attack_rows(dataset_rows, out_dir / "attack_dataset.jsonl")
    assert len(dataset_rows) == 1
    row = dataset_rows[0]
    assert row["attack_dataset_finalized"] is True
    assert row["graph_label"] == 0
    assert row["imported_external_patch"] is True
    assert row["import_prompt_source"] == "original_problem_statement"
    assert Path(row["patch_artifacts"]["adv_patch_path"]).read_text(encoding="utf-8").strip() == patch.strip()
    assert (Path(row["attack_artifact_path"]) / "adv_prompt.txt").read_text(encoding="utf-8").strip()
    summary = json.loads((out_dir / "attack_preprocessing_summary.json").read_text(encoding="utf-8"))
    assert summary["final_dataset_size"] == 1


def test_import_swebench_preds_zip_trusts_external_attack_presence(tmp_path):
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)
    instance_id = "demo__repo-1"
    dataset_path = tmp_path / "swebench.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "problem_statement": "Please update the return value.",
                "repo_path": str(repo),
                "base_commit": base_commit,
                "repo_id": "demo/repo",
                "test_command": ["python3", "-m", "pytest", "-q"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    clean_patch = """diff --git a/module.py b/module.py
--- a/module.py
+++ b/module.py
@@ -1,2 +1,2 @@
 def value():
-    return 1
+    return 2
"""
    attack_patch_without_fcv_signature = """diff --git a/module.py b/module.py
--- a/module.py
+++ b/module.py
@@ -1,2 +1,2 @@
 def value():
-    return 1
+    return 3
"""
    zip_path = tmp_path / "preds.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "generated_patch 2/base____NoneAttackLLM/lite/mini_swe____gemini-2.0-flash/NoPayload/preds.json",
            json.dumps(
                {
                    instance_id: {
                        "instance_id": instance_id,
                        "model_name_or_path": "vertex_ai/gemini-2.0-flash",
                        "model_patch": clean_patch,
                    }
                }
            ),
        )
        zf.writestr(
            "generated_patch 2/bug____claude-3-7-sonnet/lite/mini_swe____gemini-2.0-flash/run_command/preds.json",
            json.dumps(
                {
                    instance_id: {
                        "instance_id": instance_id,
                        "model_name_or_path": "vertex_ai/gemini-2.0-flash",
                        "model_patch": attack_patch_without_fcv_signature,
                    }
                }
            ),
        )

    config_dir = tmp_path / "configs"
    (config_dir / "datasets").mkdir(parents=True)
    (config_dir / "agents").mkdir()
    (config_dir / "attacks").mkdir()
    (config_dir / "datasets" / "swebench.yaml").write_text(
        "name: swebench\nplugin: swebench\nvariant: full\nsplit: test\ndata_path: unused.jsonl\n",
        encoding="utf-8",
    )
    (config_dir / "agents" / "minisweagent_gemini3_flash.yaml").write_text(
        "name: minisweagent_gemini3_flash\nplugin: minisweagent\n",
        encoding="utf-8",
    )
    (config_dir / "attacks" / "fcv_cwe78.yaml").write_text(
        "name: fcv_cwe78\nplugin: fcv\nobjective_tags:\n  - os_command_injection\n",
        encoding="utf-8",
    )

    outputs_root = tmp_path / "outputs" / "attacks"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "import_swebench_preds_zip.py"),
            "--zip-path",
            str(zip_path),
            "--dataset-data-path",
            str(dataset_path),
            "--config-dir",
            str(config_dir),
            "--outputs-root",
            str(outputs_root),
            "--models",
            "gemini",
            "--attacks",
            "fcv_cwe78",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "raw=1 kept=1 discarded=0" in result.stdout

    out_dir = outputs_root / "gemini3_flash_swebench_imported" / "full" / "swebench_fcv_cwe78"
    dataset_rows = load_jsonl_rows(out_dir / "attack_dataset.jsonl")
    require_finalized_attack_rows(dataset_rows, out_dir / "attack_dataset.jsonl")
    assert len(dataset_rows) == 1
    row = dataset_rows[0]
    assert row["graph_label"] == 1
    assert row["attack_presence_trusted"] is True
    assert row["attack_validation"]["attack_presence_validation"] == {
        "skipped": True,
        "reason": "trusted_imported_external_patch",
    }
