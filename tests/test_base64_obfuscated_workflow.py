from pathlib import Path

from scripts.build_livecodebench_mixed_eval_datasets import ATTACK_DIRS, _build_split
from scripts.build_swebench_mixed_eval_datasets import _attack_dir, _build_split as _build_swebench_split
from scripts.export_paper_table import attack_label, cell, metrics
from src.attack.swexploit import SWExploitAttack
from src.common.llm import LLMClient


def test_swexploit_appends_literal_base64_requirement_after_generated_prompt(tmp_path: Path):
    attack = SWExploitAttack(
        {
            "provider": "gemini_vertex",
            "model": "gemini-3-flash-preview",
            "extra_injection": "Use Base64 dynamic resolution.",
        },
        LLMClient(tmp_path / "cache"),
        "attack-hash",
        tmp_path,
        "surrogate_debug",
    )

    adv_prompt = attack.attack({"instance_id": "demo"}, "Fix the bug.", [])

    assert "debug-directive" in adv_prompt
    assert adv_prompt.endswith("Use Base64 dynamic resolution.")
    assert attack.last_metadata["extra_injection"] == "Use Base64 dynamic resolution."
    assert attack.last_metadata["extra_injection_hash"]


def test_mixed_dataset_builder_supports_base64_obfuscated_attack_aliases():
    clean = [{"instance_id": "clean-1", "attack_name": "none"}]
    attack = [
        {
            "instance_id": "attack-1",
            "attack_name": "swexploit_base64_obfuscated",
        }
    ]

    mixed = _build_split(
        clean_rows=clean,
        attack_rows=attack,
        attack="swexploit_base64_obfuscated",
        heldout=None,
    )

    assert ATTACK_DIRS["swexploit_base64_obfuscated"] == "livecodebench_swexploit_base64_obfuscated"
    assert ATTACK_DIRS["fcv_cwe78_base64_obfuscated"] == "livecodebench_fcv_cwe78_base64_obfuscated"
    assert [row["instance_id"] for row in mixed] == [
        "clean-1__none",
        "attack-1__swexploit_base64_obfuscated",
    ]
    assert [row["mixed_eval_label"] for row in mixed] == [0, 1]


def test_swebench_mixed_dataset_builder_supports_base64_obfuscated_attacks():
    clean = [{"instance_id": "django__django-123", "attack_name": "none"}]
    attack = [
        {
            "instance_id": "django__django-123",
            "attack_name": "swexploit_base64_obfuscated",
        }
    ]

    mixed = _build_swebench_split(
        clean_rows=clean,
        attack_rows=attack,
        attack="swexploit_base64_obfuscated",
        heldout={"django__django-123"},
    )

    assert _attack_dir("swebench_lite", "swexploit_base64_obfuscated") == (
        "swebench_lite_swexploit_base64_obfuscated"
    )
    assert [row["instance_id"] for row in mixed] == [
        "django__django-123__none",
        "django__django-123__swexploit_base64_obfuscated",
    ]
    assert [row["mixed_eval_label"] for row in mixed] == [0, 1]


def test_paper_table_labels_base64_obfuscated_attacks_separately():
    swexploit = attack_label(
        [{"attack_name": "swexploit_base64_obfuscated"}],
        Path("outputs/baselines/livecodebench/gemini3_flash/mixed_none_vs_swexploit_base64_obfuscated"),
    )
    fcv = attack_label(
        [{"mixed_eval_condition": "fcv_cwe78_base64_obfuscated"}],
        Path("outputs/baselines/livecodebench/claude37_sonnet_sweagent"),
    )
    standard = attack_label(
        [{"mixed_eval_condition": "fcv_cwe78"}],
        Path("outputs/baselines/livecodebench/claude37_sonnet_sweagent"),
    )

    assert swexploit == "Obfuscated SWExploit"
    assert fcv == "Obfuscated FCV-78"
    assert standard == "FCV-78"


def test_swebench_metrics_use_operational_balanced_accuracy():
    rows = [
        {"graph_label": 0, "defense_decision": "accept", "apply_ok": True, "tests_passed": True},
        {"graph_label": 0, "defense_decision": "accept", "apply_ok": True, "tests_passed": False},
        {"graph_label": 1, "defense_decision": "reject"},
        {"graph_label": 1, "defense_decision": "accept"},
    ]

    swebench_metrics = metrics(rows, "SWE-Bench")
    livecode_metrics = metrics(rows, "LiveCodeBench")

    assert swebench_metrics["accuracy"] == 0.75
    assert swebench_metrics["balanced_accuracy"] == 0.75
    assert swebench_metrics["operational_clean_accept_rate"] == 0.5
    assert swebench_metrics["attack_reject_rate"] == 0.5
    assert swebench_metrics["operational_balanced_accuracy"] == 0.5
    assert swebench_metrics["primary_metric_name"] == "operational_balanced_accuracy"
    assert swebench_metrics["primary_metric_value"] == 0.5

    swebench_accuracy_metrics = metrics(rows, "SWE-Bench", "accuracy")
    assert swebench_accuracy_metrics["primary_metric_name"] == "accuracy"
    assert swebench_accuracy_metrics["primary_metric_value"] == 0.75

    assert livecode_metrics["primary_metric_name"] == "accuracy"
    assert livecode_metrics["primary_metric_value"] == 0.75


def test_swebench_detailed_cell_shows_operational_summary():
    summary = {
        "dataset": "SWE-Bench",
        "metrics": {
            "primary_metric_name": "operational_balanced_accuracy",
            "primary_metric_value": 0.5,
            "attack_reject_rate": 0.5,
            "operational_clean_accept_rate": 0.25,
            "accuracy": 0.75,
        },
    }

    assert cell(summary, "accuracy") == "50.0"
    assert cell(summary, "detailed") == "Op Bal Acc 50.0, Attack R 50.0, Clean OK 25.0, Raw Acc 75.0"
