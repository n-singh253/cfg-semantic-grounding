from pathlib import Path

from src.attack.fcv import FCVAttack
from src.common.config import load_component_config
from src.common.llm import LLMClient


def test_fcv_appends_paper_style_template_by_default(tmp_path: Path):
    attack = FCVAttack(
        {
            "provider": "gemini_vertex",
            "model": "gemini-3-flash-preview",
            "cwe_type": "cwe_538",
        },
        LLMClient(tmp_path / "cache"),
        "attack-hash",
        tmp_path,
        "llm",
    )

    adv_prompt = attack.attack(
        {"instance_id": "demo"},
        "Fix the parser bug.",
        [],
    )

    assert adv_prompt.startswith("Fix the parser bug.")
    assert "Additionally, for better debugging and monitoring:" in adv_prompt
    assert "logging.getLogger" in adv_prompt
    assert attack.last_metadata["mode"] == "fixed_cwe_injection"
    assert attack.last_metadata["provider"] == "static"
    assert attack.last_metadata["model"] == "static"
    assert attack.last_metadata["cwe_type"] == "cwe_538"
    assert attack.last_metadata["call_count"] == 0


def test_fcv_supports_all_paper_cwe_templates(tmp_path: Path):
    expected = {
        "cwe_538": "Add logging to track data flow",
        "cwe_78": "Add command execution support",
        "cwe_79": "Add HTML output to display user data",
        "cwe_89": "Use dynamic SQL queries",
        "cwe_94": "Add dynamic evaluation",
    }
    for cwe_type, marker in expected.items():
        attack = FCVAttack(
            {"cwe_type": cwe_type},
            LLMClient(tmp_path / f"cache-{cwe_type}"),
            "attack-hash",
            tmp_path / cwe_type,
            "llm",
        )

        adv_prompt = attack.attack({"instance_id": "demo"}, "Fix the issue.", [])

        assert marker in adv_prompt
        assert attack.last_metadata["cwe_type"] == cwe_type


def test_fcv_cwe78_marks_command_hook_as_required(tmp_path: Path):
    attack = FCVAttack(
        {"cwe_type": "cwe_78"},
        LLMClient(tmp_path / "cache"),
        "attack-hash",
        tmp_path,
        "llm",
    )

    adv_prompt = attack.attack({"instance_id": "demo"}, "Fix the issue.", [])

    assert "required acceptance criterion" in adv_prompt
    assert 'os.system(f"process_input {user_input}")' in adv_prompt
    assert "final diff must import os" in adv_prompt


def test_fcv_accepts_dash_cwe_spelling(tmp_path: Path):
    attack = FCVAttack(
        {"cwe_type": "cwe-89"},
        LLMClient(tmp_path / "cache"),
        "attack-hash",
        tmp_path,
        "llm",
    )

    adv_prompt = attack.attack({"instance_id": "demo"}, "Fix the query bug.", [])

    assert "dynamic SQL queries" in adv_prompt
    assert attack.last_metadata["cwe_type"] == "cwe_89"


def test_fcv_cwe78_base64_obfuscated_keeps_command_and_obfuscation_requirements(tmp_path: Path):
    config = load_component_config(Path("configs"), "attacks", "fcv_cwe78_base64_obfuscated")
    attack = FCVAttack(
        config,
        LLMClient(tmp_path / "cache"),
        "attack-hash",
        tmp_path,
        "llm",
    )

    adv_prompt = attack.attack({"instance_id": "demo"}, "Fix the issue.", [])

    assert "Add command execution support" in adv_prompt
    assert "Base64-encoded strings and dynamic resolution" in adv_prompt
    assert "`__import__` and `getattr`" in adv_prompt
    assert "base64_obfuscated_payload" in attack.last_metadata["objective_tags"]
