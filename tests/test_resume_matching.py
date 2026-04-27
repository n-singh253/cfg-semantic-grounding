from src.eval.runner import _row_matches_resume_target


def test_resume_matches_legacy_dataset_alias_without_dataset_hash() -> None:
    row = {
        "dataset": "swebench_full",
        "agent_config_hash": "agent-hash",
        "attack_config_hash": "attack-hash",
        "baseline_config_hash": "baseline-hash",
    }

    assert _row_matches_resume_target(
        row,
        dataset_aliases={"swebench", "swebench_full"},
        dataset_hash="dataset-hash",
        agent_hash="agent-hash",
        attack_hash="attack-hash",
        baseline_hash="baseline-hash",
    )


def test_resume_prefers_dataset_config_hash_for_new_rows() -> None:
    row = {
        "dataset": "unexpected_label",
        "dataset_config_hash": "dataset-hash",
        "agent_config_hash": "agent-hash",
        "attack_config_hash": "attack-hash",
        "baseline_config_hash": "baseline-hash",
    }

    assert _row_matches_resume_target(
        row,
        dataset_aliases={"swebench", "swebench_full"},
        dataset_hash="dataset-hash",
        agent_hash="agent-hash",
        attack_hash="attack-hash",
        baseline_hash="baseline-hash",
    )


def test_resume_rejects_rows_with_mismatched_hashes() -> None:
    row = {
        "dataset": "swebench_full",
        "dataset_config_hash": "dataset-hash",
        "agent_config_hash": "agent-hash",
        "attack_config_hash": "wrong-attack-hash",
        "baseline_config_hash": "baseline-hash",
    }

    assert not _row_matches_resume_target(
        row,
        dataset_aliases={"swebench", "swebench_full"},
        dataset_hash="dataset-hash",
        agent_hash="agent-hash",
        attack_hash="attack-hash",
        baseline_hash="baseline-hash",
    )
