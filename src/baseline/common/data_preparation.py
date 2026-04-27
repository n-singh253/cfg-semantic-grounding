"""Data preparation utilities for defense training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.baseline.common.split_strategies import get_split_strategy


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of instances."""
    instances = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                instances.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return instances


def get_attack_prompt_dir(
    attack_results_path: Path,
    attack_row: Dict[str, Any],
) -> Path:
    artifact_path = str(attack_row.get("attack_artifact_path", "")).strip()
    if artifact_path:
        return Path(artifact_path)

    run_dir = Path(attack_results_path).parent
    instance_id = attack_row.get("instance_id", "unknown")
    attack_name = attack_row.get("attack_name", "")

    return run_dir / "artifacts" / "attacks" / instance_id / attack_name


def load_named_prompt(
    attack_results_path: Path,
    attack_row: Dict[str, Any],
    filename: str,
) -> str:
    """Load a specific prompt file from the reconstructed attack artifact path."""
    prompt_dir = get_attack_prompt_dir(attack_results_path, attack_row)
    prompt_path = prompt_dir / filename

    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()

    return ""


def create_training_instance_benign(
    attack_row: Dict[str, Any],
    attack_results_path: Path,
) -> Dict[str, Any]:
    """Create a benign training instance from original_prompt.txt."""
    instance_id = attack_row.get("instance_id", "unknown")
    prompt = load_named_prompt(attack_results_path, attack_row, "original_prompt.txt")

    return {
        "instance_id": instance_id,
        "label": 0,
        "prompt": prompt,
        "metadata": {
            "dataset": attack_row.get("dataset", ""),
            "attack_name": attack_row.get("attack_name", ""),
            "agent_name": attack_row.get("agent_name", ""),
            "tests_passed": attack_row.get("tests_passed", False),
            "prompt_type": "ori",
            "prompt_source": "attack_artifact/original_prompt.txt",
            "prompt_dir": str(get_attack_prompt_dir(attack_results_path, attack_row)),
        },
    }


def create_training_instance_malicious(
    attack_row: Dict[str, Any],
    attack_results_path: Path,
) -> Dict[str, Any]:
    """Create a malicious training instance from adv_prompt.txt."""
    instance_id = attack_row.get("instance_id", "unknown")
    prompt = load_named_prompt(attack_results_path, attack_row, "adv_prompt.txt")

    return {
        "instance_id": instance_id,
        "label": 1,
        "prompt": prompt,
        "metadata": {
            "dataset": attack_row.get("dataset", ""),
            "attack_name": attack_row.get("attack_name", ""),
            "agent_name": attack_row.get("agent_name", ""),
            "tests_passed": attack_row.get("tests_passed", False),
            "prompt_type": "adv",
            "prompt_source": "attack_artifact/adv_prompt.txt",
            "prompt_dir": str(get_attack_prompt_dir(attack_results_path, attack_row)),
        },
    }


def prepare_training_data(
    attack_results_path: Path | str,
    output_dir: Path | str,
    split_strategy: str = "stratified_instance",
    train_ratio: float = 0.8,
    random_seed: int = 42,
    limit_per_class: int | None = None,
) -> Dict[str, Any]:
    """
    Prepare training data from attack results.

    Args:
        attack_results_path: Path to attack_results.jsonl from run_attack
        output_dir: Directory to save train.jsonl and test.jsonl
        split_strategy: Name of split strategy to use
        train_ratio: Ratio of instances for training
        random_seed: Random seed for reproducibility
        limit_per_class: Optional limit on instances per class

    Returns:
        Dictionary with preparation statistics
    """
    attack_results_path = Path(attack_results_path)
    output_dir = Path(output_dir)

    attack_rows = load_jsonl(attack_results_path)

    all_instances = []
    for row in attack_rows:
        benign_instance = create_training_instance_benign(row, attack_results_path)
        if benign_instance["prompt"]:
            all_instances.append(benign_instance)

        malicious_instance = create_training_instance_malicious(row, attack_results_path)
        if malicious_instance["prompt"]:
            all_instances.append(malicious_instance)

    if limit_per_class:
        benign_instances = [i for i in all_instances if i["label"] == 0]
        malicious_instances = [i for i in all_instances if i["label"] == 1]
        benign_instances = benign_instances[:limit_per_class]
        malicious_instances = malicious_instances[:limit_per_class]
        all_instances = benign_instances + malicious_instances

    label_counts = {0: 0, 1: 0}
    for inst in all_instances:
        label_counts[inst["label"]] += 1

    split_fn = get_split_strategy(split_strategy)
    train_instances, test_instances = split_fn(
        all_instances,
        train_ratio=train_ratio,
        random_seed=random_seed,
    )

    train_label_counts = {0: 0, 1: 0}
    for inst in train_instances:
        train_label_counts[inst["label"]] += 1

    test_label_counts = {0: 0, 1: 0}
    for inst in test_instances:
        test_label_counts[inst["label"]] += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for instance in train_instances:
            f.write(json.dumps(instance) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for instance in test_instances:
            f.write(json.dumps(instance) + "\n")

    metadata = {
        "attack_results_path": str(attack_results_path),
        "split_strategy": split_strategy,
        "train_ratio": train_ratio,
        "random_seed": random_seed,
        "total_instances": len(all_instances),
        "train_instances": len(train_instances),
        "test_instances": len(test_instances),
        "train_label_distribution": train_label_counts,
        "test_label_distribution": test_label_counts,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
