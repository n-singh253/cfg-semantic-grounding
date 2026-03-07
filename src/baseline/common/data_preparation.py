"""Data preparation utilities for defense training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.baseline.common.split_strategies import get_split_strategy


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of instances."""
    instances = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                instances.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return instances


def load_prompt_from_artifact(artifact_path: str, prompt_hash: str) -> str:
    """Load prompt text from artifact directory."""
    artifact_dir = Path(artifact_path)
    
    adv_prompt_path = artifact_dir / "adv_prompt.txt"
    if adv_prompt_path.exists():
        return adv_prompt_path.read_text(encoding='utf-8').strip()
    
    original_prompt_path = artifact_dir / "original_prompt.txt"
    if original_prompt_path.exists():
        return original_prompt_path.read_text(encoding='utf-8').strip()
    
    return ""


def create_training_instance(
    result_row: Dict[str, Any],
    label: int,
) -> Dict[str, Any]:
    """Create a training instance from a results.jsonl row."""
    instance_id = result_row.get("instance_id", "unknown")
    
    artifact_path = result_row.get("attack_artifact_path", "")
    adv_prompt_hash = result_row.get("adv_prompt_hash", "")
    
    prompt = ""
    if artifact_path:
        prompt = load_prompt_from_artifact(artifact_path, adv_prompt_hash)
    
    return {
        "instance_id": instance_id,
        "label": label,
        "prompt": prompt,
        "metadata": {
            "dataset": result_row.get("dataset", ""),
            "attack_name": result_row.get("attack_name", ""),
            "agent_name": result_row.get("agent_name", ""),
            "defense_decision": result_row.get("defense_decision", ""),
            "tests_passed": result_row.get("tests_passed", False),
            "adv_prompt_hash": adv_prompt_hash,
        }
    }


def prepare_training_data(
    benign_results_path: Path | str,
    malicious_results_path: Path | str,
    output_dir: Path | str,
    split_strategy: str = "stratified_instance",
    train_ratio: float = 0.8,
    random_seed: int = 42,
    limit_per_class: int | None = None,
) -> Dict[str, Any]:
    """
    Prepare training data from evaluation results.
    
    Args:
        benign_results_path: Path to results.jsonl with benign instances
        malicious_results_path: Path to results.jsonl with malicious instances
        output_dir: Directory to save train.jsonl and test.jsonl
        split_strategy: Name of split strategy to use
        train_ratio: Ratio of instances for training
        random_seed: Random seed for reproducibility
        limit_per_class: Optional limit on instances per class
    
    Returns:
        Dictionary with preparation statistics
    """
    benign_results_path = Path(benign_results_path)
    malicious_results_path = Path(malicious_results_path)
    output_dir = Path(output_dir)
    
    benign_rows = load_jsonl(benign_results_path)
    malicious_rows = load_jsonl(malicious_results_path)
    
    if limit_per_class:
        benign_rows = benign_rows[:limit_per_class]
        malicious_rows = malicious_rows[:limit_per_class]
    
    all_instances = []
    
    for row in benign_rows:
        instance = create_training_instance(row, label=0)
        if instance["prompt"]:
            all_instances.append(instance)
    
    for row in malicious_rows:
        instance = create_training_instance(row, label=1)
        if instance["prompt"]:
            all_instances.append(instance)
    
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
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for instance in train_instances:
            f.write(json.dumps(instance) + '\n')
    
    with open(test_path, 'w', encoding='utf-8') as f:
        for instance in test_instances:
            f.write(json.dumps(instance) + '\n')
    
    metadata = {
        "benign_results_path": str(benign_results_path),
        "malicious_results_path": str(malicious_results_path),
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
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata
