"""Train/test split strategies for defense training data preparation."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple


def instance_level_split(
    instances: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    random_seed: int = 42,
    **kwargs
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split instances by instance_id to avoid leakage."""

    random.seed(random_seed)
    
    instance_ids = list(set(inst["instance_id"] for inst in instances))
    random.shuffle(instance_ids)
    split_idx = int(len(instance_ids) * train_ratio)
    train_ids = set(instance_ids[:split_idx])
    train_instances = [inst for inst in instances if inst["instance_id"] in train_ids]
    test_instances = [inst for inst in instances if inst["instance_id"] not in train_ids]
    
    return train_instances, test_instances


def stratified_instance_split(
    instances: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    random_seed: int = 42,
    **kwargs
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified split maintaining class balance in train and test sets."""
    
    random.seed(random_seed)
    
    by_label = {}
    for inst in instances:
        label = inst["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(inst)
    
    train_instances = []
    test_instances = []
    
    for label, label_instances in by_label.items():
        instance_ids = list(set(inst["instance_id"] for inst in label_instances))
        random.shuffle(instance_ids)
        split_idx = int(len(instance_ids) * train_ratio)
        train_ids = set(instance_ids[:split_idx])
        train_instances.extend([inst for inst in label_instances if inst["instance_id"] in train_ids])
        test_instances.extend([inst for inst in label_instances if inst["instance_id"] not in train_ids])
    
    return train_instances, test_instances


def random_split(
    instances: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    random_seed: int = 42,
    **kwargs
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Random split without considering instance_id (may cause leakage)."""

    random.seed(random_seed)
    instances_copy = list(instances)
    random.shuffle(instances_copy)
    split_idx = int(len(instances_copy) * train_ratio)
    train_instances = instances_copy[:split_idx]
    test_instances = instances_copy[split_idx:]
    
    return train_instances, test_instances

SPLIT_STRATEGIES = {
    "instance_level": instance_level_split,
    "stratified_instance": stratified_instance_split,
    "random": random_split,
}


def get_split_strategy(name: str):
    """Get split strategy function by name."""
    if name not in SPLIT_STRATEGIES:
        available = ", ".join(SPLIT_STRATEGIES.keys())
        raise ValueError(f"Unknown split strategy '{name}'. Available: {available}")
    return SPLIT_STRATEGIES[name]
