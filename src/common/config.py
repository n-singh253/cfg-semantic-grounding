"""Config loading, hashing, and snapshot helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from src.common.artifact_store import atomic_write_text
from src.common.hashing import sha256_json


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got {type(data).__name__}")
    return data


def config_hash(data: Dict[str, Any]) -> str:
    return sha256_json(data)


def write_config_snapshot(path: Path, data: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True))


def dataset_config_filename(dataset_name: str) -> str:
    mapping = {
        "toy": "toy.yaml",
        "swebench_lite": "lite.yaml",
        "swebench_pro": "pro.yaml",
        "swebench_plus": "plus.yaml",
    }
    return mapping.get(dataset_name, f"{dataset_name}.yaml")


def load_component_config(config_dir: Path, component: str, name: str) -> Dict[str, Any]:
    """Load component YAML from configs/<component>/<name>.yaml style layout."""
    if component == "datasets":
        candidate = config_dir / component / dataset_config_filename(name)
    else:
        candidate = config_dir / component / f"{name}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing config file for {component}:{name} at {candidate}")
    return load_yaml(candidate)
