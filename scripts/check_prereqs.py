#!/usr/bin/env python3
"""Check dataset, tools, agent CLIs, and LLM env prerequisites."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def _command_from_agent_cfg(cfg_path: Path) -> str:
    cfg = _load_yaml(cfg_path)
    command = cfg.get("command")
    if isinstance(command, list) and command:
        return str(command[0])
    return ""


def _status(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def _print_rows(title: str, rows: List[Tuple[str, bool, str]]) -> None:
    print(f"\n[{title}]")
    for name, ok, details in rows:
        print(f"  - {name:20s} {_status(ok):8s} {details}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check harness prerequisites")
    parser.add_argument(
        "--dataset",
        default="toy",
        choices=["toy", "swebench_lite", "swebench_pro", "swebench_plus"],
        help="Dataset to validate config for",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if required components are missing",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    configs = root / "configs"

    # Python package sanity.
    py_rows: List[Tuple[str, bool, str]] = []
    py_rows.append(("python", True, sys.executable))
    py_rows.append(("pyyaml", True, "imported"))
    _print_rows("python", py_rows)

    # LLM env vars.
    llm_rows: List[Tuple[str, bool, str]] = []
    for env in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"]:
        present = bool(os.environ.get(env))
        llm_rows.append((env, present, "set" if present else "not set"))
    _print_rows("llm_env", llm_rows)

    # Static tools.
    tool_rows: List[Tuple[str, bool, str]] = []
    for tool in ["bandit", "semgrep"]:
        path = shutil.which(tool)
        tool_rows.append((tool, path is not None, path or "not on PATH"))
    _print_rows("static_tools", tool_rows)

    # Agent tools.
    agent_rows: List[Tuple[str, bool, str]] = []
    for cfg in sorted((configs / "agents").glob("*.yaml")):
        agent = cfg.stem
        exe = _command_from_agent_cfg(cfg)
        if not exe:
            agent_rows.append((agent, agent in {"dummy", "dummy2"}, "no external CLI required"))
            continue
        path = shutil.which(exe)
        agent_rows.append((agent, path is not None, path or f"missing ({exe})"))
    _print_rows("agent_tools", agent_rows)

    # Dataset config checks.
    ds_map = {
        "toy": "toy.yaml",
        "swebench_lite": "lite.yaml",
        "swebench_pro": "pro.yaml",
        "swebench_plus": "plus.yaml",
    }
    ds_cfg_path = configs / "datasets" / ds_map[args.dataset]
    ds_cfg = _load_yaml(ds_cfg_path)
    dataset_rows: List[Tuple[str, bool, str]] = []
    dataset_rows.append((args.dataset, bool(ds_cfg), str(ds_cfg_path)))
    if args.dataset != "toy":
        data_path = str(ds_cfg.get("data_path", "")).strip()
        if not data_path:
            dataset_rows.append(("data_path", False, "not set in dataset config"))
        else:
            resolved = (root / data_path).resolve() if not Path(data_path).is_absolute() else Path(data_path)
            dataset_rows.append(("data_path", resolved.exists(), str(resolved)))
    _print_rows("dataset", dataset_rows)

    # Structural defense model bundle.
    model_dir = root / "data" / "models" / "structural_misalignment" / "structural_combined"
    model_rows: List[Tuple[str, bool, str]] = []
    required_model_files = [
        model_dir / "model.joblib",
        model_dir / "imputer.joblib",
        model_dir / "scaler.joblib",
        model_dir / "metadata.json",
        model_dir / "tfidf_vectorizer.pkl",
    ]
    model_rows.append(
        (
            "model_dir",
            model_dir.exists(),
            str(model_dir),
        )
    )
    for path in required_model_files:
        model_rows.append((path.name, path.exists(), str(path)))
    _print_rows("structural_model", model_rows)

    failures = 0
    if args.dataset != "toy":
        failures += sum(1 for _, ok, _ in dataset_rows if not ok)
    failures += sum(1 for _, ok, _ in tool_rows if not ok)
    failures += sum(1 for _, ok, _ in model_rows if not ok)
    if args.strict and failures > 0:
        print(f"\n[summary] strict mode: {failures} missing prerequisites")
        return 1
    print(f"\n[summary] done (missing={failures})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
