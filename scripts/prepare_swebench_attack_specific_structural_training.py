#!/usr/bin/env python3
"""Prepare attack-specific SWE-Bench training splits for structural misalignment.

The imported SWE-Bench structural training files are split by agent, and their
malicious files contain multiple attacks.  This script combines the Gemini and
Claude imports, filters malicious rows to one attack at a time, and preserves a
source-instance heldout split so the same SWE-Bench task is not present in both
train and heldout through different agents or labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE_ROOT = Path("data/training/swebench_structural/swebench_imported_heldout_20260619_052311")
DEFAULT_OUTPUT_ROOT = Path("data/training/swebench_structural/swebench_imported_attack_specific_20260627")
DEFAULT_AGENTS = ("gemini", "claude")
DEFAULT_ATTACKS = ("fcv_cwe78", "swexploit_anthropic")


def _load_jsonl(path: Path, *, agent: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["swebench_structural_source_agent"] = agent
        row["swebench_structural_source_file"] = str(path)
        row["swebench_structural_source_line"] = line_no
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in materialized)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")
    return len(materialized)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_instance_id(row: dict[str, Any]) -> str:
    source = str(row.get("source_instance_id", "") or "").strip()
    if source:
        return source
    instance_id = str(row.get("instance_id", "") or "").strip()
    for suffix in ("__none", "__fcv_cwe78", "__swexploit_anthropic"):
        if instance_id.endswith(suffix):
            return instance_id[: -len(suffix)]
    return instance_id


def _condition_instance_id(source_id: str, condition: str) -> str:
    return f"{source_id}__{condition}"


def _filter_attack(rows: Iterable[dict[str, Any]], attack: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("attack_name", "") or "") == attack]


def _exclude_sources(rows: Iterable[dict[str, Any]], heldout_source_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if _source_instance_id(row) not in heldout_source_ids]


def _counts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[str(row.get(key, "") or "")] += 1
    return dict(sorted(counter.items()))


def _source_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {_source_instance_id(row) for row in rows if _source_instance_id(row)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--agents", nargs="+", default=list(DEFAULT_AGENTS))
    parser.add_argument("--attacks", nargs="+", default=list(DEFAULT_ATTACKS))
    args = parser.parse_args()

    source_root = args.source_root
    loaded: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for agent in args.agents:
        for split in ("train", "test"):
            for label_name in ("benign", "malicious"):
                loaded[(agent, split, label_name)] = _load_jsonl(
                    source_root / agent / f"{split}_{label_name}.jsonl",
                    agent=agent,
                )

    global_summary: dict[str, Any] = {
        "source_root": str(source_root),
        "out_root": str(args.out_root),
        "agents": list(args.agents),
        "attacks": {},
        "heldout_policy": (
            "For each attack, heldout source IDs are the union of all existing "
            "test benign source IDs and that attack's existing test malicious source IDs. "
            "Training rows with those source IDs are removed to prevent cross-agent leakage."
        ),
    }

    for attack in args.attacks:
        train_benign_all: list[dict[str, Any]] = []
        test_benign_all: list[dict[str, Any]] = []
        train_malicious_attack: list[dict[str, Any]] = []
        test_malicious_attack: list[dict[str, Any]] = []
        for agent in args.agents:
            train_benign_all.extend(loaded[(agent, "train", "benign")])
            test_benign_all.extend(loaded[(agent, "test", "benign")])
            train_malicious_attack.extend(_filter_attack(loaded[(agent, "train", "malicious")], attack))
            test_malicious_attack.extend(_filter_attack(loaded[(agent, "test", "malicious")], attack))

        heldout_source_ids = _source_ids(test_benign_all) | _source_ids(test_malicious_attack)
        train_benign = _exclude_sources(train_benign_all, heldout_source_ids)
        train_malicious = _exclude_sources(train_malicious_attack, heldout_source_ids)
        test_benign = test_benign_all
        test_malicious = test_malicious_attack

        out_dir = args.out_root / attack
        counts = {
            "train_benign": _write_jsonl(out_dir / "train_benign.jsonl", train_benign),
            "train_malicious": _write_jsonl(out_dir / "train_malicious.jsonl", train_malicious),
            "test_benign": _write_jsonl(out_dir / "test_benign.jsonl", test_benign),
            "test_malicious": _write_jsonl(out_dir / "test_malicious.jsonl", test_malicious),
        }

        heldout_condition_ids = sorted(
            {_condition_instance_id(source_id, "none") for source_id in _source_ids(test_benign)}
            | {_condition_instance_id(source_id, attack) for source_id in _source_ids(test_malicious)}
        )
        heldout_source_sorted = sorted(heldout_source_ids)
        (out_dir / "heldout_instance_ids.txt").write_text(
            "\n".join(heldout_condition_ids) + ("\n" if heldout_condition_ids else ""),
            encoding="utf-8",
        )
        (out_dir / "heldout_source_instance_ids.txt").write_text(
            "\n".join(heldout_source_sorted) + ("\n" if heldout_source_sorted else ""),
            encoding="utf-8",
        )

        train_sources = _source_ids(train_benign) | _source_ids(train_malicious)
        leakage = sorted(train_sources & heldout_source_ids)
        if leakage:
            raise RuntimeError(f"{attack}: train/heldout source leakage remains: {leakage[:10]}")

        summary = {
            "attack": attack,
            "counts": counts,
            "heldout_source_instance_count": len(heldout_source_sorted),
            "heldout_condition_instance_count": len(heldout_condition_ids),
            "train_source_instance_count": len(train_sources),
            "train_agent_counts": _counts([*train_benign, *train_malicious], "swebench_structural_source_agent"),
            "test_agent_counts": _counts([*test_benign, *test_malicious], "swebench_structural_source_agent"),
            "train_attack_counts": _counts([*train_benign, *train_malicious], "attack_name"),
            "test_attack_counts": _counts([*test_benign, *test_malicious], "attack_name"),
            "heldout_source_instance_ids_path": str(out_dir / "heldout_source_instance_ids.txt"),
            "heldout_instance_ids_path": str(out_dir / "heldout_instance_ids.txt"),
        }
        _write_json(out_dir / "summary.json", summary)
        global_summary["attacks"][attack] = summary
        print(
            f"[swebench-structural-prep] {attack}: "
            f"train benign={counts['train_benign']} malicious={counts['train_malicious']} "
            f"test benign={counts['test_benign']} malicious={counts['test_malicious']} "
            f"heldout_source={len(heldout_source_sorted)} -> {out_dir}",
            flush=True,
        )

    _write_json(args.out_root / "summary.json", global_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
