#!/usr/bin/env python3
"""Build labeled clean-vs-attack SWE-Bench eval datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ATTACKS = {
    "fcv_cwe78",
    "fcv_cwe78_base64_obfuscated",
    "swexploit",
    "swexploit_gemini_vertex",
    "swexploit_base64_obfuscated",
    "swexploit_gemini",
    "fcv_gemini",
}


def _attack_dir(dataset: str, attack: str) -> str:
    return f"{dataset}_{attack}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _heldout_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _tag_row(row: dict[str, Any], *, condition: str, label: int) -> dict[str, Any]:
    original_instance_id = str(row.get("instance_id", "") or "")
    tagged = dict(row)
    tagged["source_instance_id"] = original_instance_id
    tagged["instance_id"] = f"{original_instance_id}__{condition}"
    tagged["graph_label"] = int(label)
    tagged["mixed_eval_condition"] = condition
    tagged["mixed_eval_label"] = int(label)
    tagged["mixed_eval_source_attack_name"] = str(row.get("attack_name", "") or "")
    return tagged


def _build_split(
    *,
    clean_rows: list[dict[str, Any]],
    attack_rows: list[dict[str, Any]],
    attack: str,
    heldout: set[str] | None,
) -> list[dict[str, Any]]:
    if heldout is not None:
        clean_rows = [row for row in clean_rows if str(row.get("instance_id", "") or "") in heldout]
        attack_rows = [row for row in attack_rows if str(row.get("instance_id", "") or "") in heldout]
    mixed = [_tag_row(row, condition="none", label=0) for row in clean_rows]
    mixed.extend(_tag_row(row, condition=attack, label=1) for row in attack_rows)
    return mixed


def _summary(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    clean_path: Path,
    attack_path: Path,
    attack: str,
    split: str,
    heldout_file: Path | None,
) -> dict[str, Any]:
    labels = Counter(int(row.get("graph_label", 0)) for row in rows)
    conditions = Counter(str(row.get("mixed_eval_condition", "")) for row in rows)
    return {
        "dataset": dataset,
        "attack": attack,
        "split": split,
        "rows": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "condition_counts": dict(sorted(conditions.items())),
        "clean_source": str(clean_path),
        "attack_source": str(attack_path),
        "heldout_file": str(heldout_file) if heldout_file else "",
        "heldout_policy": "filter by heldout_file" if heldout_file else "all rows",
        "instance_id_policy": "suffixed original instance_id; original retained as source_instance_id",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="swebench_lite")
    parser.add_argument("--model-key", default="gemini3_flash")
    parser.add_argument("--attacks", nargs="+", default=["fcv_cwe78_base64_obfuscated", "swexploit_base64_obfuscated"])
    parser.add_argument("--heldout-file", type=Path, default=None)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/attacks"))
    args = parser.parse_args()

    model_root = args.outputs_root / args.model_key
    clean_path = model_root / "full" / _attack_dir(args.dataset, "none") / "attack_dataset.jsonl"
    clean_rows = _load_jsonl(clean_path)
    heldout = _heldout_ids(args.heldout_file)

    for attack in args.attacks:
        if attack not in ATTACKS:
            raise ValueError(f"Unknown attack {attack!r}; known: {sorted(ATTACKS)}")
        attack_path = model_root / "full" / _attack_dir(args.dataset, attack) / "attack_dataset.jsonl"
        attack_rows = _load_jsonl(attack_path)
        out_base = model_root / "mixed" / f"{args.dataset}_none_vs_{attack}"

        full_rows = _build_split(clean_rows=clean_rows, attack_rows=attack_rows, attack=attack, heldout=None)
        full_path = out_base / "full" / "attack_dataset.jsonl"
        _write_jsonl(full_path, full_rows)
        _write_json(
            out_base / "full" / "summary.json",
            _summary(
                full_rows,
                dataset=args.dataset,
                clean_path=clean_path,
                attack_path=attack_path,
                attack=attack,
                split="full",
                heldout_file=None,
            ),
        )

        heldout_rows = _build_split(clean_rows=clean_rows, attack_rows=attack_rows, attack=attack, heldout=heldout)
        heldout_path = out_base / "heldout" / "attack_dataset.jsonl"
        _write_jsonl(heldout_path, heldout_rows)
        _write_json(
            out_base / "heldout" / "summary.json",
            _summary(
                heldout_rows,
                dataset=args.dataset,
                clean_path=clean_path,
                attack_path=attack_path,
                attack=attack,
                split="heldout",
                heldout_file=args.heldout_file,
            ),
        )

        print(f"[swebench-mixed-datasets] {attack} full={len(full_rows)} -> {full_path}")
        print(f"[swebench-mixed-datasets] {attack} heldout={len(heldout_rows)} -> {heldout_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
