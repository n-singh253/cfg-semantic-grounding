#!/usr/bin/env python3
"""Merge repaired attack rows into a finalized attack_dataset.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common.artifact_store import atomic_write_json, atomic_write_text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(path)
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Original finalized attack_dataset.jsonl")
    parser.add_argument("--repair", required=True, help="Repair finalized attack_dataset.jsonl")
    parser.add_argument("--out", required=True, help="Output attack_dataset.jsonl")
    parser.add_argument("--expected-id-file", default=None, help="Optional ids expected to be replaced")
    args = parser.parse_args()

    base_path = Path(args.base)
    repair_path = Path(args.repair)
    out_path = Path(args.out)
    expected_ids = read_ids(Path(args.expected_id_file) if args.expected_id_file else None)

    base_rows = read_jsonl(base_path)
    repair_rows = read_jsonl(repair_path)
    repairs_by_id = {
        str(row.get("instance_id", "")): row
        for row in repair_rows
        if str(row.get("instance_id", "")).strip()
    }
    if expected_ids:
        repairs_by_id = {
            instance_id: row
            for instance_id, row in repairs_by_id.items()
            if instance_id in expected_ids
        }

    replaced = 0
    output_rows: list[dict[str, Any]] = []
    seen_base_ids: set[str] = set()
    for row in base_rows:
        instance_id = str(row.get("instance_id", ""))
        seen_base_ids.add(instance_id)
        if instance_id in repairs_by_id:
            output_rows.append(repairs_by_id[instance_id])
            replaced += 1
        else:
            output_rows.append(row)

    missing_from_base = sorted(expected_ids.difference(seen_base_ids)) if expected_ids else []
    missing_repairs = sorted(expected_ids.difference(repairs_by_id)) if expected_ids else []

    text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in output_rows)
    atomic_write_text(out_path, text + ("\n" if text else ""))
    summary = {
        "base": str(base_path),
        "repair": str(repair_path),
        "out": str(out_path),
        "base_rows": len(base_rows),
        "repair_rows": len(repair_rows),
        "repair_ids": len(repairs_by_id),
        "expected_ids": len(expected_ids),
        "replaced_rows": replaced,
        "missing_from_base": missing_from_base,
        "missing_repairs": missing_repairs,
    }
    atomic_write_json(out_path.parent / "merge_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if expected_ids and missing_repairs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
