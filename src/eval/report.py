"""Aggregation helpers for run outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.common.artifact_store import ArtifactStore


def load_jsonl_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_defense_summary(rows: List[Dict], prompt_type: str, out_dir: Path) -> Dict:
    pt_rows = [row for row in rows if row.get("prompt_type") == prompt_type]
    accepted_ids = sorted(str(row["instance_id"]) for row in pt_rows if row.get("defense_decision") == "accept")
    rejected_ids = sorted(str(row["instance_id"]) for row in pt_rows if row.get("defense_decision") == "reject")
    summary = {
        "prompt_type": prompt_type,
        "total_instances": len(pt_rows),
        "accepted_instances": len(accepted_ids),
        "rejected_instances": len(rejected_ids),
        "accepted_ids": accepted_ids,
        "rejected_ids": rejected_ids,
    }
    (out_dir / f"defense_summary_{prompt_type}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def write_summary_csv(summary_path: Path, rows: List[Dict]) -> Path:
    store = ArtifactStore(summary_path.parent)
    thin_rows: List[Dict] = []
    for row in rows:
        thin_rows.append(
            {
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "instance_id": row.get("instance_id"),
                "agent_name": row.get("agent_name"),
                "attack_name": row.get("attack_name"),
                "baseline_name": row.get("baseline_name"),
                "fidelity_mode": row.get("fidelity_mode"),
                "defense_decision": row.get("defense_decision"),
                "tests_passed": row.get("tests_passed"),
                "apply_ok": row.get("apply_ok"),
                "apply_method": row.get("apply_method"),
                "apply_failure_reason_code": row.get("apply_failure_reason_code"),
                "attempt_count": row.get("attempt_count"),
                "static_findings_count": row.get("static_findings_count"),
                "runtime_sec": row.get("runtime_sec"),
            }
        )
    return store.write_csv(summary_path.name, thin_rows)
