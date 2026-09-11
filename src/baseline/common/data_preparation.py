"""Row-only data preparation utilities for defense training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from src.common.rows import (
    load_rows,
    split_rows_by_code_base,
)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_training_data(
    *,
    rows_path: Path | str,
    output_dir: Path | str,
    train_ratio: float = 0.7,
    random_seed: int = 42,
    limit: int | None = None,
) -> Dict[str, Any]:
    """Split rows by code_base and write train/test JSONL files."""

    rows = load_rows(Path(rows_path), limit=limit)
    train_rows, test_rows, manifest = split_rows_by_code_base(
        rows,
        train_ratio=train_ratio,
        seed=random_seed,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", train_rows)
    _write_jsonl(out / "test.jsonl", test_rows)
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    metadata = {
        "source_format": "rows_v1",
        "rows_path": str(Path(rows_path)),
        "output_dir": str(out),
        "split_key": "code_base",
        "train_ratio": train_ratio,
        "random_seed": random_seed,
        "row_count": len(rows),
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "train_label_counts": manifest["train_label_counts"],
        "test_label_counts": manifest["test_label_counts"],
        "train_code_base_count": manifest["train_code_base_count"],
        "test_code_base_count": manifest["test_code_base_count"],
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata
