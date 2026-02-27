"""Artifact writer helpers for reproducibility outputs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable


def atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def atomic_write_json(path: Path, payload: Dict) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, relpath: str, payload: Dict) -> Path:
        path = self.root / relpath
        return atomic_write_json(path, payload)

    def write_text(self, relpath: str, text: str) -> Path:
        path = self.root / relpath
        return atomic_write_text(path, text)

    def append_jsonl(self, relpath: str, row: Dict) -> Path:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return path

    def write_csv(self, relpath: str, rows: Iterable[Dict]) -> Path:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        rows_list = list(rows)
        if not rows_list:
            return atomic_write_text(path, "")
        headers = list(rows_list[0].keys())
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows_list)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return path
