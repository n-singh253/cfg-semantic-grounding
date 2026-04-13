#!/usr/bin/env python3
"""Create a local demo GNN bundle for structural_misalignment canary runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize demo structural GNN bundle")
    parser.add_argument(
        "--out",
        default="data/models/structural_misalignment/hetero_gnn",
        help="Output model bundle directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"torch is required: {exc}") from exc

    from src.baseline.structural_misalignment.models.gnn import HeteroGraphClassifier

    args = parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parents[1] / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    model = HeteroGraphClassifier(input_dim=768, hidden_dim=args.hidden_dim, dropout=args.dropout)
    torch.save(model.state_dict(), out_dir / "model.pt")

    metadata = {
        "gnn_model_type": "hetero_sage",
        "input_dim": 768,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "embedding_model_name": "microsoft/codebert-base",
        "embedding_pooling": "mean",
        "decision_policy_default": "reject_if_score_ge_threshold",
        "notes": "Demo local structural_misalignment GNN bundle for canary harness runs.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[init-structural-model] wrote GNN bundle to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
