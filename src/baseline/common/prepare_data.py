#!/usr/bin/env python3
"""Wrapper for data preparation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline.common.data_preparation import prepare_training_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for defense models from evaluation results"
    )
    
    parser.add_argument(
        "--benign-results",
        type=Path,
        required=True,
        help="Path to results.jsonl with benign instances (attack_name='none')"
    )
    
    parser.add_argument(
        "--malicious-results",
        type=Path,
        required=True,
        help="Path to results.jsonl with malicious instances (e.g., attack_name='swexploit')"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save train.jsonl and test.jsonl"
    )
    
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="stratified_instance",
        choices=["instance_level", "stratified_instance", "random"],
        help="Train/test split strategy (default: stratified_instance)"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of instances for training (default: 0.8)"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional limit on instances per class"
    )
    
    args = parser.parse_args()
    
    if not args.benign_results.exists():
        print(f"Error: Benign results file not found: {args.benign_results}", file=sys.stderr)
        sys.exit(1)
    
    if not args.malicious_results.exists():
        print(f"Error: Malicious results file not found: {args.malicious_results}", file=sys.stderr)
        sys.exit(1)
    
    if not 0 < args.train_ratio < 1:
        print(f"Error: Train ratio must be between 0 and 1, got: {args.train_ratio}", file=sys.stderr)
        sys.exit(1)
    
    metadata = prepare_training_data(
        benign_results_path=args.benign_results,
        malicious_results_path=args.malicious_results,
        output_dir=args.output_dir,
        split_strategy=args.split_strategy,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        limit_per_class=args.limit_per_class,
    )

if __name__ == "__main__":
    main()
