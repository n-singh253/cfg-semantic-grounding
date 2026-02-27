#!/usr/bin/env python3
"""Thin wrapper for run_matrix."""

import sys

from src.eval.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run_matrix", *sys.argv[1:]]))
