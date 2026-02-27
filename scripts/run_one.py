#!/usr/bin/env python3
"""Thin wrapper for run_one."""

import sys

from src.eval.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run_one", *sys.argv[1:]]))
