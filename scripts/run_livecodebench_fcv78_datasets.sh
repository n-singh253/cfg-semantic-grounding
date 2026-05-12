#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SHARDS="${SHARDS:-8}"
PARALLEL="${PARALLEL:-4}"
DRY_RUN="${DRY_RUN:-0}"

GEMINI_AGENT="${GEMINI_AGENT:-minisweagent_gemini3_flash}"
GEMINI_OUT_ROOT="${GEMINI_OUT_ROOT:-outputs/attacks/gemini3_flash}"
CLAUDE_AGENT="${CLAUDE_AGENT:-sweagent_claude37_sonnet_vertex}"
CLAUDE_OUT_ROOT="${CLAUDE_OUT_ROOT:-outputs/attacks/claude37_sonnet_sweagent}"

run_cmd() {
  echo
  printf '[fcv78-datasets] run:'
  printf ' %q' "$@"
  echo
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

cd "$ROOT"

run_cmd "$PYTHON" -u scripts/run_attack_sharded.py \
  --dataset livecodebench \
  --split test \
  --agent "$GEMINI_AGENT" \
  --attack fcv_cwe78 \
  --out-root "$GEMINI_OUT_ROOT" \
  --mode full \
  --shards "$SHARDS" \
  --parallel "$PARALLEL"

run_cmd "$PYTHON" -u scripts/run_attack_sharded.py \
  --dataset livecodebench \
  --split test \
  --agent "$CLAUDE_AGENT" \
  --attack fcv_cwe78 \
  --out-root "$CLAUDE_OUT_ROOT" \
  --mode full \
  --shards "$SHARDS" \
  --parallel "$PARALLEL"
