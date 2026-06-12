#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATASET="${DATASET:-swebench_lite}"
ATTACKS="${ATTACKS:-fcv_cwe78 swexploit_gemini_vertex}"
RUN_GEMINI="${RUN_GEMINI:-1}"
RUN_CLAUDE="${RUN_CLAUDE:-1}"
GEMINI_MODEL_KEY="${GEMINI_MODEL_KEY:-gemini3_flash}"
CLAUDE_MODEL_KEY="${CLAUDE_MODEL_KEY:-claude37_sonnet_sweagent}"
HELDOUT_FILE="${HELDOUT_FILE:-}"

run_builder() {
  local model_key="$1"
  local -a cmd=(
    "$PYTHON" scripts/build_swebench_mixed_eval_datasets.py
    --dataset "$DATASET"
    --model-key "$model_key"
    --attacks $ATTACKS
  )
  if [[ -n "$HELDOUT_FILE" ]]; then
    cmd+=(--heldout-file "$HELDOUT_FILE")
  fi
  echo
  printf '[swebench-mixed] run:'
  printf ' %q' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

if [[ "$RUN_GEMINI" == "1" ]]; then
  run_builder "$GEMINI_MODEL_KEY"
fi

if [[ "$RUN_CLAUDE" == "1" ]]; then
  run_builder "$CLAUDE_MODEL_KEY"
fi
