#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SHARDS="${SHARDS:-8}"
PARALLEL="${PARALLEL:-4}"
DRY_RUN="${DRY_RUN:-0}"
RETRY_DISCARDED="${RETRY_DISCARDED:-0}"
LIMIT="${LIMIT:-}"
RUN_NONE="${RUN_NONE:-auto}"

PROJECT="${GOOGLE_CLOUD_PROJECT:-ucr-ursa-major-socal-lab}"
GEMINI_AGENT="${GEMINI_AGENT:-minisweagent_gemini3_flash}"
GEMINI_MODEL_KEY="${GEMINI_MODEL_KEY:-gemini3_flash}"
GEMINI_OUT_ROOT="${GEMINI_OUT_ROOT:-outputs/attacks/$GEMINI_MODEL_KEY}"
HELDOUT_FILE="${HELDOUT_FILE:-data/models/structural_misalignment/livecodebench_gemini/hetero_gnn/heldout_instance_ids.txt}"

run_cmd() {
  echo
  printf '[fcv89-datasets] run:'
  printf ' %q' "$@"
  echo
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[fcv89-datasets] missing required file: $path" >&2
    exit 1
  fi
}

set_gemini_env() {
  export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/cfg-semantic-grounding/gemini_adc.json}"
  export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$PROJECT}"
  export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
  export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
  export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
  export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-$GOOGLE_CLOUD_LOCATION}"
  require_file "$GOOGLE_APPLICATION_CREDENTIALS"
}

cd "$ROOT"
require_file "$PYTHON"
set_gemini_env

echo "[fcv89-datasets] agent=$GEMINI_AGENT model_key=$GEMINI_MODEL_KEY shards=$SHARDS parallel=$PARALLEL retry_discarded=$RETRY_DISCARDED run_none=$RUN_NONE"
echo "[fcv89-datasets] credentials=$GOOGLE_APPLICATION_CREDENTIALS project=$GOOGLE_CLOUD_PROJECT location=$VERTEXAI_LOCATION"

run_attack() {
  local attack="$1"
  local -a cmd=(
  "$PYTHON" -u scripts/run_attack_sharded.py
  --dataset livecodebench
  --split test
  --agent "$GEMINI_AGENT"
    --attack "$attack"
  --out-root "$GEMINI_OUT_ROOT"
  --mode full
  --shards "$SHARDS"
  --parallel "$PARALLEL"
)
  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi
  if [[ "$RETRY_DISCARDED" == "1" ]]; then
    cmd+=(--retry-discarded)
  fi
  run_cmd "${cmd[@]}"
}

clean_dataset="$ROOT/$GEMINI_OUT_ROOT/full/livecodebench_none/attack_dataset.jsonl"
if [[ "$RUN_NONE" == "1" ]] || { [[ "$RUN_NONE" == "auto" ]] && [[ ! -f "$clean_dataset" ]]; }; then
  run_attack none
else
  echo "[fcv89-datasets] reuse clean dataset: $clean_dataset"
fi

run_attack fcv_cwe89

run_cmd "$PYTHON" scripts/build_livecodebench_mixed_eval_datasets.py \
  --model-key "$GEMINI_MODEL_KEY" \
  --attacks fcv_cwe89 \
  --heldout-file "$HELDOUT_FILE"

echo
echo "[fcv89-datasets] completed"
echo "[fcv89-datasets] full mixed dataset: $ROOT/outputs/attacks/$GEMINI_MODEL_KEY/mixed/livecodebench_none_vs_fcv_cwe89/full/attack_dataset.jsonl"
echo "[fcv89-datasets] heldout mixed dataset: $ROOT/outputs/attacks/$GEMINI_MODEL_KEY/mixed/livecodebench_none_vs_fcv_cwe89/heldout/attack_dataset.jsonl"
