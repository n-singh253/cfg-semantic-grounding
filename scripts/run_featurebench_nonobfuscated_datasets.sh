#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATASET="${DATASET:-featurebench_full}"
SPLIT="${SPLIT:-test}"
ATTACKS="${ATTACKS:-none fcv_cwe78 swexploit_gemini_vertex}"
SHARDS="${SHARDS:-8}"
PARALLEL="${PARALLEL:-4}"
DRY_RUN="${DRY_RUN:-0}"
RUN_GEMINI="${RUN_GEMINI:-1}"
RUN_CLAUDE="${RUN_CLAUDE:-1}"
RUN_IN_PARALLEL="${RUN_IN_PARALLEL:-0}"
RETRY_DISCARDED="${RETRY_DISCARDED:-0}"
LIMIT="${LIMIT:-}"
DATASET_DATA_PATH="${DATASET_DATA_PATH:-}"

PROJECT="${GOOGLE_CLOUD_PROJECT:-ucr-ursa-major-congliu-lab}"
GEMINI_AGENT="${GEMINI_AGENT:-minisweagent_gemini3_flash}"
GEMINI_OUT_ROOT="${GEMINI_OUT_ROOT:-outputs/attacks/gemini3_flash}"
CLAUDE_AGENT="${CLAUDE_AGENT:-sweagent_claude37_sonnet_anthropic}"
CLAUDE_OUT_ROOT="${CLAUDE_OUT_ROOT:-outputs/attacks/claude37_sonnet_sweagent}"

run_cmd() {
  echo
  printf '[featurebench-nonobfuscated-datasets] run:'
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
    echo "[featurebench-nonobfuscated-datasets] missing required file: $path" >&2
    exit 1
  fi
}

set_common_vertex_env() {
  export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/cfg-semantic-grounding/gemini_adc.json}"
  export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$PROJECT}"
  export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
  export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
  export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
  export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-$GOOGLE_CLOUD_LOCATION}"
  require_file "$GOOGLE_APPLICATION_CREDENTIALS"
}

set_gemini_env() {
  set_common_vertex_env
}

set_claude_env() {
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "[featurebench-nonobfuscated-datasets] ANTHROPIC_API_KEY is required for CLAUDE_AGENT=$CLAUDE_AGENT" >&2
    exit 1
  fi
  # SWExploit still uses the Gemini Vertex attack prompt config; keep Vertex env
  # available for that attack while routing SWE-agent's Claude calls through Anthropic.
  if [[ " $ATTACKS " == *" swexploit_gemini_vertex "* ]]; then
    set_common_vertex_env
  fi
}

run_model_datasets() {
  local agent="$1"
  local out_root="$2"
  local attack
  for attack in $ATTACKS; do
    local -a cmd=(
      "$PYTHON" -u scripts/run_attack_sharded.py
      --dataset "$DATASET"
      --split "$SPLIT"
      --agent "$agent"
      --attack "$attack"
      --out-root "$out_root"
      --mode full
      --shards "$SHARDS"
      --parallel "$PARALLEL"
    )
    if [[ -n "$LIMIT" ]]; then
      cmd+=(--limit "$LIMIT")
    fi
    if [[ -n "$DATASET_DATA_PATH" ]]; then
      cmd+=(--dataset-data-path "$DATASET_DATA_PATH")
    fi
    if [[ "$RETRY_DISCARDED" == "1" ]]; then
      cmd+=(--retry-discarded)
    fi
    run_cmd "${cmd[@]}"
  done
}

require_file "$PYTHON"
case "$DATASET" in
  featurebench_full) require_file "$ROOT/data/featurebench_full_attack_ready.jsonl" ;;
  featurebench|featurebench_lite) require_file "$ROOT/data/featurebench_lite_attack_ready.jsonl" ;;
esac

echo "[featurebench-nonobfuscated-datasets] dataset=$DATASET split=$SPLIT attacks=$ATTACKS shards=$SHARDS parallel=$PARALLEL run_in_parallel=$RUN_IN_PARALLEL"
echo "[featurebench-nonobfuscated-datasets] project=${GOOGLE_CLOUD_PROJECT:-$PROJECT} credentials=${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/cfg-semantic-grounding/gemini_adc.json}"

if [[ "$RUN_IN_PARALLEL" == "1" ]]; then
  pids=()
  if [[ "$RUN_GEMINI" == "1" ]]; then
    (set_gemini_env; run_model_datasets "$GEMINI_AGENT" "$GEMINI_OUT_ROOT") &
    pids+=("$!")
  fi
  if [[ "$RUN_CLAUDE" == "1" ]]; then
    (set_claude_env; run_model_datasets "$CLAUDE_AGENT" "$CLAUDE_OUT_ROOT") &
    pids+=("$!")
  fi
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
else
  if [[ "$RUN_GEMINI" == "1" ]]; then
    set_gemini_env
    run_model_datasets "$GEMINI_AGENT" "$GEMINI_OUT_ROOT"
  fi
  if [[ "$RUN_CLAUDE" == "1" ]]; then
    set_claude_env
    run_model_datasets "$CLAUDE_AGENT" "$CLAUDE_OUT_ROOT"
  fi
fi

echo "[featurebench-nonobfuscated-datasets] completed"
