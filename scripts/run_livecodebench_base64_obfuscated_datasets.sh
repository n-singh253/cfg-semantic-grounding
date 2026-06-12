#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
ATTACKS="${ATTACKS:-fcv_cwe78_base64_obfuscated swexploit_base64_obfuscated}"
SHARDS="${SHARDS:-16}"
PARALLEL="${PARALLEL:-4}"
DRY_RUN="${DRY_RUN:-0}"
RUN_GEMINI="${RUN_GEMINI:-1}"
RUN_CLAUDE="${RUN_CLAUDE:-1}"
RUN_IN_PARALLEL="${RUN_IN_PARALLEL:-0}"
RETRY_DISCARDED="${RETRY_DISCARDED:-0}"
ARCHIVE_EXISTING="${ARCHIVE_EXISTING:-0}"

PROJECT="${GOOGLE_CLOUD_PROJECT:-ucr-ursa-major-congliu-lab}"
GEMINI_AGENT="${GEMINI_AGENT:-minisweagent_gemini3_flash}"
GEMINI_OUT_ROOT="${GEMINI_OUT_ROOT:-outputs/attacks/gemini3_flash}"
CLAUDE_AGENT="${CLAUDE_AGENT:-sweagent_claude37_sonnet_vertex}"
CLAUDE_OUT_ROOT="${CLAUDE_OUT_ROOT:-outputs/attacks/claude37_sonnet_sweagent}"

run_cmd() {
  echo
  printf '[base64-obfuscated-datasets] run:'
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
    echo "[base64-obfuscated-datasets] missing required file: $path" >&2
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
  set_common_vertex_env
  export GOOGLE_CLOUD_LOCATION="${CLAUDE_VERTEX_LOCATION:-us-east5}"
  export VERTEXAI_LOCATION="${CLAUDE_VERTEX_LOCATION:-us-east5}"
  export ANTHROPIC_VERTEX_PROJECT_ID="${ANTHROPIC_VERTEX_PROJECT_ID:-$GOOGLE_CLOUD_PROJECT}"
  export ANTHROPIC_VERTEX_REGION="${ANTHROPIC_VERTEX_REGION:-$VERTEXAI_LOCATION}"
  export CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC="${CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC:-90}"
  export CFG_ANTHROPIC_VERTEX_MAX_CONCURRENT_CALLS="${CFG_ANTHROPIC_VERTEX_MAX_CONCURRENT_CALLS:-2}"
}

archive_attack_outputs() {
  local out_root="$1"
  local label="$2"
  local stamp archive_root attack final_dir shard_leaf
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  archive_root="$ROOT/outputs/attacks/_archived_base64_obfuscated/$stamp/$label"
  for attack in $ATTACKS; do
    final_dir="$ROOT/$out_root/full/livecodebench_$attack"
    if [[ -e "$final_dir" ]]; then
      mkdir -p "$archive_root/final"
      echo "[base64-obfuscated-datasets] archive $final_dir"
      mv "$final_dir" "$archive_root/final/"
    fi
    shopt -s nullglob
    for shard_leaf in "$ROOT/$out_root"/full/shards/shard_*/"livecodebench_$attack"; do
      mkdir -p "$archive_root/shards/$(basename "$(dirname "$shard_leaf")")"
      echo "[base64-obfuscated-datasets] archive $shard_leaf"
      mv "$shard_leaf" "$archive_root/shards/$(basename "$(dirname "$shard_leaf")")/"
    done
    shopt -u nullglob
  done
}

run_model_datasets() {
  local agent="$1"
  local out_root="$2"
  local attack
  for attack in $ATTACKS; do
    local -a cmd=(
      "$PYTHON" -u scripts/run_attack_sharded.py
      --dataset livecodebench
      --split test
      --agent "$agent"
      --attack "$attack"
      --out-root "$out_root"
      --mode full
      --shards "$SHARDS"
      --parallel "$PARALLEL"
    )
    if [[ "$RETRY_DISCARDED" == "1" ]]; then
      cmd+=(--retry-discarded)
    fi
    run_cmd "${cmd[@]}"
  done
}

require_file "$PYTHON"
set_common_vertex_env

if [[ "$ARCHIVE_EXISTING" == "1" ]]; then
  [[ "$RUN_GEMINI" == "1" ]] && archive_attack_outputs "$GEMINI_OUT_ROOT" "gemini3_flash"
  [[ "$RUN_CLAUDE" == "1" ]] && archive_attack_outputs "$CLAUDE_OUT_ROOT" "claude37_sonnet_sweagent"
fi

echo "[base64-obfuscated-datasets] attacks=$ATTACKS shards=$SHARDS parallel=$PARALLEL run_in_parallel=$RUN_IN_PARALLEL"
echo "[base64-obfuscated-datasets] project=$GOOGLE_CLOUD_PROJECT credentials=$GOOGLE_APPLICATION_CREDENTIALS"

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

echo "[base64-obfuscated-datasets] completed"
