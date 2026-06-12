#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SHARDS="${SHARDS:-32}"
PARALLEL="${PARALLEL:-32}"
DRY_RUN="${DRY_RUN:-0}"
RUN_GEMINI="${RUN_GEMINI:-1}"
RUN_CLAUDE="${RUN_CLAUDE:-1}"
RUN_IN_PARALLEL="${RUN_IN_PARALLEL:-1}"
ARCHIVE_EXISTING="${ARCHIVE_EXISTING:-0}"
RETRY_DISCARDED="${RETRY_DISCARDED:-0}"

PROJECT="${GOOGLE_CLOUD_PROJECT:-ucr-ursa-major-congliu-lab}"

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

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[fcv78-datasets] missing required file: $path" >&2
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
  export VERTEXAI_LOCATION="${CLAUDE_VERTEX_LOCATION:-us-east5}"
  export GOOGLE_CLOUD_LOCATION="${CLAUDE_VERTEX_LOCATION:-us-east5}"
  export ANTHROPIC_VERTEX_PROJECT_ID="${ANTHROPIC_VERTEX_PROJECT_ID:-$GOOGLE_CLOUD_PROJECT}"
  export ANTHROPIC_VERTEX_REGION="${ANTHROPIC_VERTEX_REGION:-$VERTEXAI_LOCATION}"
}

archive_existing_outputs() {
  local out_root="$1"
  local label="$2"
  local stamp archive_root final_dir shard_leaf
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  archive_root="$ROOT/outputs/attacks/_archived_fcv_cwe78/$stamp/$label"
  final_dir="$ROOT/$out_root/full/livecodebench_fcv_cwe78"
  if [[ -e "$final_dir" ]]; then
    mkdir -p "$archive_root/final"
    echo "[fcv78-datasets] archive $final_dir -> $archive_root/final/livecodebench_fcv_cwe78"
    mv "$final_dir" "$archive_root/final/livecodebench_fcv_cwe78"
  fi
  shopt -s nullglob
  for shard_leaf in "$ROOT/$out_root"/full/shards/shard_*/livecodebench_fcv_cwe78; do
    mkdir -p "$archive_root/shards/$(basename "$(dirname "$shard_leaf")")"
    echo "[fcv78-datasets] archive $shard_leaf -> $archive_root/shards/$(basename "$(dirname "$shard_leaf")")/livecodebench_fcv_cwe78"
    mv "$shard_leaf" "$archive_root/shards/$(basename "$(dirname "$shard_leaf")")/livecodebench_fcv_cwe78"
  done
  shopt -u nullglob
}

run_dataset() {
  local agent="$1"
  local out_root="$2"
  local cmd=(
    "$PYTHON" -u scripts/run_attack_sharded.py
    --dataset livecodebench
    --split test
    --agent "$agent"
    --attack fcv_cwe78
    --out-root "$out_root"
    --mode full
    --shards "$SHARDS"
    --parallel "$PARALLEL"
  )
  if [[ "$RETRY_DISCARDED" == "1" ]]; then
    cmd+=(--retry-discarded)
  fi
  run_cmd "${cmd[@]}"
}

cd "$ROOT"
require_file "$PYTHON"
set_common_vertex_env

if [[ "$ARCHIVE_EXISTING" == "1" ]]; then
  [[ "$RUN_GEMINI" == "1" ]] && archive_existing_outputs "$GEMINI_OUT_ROOT" "gemini3_flash"
  [[ "$RUN_CLAUDE" == "1" ]] && archive_existing_outputs "$CLAUDE_OUT_ROOT" "claude37_sonnet_sweagent"
fi

echo "[fcv78-datasets] shards=$SHARDS parallel=$PARALLEL run_in_parallel=$RUN_IN_PARALLEL"
echo "[fcv78-datasets] credentials=$GOOGLE_APPLICATION_CREDENTIALS project=$GOOGLE_CLOUD_PROJECT location=$VERTEXAI_LOCATION"

if [[ "$RUN_IN_PARALLEL" == "1" ]]; then
  pids=()
  if [[ "$RUN_GEMINI" == "1" ]]; then
    (set_gemini_env; run_dataset "$GEMINI_AGENT" "$GEMINI_OUT_ROOT") &
    pids+=("$!")
  fi
  if [[ "$RUN_CLAUDE" == "1" ]]; then
    (set_claude_env; run_dataset "$CLAUDE_AGENT" "$CLAUDE_OUT_ROOT") &
    pids+=("$!")
  fi
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
else
  if [[ "$RUN_GEMINI" == "1" ]]; then
    set_gemini_env
    run_dataset "$GEMINI_AGENT" "$GEMINI_OUT_ROOT"
  fi
  if [[ "$RUN_CLAUDE" == "1" ]]; then
    set_claude_env
    run_dataset "$CLAUDE_AGENT" "$CLAUDE_OUT_ROOT"
  fi
fi

echo
echo "[fcv78-datasets] all requested dataset runs completed"
