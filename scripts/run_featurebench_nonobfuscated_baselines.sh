#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
GEMINI_MODEL_KEY="${GEMINI_MODEL_KEY:-gemini3_flash_socal}"
CLAUDE_MODEL_KEY="${CLAUDE_MODEL_KEY:-claude_sonnet46_socal}"
MODEL_DIR="${MODEL_DIR:-$ROOT/data/models/structural_misalignment/featurebench_all/hetero_gnn}"
HELDOUT_FILE="${HELDOUT_FILE:-$MODEL_DIR/heldout_instance_ids.txt}"
ATTACKS="${ATTACKS:-fcv_cwe78 swexploit_anthropic}"
# Ours is evaluated separately with the improved FeatureBench v2 GNN after
# these non-trained baselines finish.
BASELINES="${BASELINES:-semgrep bandit llm_judge}"
RUN_GEMINI="${RUN_GEMINI:-1}"
RUN_CLAUDE="${RUN_CLAUDE:-1}"
SHARDS="${SHARDS:-8}"
PARALLEL="${PARALLEL:-4}"
STALE_TIMEOUT_SEC="${STALE_TIMEOUT_SEC:-7200}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  echo
  printf '[featurebench-baselines] run:'
  printf ' %q' "$@"
  echo
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[featurebench-baselines] missing required file: $1" >&2
    exit 1
  fi
}

set_common_vertex_env() {
  export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/cfg-semantic-grounding/gemini_adc.json}"
  export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-ucr-ursa-major-socal-lab}"
  export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
  export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
  export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
  export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-$GOOGLE_CLOUD_LOCATION}"
}

set_gemini_env() {
  set_common_vertex_env
  export CFG_GEMINI_VERTEX_SAFETY_THRESHOLD="${CFG_GEMINI_VERTEX_SAFETY_THRESHOLD:-BLOCK_NONE}"
  export CFG_GEMINI_VERTEX_MAX_CONCURRENT_CALLS="${CFG_GEMINI_VERTEX_MAX_CONCURRENT_CALLS:-2}"
  export CFG_GEMINI_VERTEX_TIMEOUT_MS="${CFG_GEMINI_VERTEX_TIMEOUT_MS:-90000}"
  export CFG_GEMINI_VERTEX_THINKING_BUDGET="${CFG_GEMINI_VERTEX_THINKING_BUDGET:-0}"
}

set_claude_env() {
  set_common_vertex_env
  export ANTHROPIC_VERTEX_PROJECT_ID="${ANTHROPIC_VERTEX_PROJECT_ID:-$GOOGLE_CLOUD_PROJECT}"
  export ANTHROPIC_VERTEX_REGION="${ANTHROPIC_VERTEX_REGION:-$VERTEXAI_LOCATION}"
  export CLAUDE_VERTEX_LOCATION="${CLAUDE_VERTEX_LOCATION:-$VERTEXAI_LOCATION}"
  export CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC="${CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC:-90}"
  export CFG_ANTHROPIC_VERTEX_MAX_CONCURRENT_CALLS="${CFG_ANTHROPIC_VERTEX_MAX_CONCURRENT_CALLS:-2}"
}

resolved_baseline() {
  local family="$1"
  local alias="$2"
  case "$alias" in
    semgrep|bandit)
      printf '%s' "$alias"
      ;;
    llm_judge)
      if [[ "$family" == "gemini" ]]; then
        printf 'llm_judge_gemini_vertex'
      else
        printf 'llm_judge_claude37_sonnet_vertex'
      fi
      ;;
    structural_misalignment)
      printf 'structural_misalignment_featurebench_all_trained'
      ;;
    *)
      echo "[featurebench-baselines] unknown baseline alias: $alias" >&2
      exit 2
      ;;
  esac
}

build_mixed() {
  local model_key="$1"
  run_cmd "$PYTHON" scripts/build_featurebench_mixed_eval_datasets.py \
    --dataset featurebench_full \
    --model-key "$model_key" \
    --heldout-file "$HELDOUT_FILE" \
    --attacks $ATTACKS
}

run_model() {
  local family="$1"
  local model_key="$2"
  local attack alias baseline split dataset out
  for attack in $ATTACKS; do
    for alias in $BASELINES; do
      split="full"
      if [[ "$alias" == "structural_misalignment" ]]; then
        split="heldout"
      fi
      dataset="$ROOT/outputs/attacks/$model_key/mixed/featurebench_full_none_vs_$attack/$split/attack_dataset.jsonl"
      require_file "$dataset"
      baseline="$(resolved_baseline "$family" "$alias")"
      out="$ROOT/outputs/baselines/featurebench_nonobfuscated/$model_key/$attack/$baseline"
      run_cmd "$PYTHON" -u scripts/run_defense_sharded.py \
        --attack-results "$dataset" \
        --baseline "$baseline" \
        --fidelity-mode llm \
        --out "$out" \
        --shards "$SHARDS" \
        --parallel "$PARALLEL" \
        --stale-timeout-sec "$STALE_TIMEOUT_SEC" \
        --poll-interval-sec "$POLL_INTERVAL_SEC" \
        --isolate-repos \
        --cleanup-repo-copies
    done
  done
}

main() {
  require_file "$PYTHON"
  require_file "$MODEL_DIR/model.pt"
  require_file "$MODEL_DIR/metadata.json"
  require_file "$MODEL_DIR/graph_cache_index.json"
  require_file "$HELDOUT_FILE"

  if [[ "$RUN_GEMINI" == "1" ]]; then
    set_gemini_env
    build_mixed "$GEMINI_MODEL_KEY"
    run_model gemini "$GEMINI_MODEL_KEY"
  fi
  if [[ "$RUN_CLAUDE" == "1" ]]; then
    set_claude_env
    build_mixed "$CLAUDE_MODEL_KEY"
    run_model claude "$CLAUDE_MODEL_KEY"
  fi

  echo
  echo '[featurebench-baselines] all requested runs completed'
}

main "$@"
