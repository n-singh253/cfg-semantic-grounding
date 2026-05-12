#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SHARDS="${SHARDS:-8}"
PARALLEL="${PARALLEL:-4}"
STALE_TIMEOUT_SEC="${STALE_TIMEOUT_SEC:-7200}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
DRY_RUN="${DRY_RUN:-0}"
RUN_GEMINI="${RUN_GEMINI:-1}"
RUN_CLAUDE="${RUN_CLAUDE:-1}"
ATTACKS="${ATTACKS:-none fcv_cwe78 swexploit}"
BASELINES="${BASELINES:-semgrep bandit llm_judge structural_misalignment}"

PROJECT="${GOOGLE_CLOUD_PROJECT:-ucr-ursa-major-congliu-lab}"

run_cmd() {
  echo
  printf '[fcv78-baselines] run:'
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
    echo "[fcv78-baselines] missing required file: $path" >&2
    exit 1
  fi
}

set_common_vertex_env() {
  export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/cfg-semantic-grounding/gemini_adc.json}"
  export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$PROJECT}"
  export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
  export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
  export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
}

set_gemini_env() {
  set_common_vertex_env
  export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-global}"
  export CFG_GEMINI_VERTEX_SAFETY_THRESHOLD="${CFG_GEMINI_VERTEX_SAFETY_THRESHOLD:-BLOCK_NONE}"
  export CFG_GEMINI_VERTEX_MAX_CONCURRENT_CALLS="${CFG_GEMINI_VERTEX_MAX_CONCURRENT_CALLS:-2}"
  export CFG_GEMINI_VERTEX_TIMEOUT_MS="${CFG_GEMINI_VERTEX_TIMEOUT_MS:-90000}"
  export CFG_GEMINI_VERTEX_MAX_OUTPUT_TOKENS="${CFG_GEMINI_VERTEX_MAX_OUTPUT_TOKENS:-4096}"
}

set_claude_env() {
  set_common_vertex_env
  export VERTEXAI_LOCATION="us-east5"
  export ANTHROPIC_VERTEX_PROJECT_ID="${ANTHROPIC_VERTEX_PROJECT_ID:-$GOOGLE_CLOUD_PROJECT}"
  export ANTHROPIC_VERTEX_REGION="${ANTHROPIC_VERTEX_REGION:-us-east5}"
  export CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC="${CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC:-90}"
  export CFG_ANTHROPIC_VERTEX_MAX_CONCURRENT_CALLS="${CFG_ANTHROPIC_VERTEX_MAX_CONCURRENT_CALLS:-2}"
}

attack_dataset() {
  local model_key="$1"
  local attack="$2"
  local dataset_attack="$attack"
  if [[ "$attack" == "swexploit" ]]; then
    dataset_attack="swexploit_gemini_vertex"
  fi
  printf '%s/outputs/attacks/%s/full/livecodebench_%s/attack_dataset.jsonl' "$ROOT" "$model_key" "$dataset_attack"
}

baseline_out() {
  local model_key="$1"
  local baseline="$2"
  local attack="$3"
  printf '%s/outputs/baselines/livecodebench/%s/%s/%s' "$ROOT" "$model_key" "$baseline" "$attack"
}

resolved_baseline() {
  local model_family="$1"
  local baseline="$2"
  case "$baseline" in
    semgrep|bandit)
      printf '%s' "$baseline"
      ;;
    llm_judge)
      if [[ "$model_family" == "gemini" ]]; then
        printf 'llm_judge_gemini_vertex'
      else
        printf 'llm_judge_claude37_sonnet_vertex'
      fi
      ;;
    structural_misalignment)
      if [[ "$model_family" == "gemini" ]]; then
        printf 'structural_misalignment_livecodebench_gemini'
      else
        printf 'structural_misalignment_livecodebench_claude'
      fi
      ;;
    *)
      echo "[fcv78-baselines] unknown baseline alias: $baseline" >&2
      exit 2
      ;;
  esac
}

run_one() {
  local model_family="$1"
  local model_key="$2"
  local attack="$3"
  local baseline_alias="$4"
  local heldout_file="$5"
  local baseline
  baseline="$(resolved_baseline "$model_family" "$baseline_alias")"
  local dataset
  dataset="$(attack_dataset "$model_key" "$attack")"
  require_file "$dataset"
  require_file "$heldout_file"

  local out
  out="$(baseline_out "$model_key" "$baseline" "$attack")"
  local cmd=(
    "$PYTHON" -u "$ROOT/scripts/run_defense_sharded.py"
    --attack-results "$dataset"
    --baseline "$baseline"
    --fidelity-mode llm
    --out "$out"
    --instance-id-file "$heldout_file"
    --shards "$SHARDS"
    --parallel "$PARALLEL"
    --stale-timeout-sec "$STALE_TIMEOUT_SEC"
    --poll-interval-sec "$POLL_INTERVAL_SEC"
  )

  if [[ "$baseline_alias" == "semgrep" || "$baseline_alias" == "bandit" ]]; then
    cmd+=(--isolate-repos --cleanup-repo-copies)
  fi

  run_cmd "${cmd[@]}"
}

run_model() {
  local model_family="$1"
  local model_key="$2"
  local heldout_file="$3"
  local attack baseline_alias
  for attack in $ATTACKS; do
    for baseline_alias in $BASELINES; do
      run_one "$model_family" "$model_key" "$attack" "$baseline_alias" "$heldout_file"
    done
  done
}

main() {
  cd "$ROOT"
  require_file "$PYTHON"
  require_file "$ROOT/scripts/run_defense_sharded.py"

  if [[ "$RUN_GEMINI" == "1" ]]; then
    set_gemini_env
    run_model \
      gemini \
      gemini3_flash \
      "$ROOT/data/models/structural_misalignment/livecodebench_gemini/hetero_gnn/heldout_instance_ids.txt"
  fi

  if [[ "$RUN_CLAUDE" == "1" ]]; then
    set_claude_env
    run_model \
      claude \
      claude37_sonnet_sweagent \
      "$ROOT/data/models/structural_misalignment/livecodebench_claude37_sonnet_sweagent/hetero_gnn/heldout_instance_ids.txt"
  fi

  echo
  echo "[fcv78-baselines] all requested baseline runs completed"
}

main "$@"
