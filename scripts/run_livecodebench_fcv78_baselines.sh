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
ATTACKS="${ATTACKS:-fcv_cwe78 swexploit}"
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
  export CFG_GEMINI_VERTEX_MAX_OUTPUT_TOKENS="${CFG_GEMINI_VERTEX_MAX_OUTPUT_TOKENS:-512}"
  export CFG_GEMINI_VERTEX_THINKING_BUDGET="${CFG_GEMINI_VERTEX_THINKING_BUDGET:-0}"
  export CFG_GEMINI_VERTEX_RESPONSE_MIME_TYPE="${CFG_GEMINI_VERTEX_RESPONSE_MIME_TYPE:-application/json}"
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
  local split="$3"
  printf '%s/outputs/attacks/%s/mixed/livecodebench_none_vs_%s/%s/attack_dataset.jsonl' "$ROOT" "$model_key" "$attack" "$split"
}

baseline_out() {
  local model_key="$1"
  local baseline="$2"
  local attack="$3"
  local split="$4"
  printf '%s/outputs/baselines/livecodebench/%s/%s/mixed_none_vs_%s_%s' "$ROOT" "$model_key" "$baseline" "$attack" "$split"
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
  local baseline
  baseline="$(resolved_baseline "$model_family" "$baseline_alias")"
  local split="full"
  if [[ "$baseline_alias" == "structural_misalignment" ]]; then
    split="heldout"
  fi
  local dataset
  dataset="$(attack_dataset "$model_key" "$attack" "$split")"
  require_file "$dataset"

  local out
  out="$(baseline_out "$model_key" "$baseline" "$attack" "$split")"
  local cmd=(
    "$PYTHON" -u "$ROOT/scripts/run_defense_sharded.py"
    --attack-results "$dataset"
    --baseline "$baseline"
    --fidelity-mode llm
    --out "$out"
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
  local attack baseline_alias
  for attack in $ATTACKS; do
    for baseline_alias in $BASELINES; do
      run_one "$model_family" "$model_key" "$attack" "$baseline_alias"
    done
  done
}

prepare_mixed_datasets() {
  local model_family="$1"
  local model_key="$2"
  local heldout_file="$3"
  if [[ "$model_family" == "gemini" ]]; then
    run_cmd \
      "$PYTHON" "$ROOT/scripts/build_livecodebench_mixed_eval_datasets.py" \
      --model-key "$model_key" \
      --heldout-file "$heldout_file" \
      --attacks $ATTACKS
  else
    echo "[fcv78-baselines] mixed dataset builder currently supports gemini attack naming only" >&2
    exit 2
  fi
}

main() {
  cd "$ROOT"
  require_file "$PYTHON"
  require_file "$ROOT/scripts/run_defense_sharded.py"

  if [[ "$RUN_GEMINI" == "1" ]]; then
    set_gemini_env
    gemini_heldout="$ROOT/data/models/structural_misalignment/livecodebench_gemini/hetero_gnn/heldout_instance_ids.txt"
    prepare_mixed_datasets gemini gemini3_flash "$gemini_heldout"
    run_model gemini gemini3_flash
  fi

  if [[ "$RUN_CLAUDE" == "1" ]]; then
    set_claude_env
    claude_heldout="$ROOT/data/models/structural_misalignment/livecodebench_claude37_sonnet_sweagent/hetero_gnn/heldout_instance_ids.txt"
    prepare_mixed_datasets claude claude37_sonnet_sweagent "$claude_heldout"
    run_model claude claude37_sonnet_sweagent
  fi

  echo
  echo "[fcv78-baselines] all requested baseline runs completed"
}

main "$@"
