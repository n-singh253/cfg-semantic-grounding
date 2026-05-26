#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODEL_KEY="claude37_sonnet_sweagent"
HELDOUT_FILE="$ROOT/data/models/structural_misalignment/livecodebench_claude37_sonnet_sweagent/hetero_gnn/heldout_instance_ids.txt"

SHARDS="${SHARDS:-8}"
PARALLEL="${PARALLEL:-4}"
STALE_TIMEOUT_SEC="${STALE_TIMEOUT_SEC:-7200}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="${RUN_TAG:-claude_regenerated_20260521}"
ATTACKS="${ATTACKS:-fcv_cwe78 swexploit}"
BASELINES="${BASELINES:-semgrep bandit llm_judge structural_misalignment}"

PROJECT="${GOOGLE_CLOUD_PROJECT:-ucr-ursa-major-congliu-lab}"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/cfg-semantic-grounding/gemini_adc.json}"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$PROJECT}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-us-east5}"
export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-east5}"
export ANTHROPIC_VERTEX_PROJECT_ID="${ANTHROPIC_VERTEX_PROJECT_ID:-$GOOGLE_CLOUD_PROJECT}"
export ANTHROPIC_VERTEX_REGION="${ANTHROPIC_VERTEX_REGION:-us-east5}"
export CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC="${CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC:-90}"
export CFG_ANTHROPIC_VERTEX_MAX_CONCURRENT_CALLS="${CFG_ANTHROPIC_VERTEX_MAX_CONCURRENT_CALLS:-2}"
export CFG_TEST_TIMEOUT_SEC="${CFG_TEST_TIMEOUT_SEC:-120}"

run_cmd() {
  echo
  printf '[claude-baselines] run:'
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
    echo "[claude-baselines] missing required file: $path" >&2
    exit 1
  fi
}

attack_leaf() {
  case "$1" in
    fcv_cwe78) printf 'livecodebench_fcv_cwe78' ;;
    swexploit) printf 'livecodebench_swexploit_gemini_vertex' ;;
    *)
      echo "[claude-baselines] unsupported attack alias: $1" >&2
      exit 2
      ;;
  esac
}

resolved_baseline() {
  case "$1" in
    semgrep|bandit) printf '%s' "$1" ;;
    llm_judge) printf 'llm_judge_claude37_sonnet_vertex' ;;
    structural_misalignment) printf 'structural_misalignment_livecodebench_claude' ;;
    *)
      echo "[claude-baselines] unsupported baseline alias: $1" >&2
      exit 2
      ;;
  esac
}

mixed_dataset() {
  local attack="$1"
  local split="$2"
  printf '%s/outputs/attacks/%s/mixed/livecodebench_none_vs_%s/%s/attack_dataset.jsonl' \
    "$ROOT" "$MODEL_KEY" "$attack" "$split"
}

baseline_out() {
  local baseline="$1"
  local attack="$2"
  local split="$3"
  printf '%s/outputs/baselines/livecodebench/%s/%s/mixed_none_vs_%s_%s_%s' \
    "$ROOT" "$MODEL_KEY" "$baseline" "$attack" "$split" "$RUN_TAG"
}

verify_inputs() {
  local attack leaf
  require_file "$PYTHON"
  require_file "$HELDOUT_FILE"
  require_file "$ROOT/outputs/attacks/$MODEL_KEY/full/livecodebench_none/attack_dataset.jsonl"
  for attack in $ATTACKS; do
    leaf="$(attack_leaf "$attack")"
    require_file "$ROOT/outputs/attacks/$MODEL_KEY/full/$leaf/attack_dataset.jsonl"
  done

  "$PYTHON" - <<'PY'
from pathlib import Path
from src.common.config import load_component_config

config = load_component_config(
    Path("configs"),
    "baselines",
    "structural_misalignment_livecodebench_claude",
)
threshold = float(config.get("threshold"))
print("[claude-baselines] structural threshold=", threshold)
if threshold != 0.8:
    raise SystemExit("structural_misalignment_livecodebench_claude must use threshold: 0.8")
PY
}

build_mixed_datasets() {
  local -a attack_args
  read -r -a attack_args <<<"$ATTACKS"
  run_cmd \
    "$PYTHON" "$ROOT/scripts/build_livecodebench_mixed_eval_datasets.py" \
    --model-key "$MODEL_KEY" \
    --heldout-file "$HELDOUT_FILE" \
    --attacks "${attack_args[@]}"
}

run_one() {
  local attack="$1"
  local baseline_alias="$2"
  local baseline split dataset out

  baseline="$(resolved_baseline "$baseline_alias")"
  split="full"
  if [[ "$baseline_alias" == "structural_misalignment" ]]; then
    split="heldout"
  fi
  dataset="$(mixed_dataset "$attack" "$split")"
  if [[ "$DRY_RUN" != "1" ]]; then
    require_file "$dataset"
  fi
  out="$(baseline_out "$baseline" "$attack" "$split")"

  local -a cmd=(
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

main() {
  local attack baseline
  verify_inputs
  build_mixed_datasets
  for attack in $ATTACKS; do
    for baseline in $BASELINES; do
      run_one "$attack" "$baseline"
    done
  done
  echo
  echo "[claude-baselines] completed RUN_TAG=$RUN_TAG"
}

main "$@"
