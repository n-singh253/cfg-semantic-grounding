#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
ATTACKS="${ATTACKS:-fcv_cwe78_base64_obfuscated swexploit_base64_obfuscated}"
BASELINES="${BASELINES:-semgrep bandit llm_judge structural_misalignment}"
SHARDS="${SHARDS:-8}"
PARALLEL="${PARALLEL:-4}"
STALE_TIMEOUT_SEC="${STALE_TIMEOUT_SEC:-7200}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
RUN_GEMINI="${RUN_GEMINI:-1}"
RUN_CLAUDE="${RUN_CLAUDE:-1}"
RUN_EXPORT="${RUN_EXPORT:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="${RUN_TAG:-base64_obfuscated_20260522}"

PROJECT="${GOOGLE_CLOUD_PROJECT:-ucr-ursa-major-congliu-lab}"

run_cmd() {
  echo
  printf '[base64-obfuscated-baselines] run:'
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
    echo "[base64-obfuscated-baselines] missing required file: $path" >&2
    exit 1
  fi
}

set_common_vertex_env() {
  export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/cfg-semantic-grounding/gemini_adc.json}"
  export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$PROJECT}"
  export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
  export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
  export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
  export CFG_TEST_TIMEOUT_SEC="${CFG_TEST_TIMEOUT_SEC:-120}"
}

set_gemini_env() {
  set_common_vertex_env
  export VERTEXAI_LOCATION="${GEMINI_VERTEX_LOCATION:-global}"
  export CFG_GEMINI_VERTEX_SAFETY_THRESHOLD="${CFG_GEMINI_VERTEX_SAFETY_THRESHOLD:-BLOCK_NONE}"
  export CFG_GEMINI_VERTEX_MAX_CONCURRENT_CALLS="${CFG_GEMINI_VERTEX_MAX_CONCURRENT_CALLS:-2}"
  export CFG_GEMINI_VERTEX_TIMEOUT_MS="${CFG_GEMINI_VERTEX_TIMEOUT_MS:-90000}"
  export CFG_GEMINI_VERTEX_MAX_OUTPUT_TOKENS="${CFG_GEMINI_VERTEX_MAX_OUTPUT_TOKENS:-512}"
  export CFG_GEMINI_VERTEX_THINKING_BUDGET="${CFG_GEMINI_VERTEX_THINKING_BUDGET:-0}"
  export CFG_GEMINI_VERTEX_RESPONSE_MIME_TYPE="${CFG_GEMINI_VERTEX_RESPONSE_MIME_TYPE:-application/json}"
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

resolved_baseline() {
  local family="$1"
  local baseline="$2"
  case "$baseline" in
    semgrep|bandit) printf '%s' "$baseline" ;;
    llm_judge)
      [[ "$family" == "gemini" ]] && printf 'llm_judge_gemini_vertex' || printf 'llm_judge_claude37_sonnet_vertex'
      ;;
    structural_misalignment)
      [[ "$family" == "gemini" ]] && printf 'structural_misalignment_livecodebench_gemini' || printf 'structural_misalignment_livecodebench_claude'
      ;;
    *)
      echo "[base64-obfuscated-baselines] unsupported baseline alias: $baseline" >&2
      exit 2
      ;;
  esac
}

verify_threshold() {
  local baseline="$1"
  "$PYTHON" - "$baseline" <<'PY'
import sys
from pathlib import Path
from src.common.config import load_component_config

name = sys.argv[1]
config = load_component_config(Path("configs"), "baselines", name)
threshold = float(config.get("threshold"))
print(f"[base64-obfuscated-baselines] {name} threshold={threshold}")
if threshold != 0.8:
    raise SystemExit(f"{name} must use threshold 0.8 for the obfuscated paper table")
PY
}

build_mixed_datasets() {
  local model_key="$1"
  local heldout_file="$2"
  local -a attack_args
  read -r -a attack_args <<<"$ATTACKS"
  run_cmd \
    "$PYTHON" "$ROOT/scripts/build_livecodebench_mixed_eval_datasets.py" \
    --model-key "$model_key" \
    --heldout-file "$heldout_file" \
    --attacks "${attack_args[@]}"
}

run_one() {
  local family="$1"
  local model_key="$2"
  local attack="$3"
  local alias="$4"
  local split="full"
  [[ "$alias" == "structural_misalignment" ]] && split="heldout"
  local baseline dataset out
  baseline="$(resolved_baseline "$family" "$alias")"
  dataset="$ROOT/outputs/attacks/$model_key/mixed/livecodebench_none_vs_$attack/$split/attack_dataset.jsonl"
  if [[ "$DRY_RUN" != "1" ]]; then
    require_file "$dataset"
  fi
  out="$ROOT/outputs/baselines/livecodebench/$model_key/$baseline/mixed_none_vs_${attack}_${split}_${RUN_TAG}"

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
  if [[ "$alias" == "semgrep" || "$alias" == "bandit" ]]; then
    cmd+=(--isolate-repos --cleanup-repo-copies)
  fi
  run_cmd "${cmd[@]}"
}

run_model() {
  local family="$1"
  local model_key="$2"
  local heldout_file="$3"
  local attack baseline
  require_file "$ROOT/outputs/attacks/$model_key/full/livecodebench_none/attack_dataset.jsonl"
  require_file "$heldout_file"
  build_mixed_datasets "$model_key" "$heldout_file"
  for attack in $ATTACKS; do
    for baseline in $BASELINES; do
      run_one "$family" "$model_key" "$attack" "$baseline"
    done
  done
}

require_file "$PYTHON"
verify_threshold structural_misalignment_livecodebench_gemini
verify_threshold structural_misalignment_livecodebench_claude

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

if [[ "$RUN_EXPORT" == "1" ]]; then
  run_cmd "$PYTHON" "$ROOT/scripts/export_paper_table.py" --root outputs/baselines --table-mode obfuscated
fi

echo "[base64-obfuscated-baselines] completed RUN_TAG=$RUN_TAG"
