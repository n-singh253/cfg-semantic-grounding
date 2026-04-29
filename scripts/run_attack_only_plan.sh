#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" && -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
AGENT="${AGENT:-minisweagent_gemini3_flash}"
SMOKE_LIMIT="${SMOKE_LIMIT:-3}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/run_attack_only_plan.sh [--smoke|--full]

Environment:
  GOOGLE_APPLICATION_CREDENTIALS  Path to ignored ADC JSON.
  GOOGLE_CLOUD_PROJECT           Vertex project id.
  GOOGLE_CLOUD_LOCATION          Vertex location, defaults to global.
  GOOGLE_GENAI_USE_VERTEXAI       Should be true.
  PYTHON                          Python executable, defaults to python3.
  AGENT                           Agent config, defaults to minisweagent_gemini3_flash.
  SMOKE_LIMIT                     Smoke row count, defaults to 3.
  PYTHONUNBUFFERED                Defaults to 1 so runner output appears promptly.
EOF
}

fmt_duration() {
  local seconds="$1"
  local hours=$((seconds / 3600))
  local minutes=$(((seconds % 3600) / 60))
  local secs=$((seconds % 60))
  if (( hours > 0 )); then
    printf "%dh%02dm%02ds" "$hours" "$minutes" "$secs"
  else
    printf "%dm%02ds" "$minutes" "$secs"
  fi
}

estimate_instances() {
  local dataset="$1"
  local configured_limit="$2"
  if [[ "$MODE" == "smoke" ]]; then
    printf "%s" "$configured_limit"
    return
  fi
  case "$dataset" in
    featurebench_full)
      if [[ -f data/featurebench_full.jsonl ]]; then
        wc -l < data/featurebench_full.jsonl | tr -d ' '
      else
        printf "unknown"
      fi
      ;;
    livecodebench)
      if [[ -f data/livecodebench_code_generation_lite_release_latest.jsonl ]]; then
        wc -l < data/livecodebench_code_generation_lite_release_latest.jsonl | tr -d ' '
      else
        printf "unknown"
      fi
      ;;
    *)
      printf "unknown"
      ;;
  esac
}

MODE="smoke"
if [[ "${1:-}" == "--full" ]]; then
  MODE="full"
elif [[ "${1:-}" == "--smoke" || "${1:-}" == "" ]]; then
  MODE="smoke"
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
else
  usage
  exit 2
fi

: "${GOOGLE_APPLICATION_CREDENTIALS:?Set GOOGLE_APPLICATION_CREDENTIALS first.}"
: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT first.}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-$GOOGLE_CLOUD_LOCATION}"
export MSWEA_MODEL_NAME="${MSWEA_MODEL_NAME:-vertex_ai/gemini-3-flash-preview}"
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"
export PYTHONUNBUFFERED

cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

datasets=(featurebench_full livecodebench)
attacks=(none fcv_cwe538 fcv_cwe79 fcv_cwe89 fcv_cwe94 swexploit_gemini_vertex)
total_jobs=$((${#datasets[@]} * ${#attacks[@]}))
job_idx=0
overall_start=$(date +%s)

echo "[attack-plan] mode=$MODE agent=$AGENT model=$MSWEA_MODEL_NAME"
echo "[attack-plan] datasets=${datasets[*]}"
echo "[attack-plan] attacks=${attacks[*]}"
echo "[attack-plan] total combinations=$total_jobs"
echo

for dataset in "${datasets[@]}"; do
  for attack in "${attacks[@]}"; do
    job_idx=$((job_idx + 1))
    out="outputs/attacks/${MODE}/${dataset}_${attack}"
    expected_instances="$(estimate_instances "$dataset" "$SMOKE_LIMIT")"
    job_start=$(date +%s)
    elapsed=$((job_start - overall_start))
    if (( job_idx > 1 )); then
      avg=$((elapsed / (job_idx - 1)))
      remaining=$((avg * (total_jobs - job_idx + 1)))
      eta="$(fmt_duration "$remaining")"
    else
      eta="estimating"
    fi
    echo "================================================================"
    echo "[attack-plan] [$job_idx/$total_jobs] dataset=$dataset attack=$attack"
    echo "[attack-plan] expected instances=$expected_instances output=$out"
    echo "[attack-plan] elapsed=$(fmt_duration "$elapsed") eta=$eta"
    echo "================================================================"
    args=(
      -m src.eval.cli run_attack
      --dataset "$dataset"
      --split test
      --agent "$AGENT"
      --attack "$attack"
      --fidelity-mode llm
      --out "$out"
    )
    if [[ "$MODE" == "smoke" ]]; then
      args+=(--limit "$SMOKE_LIMIT")
    fi
    echo "[attack-plan] command: $PYTHON ${args[*]}"
    "$PYTHON" "${args[@]}"
    job_end=$(date +%s)
    summary="$out/attack_preprocessing_summary.json"
    if [[ -f "$summary" ]]; then
      "$PYTHON" - "$summary" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
print(
    "[attack-plan] summary: "
    f"generated={summary.get('total_attacks_generated')} "
    f"kept={summary.get('final_dataset_size')} "
    f"discarded={summary.get('total_failed_attacks_discarded')} "
    f"reasons={summary.get('discard_reasons')}"
)
PY
    fi
    echo "[attack-plan] finished [$job_idx/$total_jobs] in $(fmt_duration "$((job_end - job_start))")"
    echo
  done
done

echo "[attack-plan] all combinations finished in $(fmt_duration "$(( $(date +%s) - overall_start ))")"
