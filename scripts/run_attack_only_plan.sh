#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" && -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
SMOKE_LIMIT="${SMOKE_LIMIT:-1}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
SMOKE_FEATUREBENCH_IDS="${SMOKE_FEATUREBENCH_IDS:-python__mypy.8e2ce962.testconstraints.db380fe7.lv1}"
SMOKE_LIVECODEBENCH_IDS="${SMOKE_LIVECODEBENCH_IDS:-lcb_release_latest_atcoder_abc333_a}"
ATTACK_JOB_RETRIES="${ATTACK_JOB_RETRIES:-2}"
ATTACK_ATTACKS="${ATTACK_ATTACKS:-none fcv_cwe538 fcv_cwe79 fcv_cwe89 fcv_cwe94 swexploit_gemini_vertex}"
SMOKE_FEATUREBENCH_AGENT="${SMOKE_FEATUREBENCH_AGENT:-minisweagent_gemini3_flash_featurebench_smoke}"
SMOKE_LIVECODEBENCH_AGENT="${SMOKE_LIVECODEBENCH_AGENT:-minisweagent_gemini3_flash_smoke}"
FULL_AGENT="${FULL_AGENT:-minisweagent_gemini3_flash}"
DRY_RUN="${DRY_RUN:-0}"
FEATUREBENCH_FULL_EXPECTED_ROWS="${FEATUREBENCH_FULL_EXPECTED_ROWS:-200}"
ATTACK_OUTPUT_ROOT="${ATTACK_OUTPUT_ROOT:-outputs/attacks}"
ATTACK_OUT_SUFFIX="${ATTACK_OUT_SUFFIX:-}"

usage() {
  cat <<'EOF'
Usage: scripts/run_attack_only_plan.sh [--smoke|--full]

Environment:
  GOOGLE_APPLICATION_CREDENTIALS  Path to ignored ADC JSON.
  GOOGLE_CLOUD_PROJECT           Vertex project id.
  GOOGLE_CLOUD_LOCATION          Vertex location, defaults to global.
  GOOGLE_GENAI_USE_VERTEXAI       Should be true.
  PYTHON                          Python executable, defaults to python3.
  AGENT                           Agent config override for all datasets.
  FULL_AGENT                      Full-run agent, defaults to minisweagent_gemini3_flash.
  SMOKE_FEATUREBENCH_AGENT        FeatureBench smoke agent, defaults to minisweagent_gemini3_flash_featurebench_smoke.
  SMOKE_LIVECODEBENCH_AGENT       LiveCodeBench smoke agent, defaults to minisweagent_gemini3_flash_smoke.
  SMOKE_LIMIT                     Smoke row count fallback, defaults to 1.
  SMOKE_FEATUREBENCH_IDS          Comma-separated FeatureBench smoke ids, defaults to one stable id.
  SMOKE_LIVECODEBENCH_IDS         Comma-separated LiveCodeBench smoke ids.
  ATTACK_JOB_RETRIES              Per-combination attempts after failure, defaults to 2.
  ATTACK_DATASETS                 Space-separated datasets to run. Defaults to featurebench_full for smoke, both datasets for full.
  ATTACK_ATTACKS                  Space-separated attacks to run.
  FEATUREBENCH_FULL_EXPECTED_ROWS Expected rows for full FeatureBench, defaults to 200.
  ATTACK_OUTPUT_ROOT              Root output directory, defaults to outputs/attacks.
  ATTACK_OUT_SUFFIX               Optional suffix for output leaf dirs, e.g. _claude37.
  DRY_RUN                         Set to 1 to print the plan without running attacks or requiring credentials.
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
      if [[ -f data/featurebench_full_attack_ready.jsonl ]]; then
        wc -l < data/featurebench_full_attack_ready.jsonl | tr -d ' '
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

smoke_instance_ids() {
  local dataset="$1"
  case "$dataset" in
    featurebench_full)
      printf "%s" "$SMOKE_FEATUREBENCH_IDS"
      ;;
    livecodebench)
      printf "%s" "$SMOKE_LIVECODEBENCH_IDS"
      ;;
    *)
      printf ""
      ;;
  esac
}

count_csv_items() {
  local value="$1"
  if [[ -z "$value" ]]; then
    printf "0"
    return
  fi
  awk -F',' '{print NF}' <<<"$value"
}

validate_dataset_ready() {
  local dataset="$1"
  case "$dataset" in
    featurebench_full)
      local path="data/featurebench_full_attack_ready.jsonl"
      if [[ ! -f "$path" ]]; then
        echo "[attack-plan] missing $path" >&2
        echo "[attack-plan] run: $PYTHON scripts/setup_featurebench.py --variant full --force" >&2
        exit 1
      fi
      if [[ "$MODE" == "full" ]]; then
        local rows
        rows="$(wc -l < "$path" | tr -d ' ')"
        if (( rows < FEATUREBENCH_FULL_EXPECTED_ROWS )); then
          echo "[attack-plan] $path has only $rows rows; expected at least $FEATUREBENCH_FULL_EXPECTED_ROWS for --full" >&2
          echo "[attack-plan] run: $PYTHON scripts/setup_featurebench.py --variant full --force" >&2
          exit 1
        fi
      fi
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

if [[ -z "${ATTACK_DATASETS:-}" ]]; then
  if [[ "$MODE" == "smoke" ]]; then
    ATTACK_DATASETS="featurebench_full"
  else
    ATTACK_DATASETS="featurebench_full livecodebench"
  fi
fi

agent_for_dataset() {
  local dataset="$1"
  if [[ -n "${AGENT:-}" ]]; then
    printf "%s" "$AGENT"
    return
  fi
  if [[ "$MODE" == "full" ]]; then
    printf "%s" "$FULL_AGENT"
    return
  fi
  case "$dataset" in
    featurebench_full)
      printf "%s" "$SMOKE_FEATUREBENCH_AGENT"
      ;;
    livecodebench)
      printf "%s" "$SMOKE_LIVECODEBENCH_AGENT"
      ;;
    *)
      printf "%s" "$SMOKE_FEATUREBENCH_AGENT"
      ;;
  esac
}

if [[ "$DRY_RUN" != "1" ]]; then
  : "${GOOGLE_APPLICATION_CREDENTIALS:?Set GOOGLE_APPLICATION_CREDENTIALS first.}"
  : "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT first.}"
fi
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-$GOOGLE_CLOUD_LOCATION}"
export MSWEA_MODEL_NAME="${MSWEA_MODEL_NAME:-vertex_ai/gemini-3-flash-preview}"
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"
export PYTHONUNBUFFERED

cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

read -r -a datasets <<<"$ATTACK_DATASETS"
read -r -a attacks <<<"$ATTACK_ATTACKS"
total_jobs=$((${#datasets[@]} * ${#attacks[@]}))
job_idx=0
overall_start=$(date +%s)

if [[ "$DRY_RUN" != "1" ]]; then
  for dataset in "${datasets[@]}"; do
    validate_dataset_ready "$dataset"
  done
fi

if [[ -n "${AGENT:-}" ]]; then
  agent_banner="$AGENT"
elif [[ "$MODE" == "full" ]]; then
  agent_banner="$FULL_AGENT"
else
  agent_banner="featurebench_full:$SMOKE_FEATUREBENCH_AGENT livecodebench:$SMOKE_LIVECODEBENCH_AGENT"
fi
echo "[attack-plan] mode=$MODE agent=$agent_banner model=$MSWEA_MODEL_NAME"
echo "[attack-plan] datasets=${datasets[*]}"
echo "[attack-plan] attacks=${attacks[*]}"
echo "[attack-plan] total combinations=$total_jobs"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[attack-plan] dry-run=1, commands will not be executed"
fi
echo

for dataset in "${datasets[@]}"; do
  for attack in "${attacks[@]}"; do
    job_idx=$((job_idx + 1))
    out="${ATTACK_OUTPUT_ROOT}/${MODE}/${dataset}_${attack}${ATTACK_OUT_SUFFIX}"
    instance_ids=""
    if [[ "$MODE" == "smoke" ]]; then
      instance_ids="$(smoke_instance_ids "$dataset")"
    fi
    if [[ -n "$instance_ids" ]]; then
      expected_instances="$(count_csv_items "$instance_ids")"
    else
      expected_instances="$(estimate_instances "$dataset" "$SMOKE_LIMIT")"
    fi
    job_start=$(date +%s)
    job_agent="$(agent_for_dataset "$dataset")"
    elapsed=$((job_start - overall_start))
    if (( job_idx > 1 )); then
      avg=$((elapsed / (job_idx - 1)))
      remaining=$((avg * (total_jobs - job_idx + 1)))
      eta="$(fmt_duration "$remaining")"
    else
      eta="estimating"
    fi
    echo "================================================================"
    echo "[attack-plan] [$job_idx/$total_jobs] dataset=$dataset attack=$attack agent=$job_agent"
    echo "[attack-plan] expected instances=$expected_instances output=$out"
    echo "[attack-plan] elapsed=$(fmt_duration "$elapsed") eta=$eta"
    echo "================================================================"
    args=(
      -m src.eval.cli run_attack
      --dataset "$dataset"
      --split test
      --agent "$job_agent"
      --attack "$attack"
      --fidelity-mode llm
      --out "$out"
    )
    if [[ "$MODE" == "smoke" ]]; then
      if [[ -n "$instance_ids" ]]; then
        args+=(--instance-id "$instance_ids")
      else
        args+=(--limit "$SMOKE_LIMIT")
      fi
    fi
    echo "[attack-plan] command: $PYTHON ${args[*]}"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[attack-plan] dry-run: skipped execution"
      echo
      continue
    fi
    attempt=1
    while true; do
      if (( attempt > 1 )); then
        echo "[attack-plan] retry attempt $attempt/$ATTACK_JOB_RETRIES for dataset=$dataset attack=$attack"
      fi
      if "$PYTHON" "${args[@]}"; then
        break
      fi
      if (( attempt >= ATTACK_JOB_RETRIES )); then
        echo "[attack-plan] failed after $attempt attempt(s): dataset=$dataset attack=$attack" >&2
        echo "[attack-plan] recent stderr logs:" >&2
        find "$out" -path '*stderr.log' -type f -printf '%T@ %p\n' 2>/dev/null \
          | sort -nr \
          | head -3 \
          | cut -d' ' -f2- \
          | while read -r log; do
              echo "---- $log ----" >&2
              tail -40 "$log" >&2 || true
            done
        exit 1
      fi
      attempt=$((attempt + 1))
      sleep 10
    done
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
if int(summary.get("total_attacks_generated") or 0) > 0 and int(summary.get("final_dataset_size") or 0) == 0:
    print("[attack-plan] warning: finalized attack dataset is empty for this combination")
PY
    fi
    echo "[attack-plan] finished [$job_idx/$total_jobs] in $(fmt_duration "$((job_end - job_start))")"
    echo
  done
done

echo "[attack-plan] all combinations finished in $(fmt_duration "$(( $(date +%s) - overall_start ))")"
