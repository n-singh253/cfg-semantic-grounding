#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SHARDS="${SHARDS:-8}"
PARALLEL="${PARALLEL:-4}"
GRAPH_WORKERS="${GRAPH_WORKERS:-2}"
STALE_TIMEOUT_SEC="${STALE_TIMEOUT_SEC:-7200}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
PREP_ROOT="${PREP_ROOT:-data/training/swebench_structural/swebench_imported_attack_specific_20260627}"
OUT_ROOT="${OUT_ROOT:-outputs/baselines/swebench_nonobfuscated_attack_specific_ours}"

IFS=' ' read -r -a MODEL_KEYS <<< "${MODEL_KEYS:-gemini3_flash_swebench_imported claude_sonnet46_swebench_imported}"
IFS=' ' read -r -a ATTACK_LIST <<< "${ATTACKS:-fcv_cwe78 swexploit_anthropic}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[swebench-ours-attack-specific] missing required file: $1" >&2
    exit 1
  fi
}

baseline_for_attack() {
  case "$1" in
    fcv_cwe78) printf 'structural_misalignment_swebench_fcv_logistic_hybrid' ;;
    swexploit_anthropic) printf 'structural_misalignment_swebench_swexploit_raw_residual' ;;
    *) echo "unknown attack: $1" >&2; exit 2 ;;
  esac
}

train_config_for_attack() {
  case "$1" in
    fcv_cwe78) printf 'configs/baselines/structural_misalignment_train_swebench_fcv_logistic_hybrid.yaml' ;;
    swexploit_anthropic) printf 'configs/baselines/structural_misalignment_train_swebench_swexploit_raw_residual.yaml' ;;
    *) echo "unknown attack: $1" >&2; exit 2 ;;
  esac
}

model_dir_for_attack() {
  case "$1" in
    fcv_cwe78) printf 'data/models/structural_misalignment/swebench_nonobfuscated_fcv_logistic_hybrid/hetero_gnn' ;;
    swexploit_anthropic) printf 'data/models/structural_misalignment/swebench_nonobfuscated_swexploit_raw_residual/hetero_gnn' ;;
    *) echo "unknown attack: $1" >&2; exit 2 ;;
  esac
}

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  "$PYTHON" scripts/prepare_swebench_attack_specific_structural_training.py \
    --out-root "$PREP_ROOT"
fi

train_one() {
  local attack="$1"
  local config model_dir
  config="$(train_config_for_attack "$attack")"
  model_dir="$(model_dir_for_attack "$attack")"

  if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    echo "[swebench-ours-attack-specific] skip train $attack because SKIP_TRAIN=1"
    return
  fi

  if [[ "${FORCE_TRAIN:-0}" != "1" && -f "$model_dir/model.pt" && -f "$model_dir/metadata.json" && -f "$model_dir/graph_cache_index.json" ]]; then
    echo "[swebench-ours-attack-specific] model already complete for $attack: $model_dir"
    return
  fi

  echo "[swebench-ours-attack-specific] train $attack with $config"
  "$PYTHON" -u src/baseline/structural_misalignment/train_gnn.py \
    --config "$config" \
    --graph-workers "$GRAPH_WORKERS"
}

run_eval_one() {
  local model_key="$1"
  local attack="$2"
  local baseline model_dir heldout dataset out
  baseline="$(baseline_for_attack "$attack")"
  model_dir="$(model_dir_for_attack "$attack")"
  heldout="$PREP_ROOT/$attack/heldout_source_instance_ids.txt"

  require_file "$model_dir/model.pt"
  require_file "$model_dir/metadata.json"
  require_file "$model_dir/graph_cache_index.json"
  require_file "$heldout"

  "$PYTHON" scripts/build_swebench_mixed_eval_datasets.py \
    --dataset swebench \
    --model-key "$model_key" \
    --heldout-file "$heldout" \
    --attacks "$attack"

  dataset="$ROOT/outputs/attacks/$model_key/mixed/swebench_none_vs_$attack/heldout/attack_dataset.jsonl"
  require_file "$dataset"

  out="$ROOT/$OUT_ROOT/$model_key/$attack/$baseline"
  echo "[swebench-ours-attack-specific] eval model_key=$model_key attack=$attack baseline=$baseline"
  "$PYTHON" -u scripts/run_defense_sharded.py \
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
}

for attack in "${ATTACK_LIST[@]}"; do
  train_one "$attack"
done

if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  echo "[swebench-ours-attack-specific] skip eval because SKIP_EVAL=1"
  exit 0
fi

for model_key in "${MODEL_KEYS[@]}"; do
  for attack in "${ATTACK_LIST[@]}"; do
    run_eval_one "$model_key" "$attack"
  done
done

echo '[swebench-ours-attack-specific] all requested runs completed'
