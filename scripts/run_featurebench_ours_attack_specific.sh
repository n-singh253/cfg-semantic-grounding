#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SHARDS="${SHARDS:-8}"
PARALLEL="${PARALLEL:-4}"
STALE_TIMEOUT_SEC="${STALE_TIMEOUT_SEC:-7200}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"

FCV_MODEL_DIR="$ROOT/data/models/structural_misalignment/featurebench_fcv_logistic_hybrid/hetero_gnn"
SW_MODEL_DIR="$ROOT/data/models/structural_misalignment/featurebench_swexploit/hetero_gnn"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[featurebench-ours-attack-specific] missing required file: $1" >&2
    exit 1
  fi
}

baseline_for_attack() {
  case "$1" in
    fcv_cwe78) printf 'structural_misalignment_featurebench_fcv_logistic_hybrid' ;;
    swexploit_anthropic) printf 'structural_misalignment_featurebench_swexploit_trained' ;;
    *) echo "unknown attack: $1" >&2; exit 2 ;;
  esac
}

heldout_for_attack() {
  case "$1" in
    fcv_cwe78) printf '%s/heldout_instance_ids.txt' "$FCV_MODEL_DIR" ;;
    swexploit_anthropic) printf '%s/heldout_instance_ids.txt' "$SW_MODEL_DIR" ;;
    *) echo "unknown attack: $1" >&2; exit 2 ;;
  esac
}

run_one() {
  local model_key="$1"
  local attack="$2"
  local baseline heldout dataset out
  baseline="$(baseline_for_attack "$attack")"
  heldout="$(heldout_for_attack "$attack")"
  require_file "$heldout"

  "$PYTHON" scripts/build_featurebench_mixed_eval_datasets.py \
    --dataset featurebench_full \
    --model-key "$model_key" \
    --heldout-file "$heldout" \
    --attacks "$attack"

  dataset="$ROOT/outputs/attacks/$model_key/mixed/featurebench_full_none_vs_$attack/heldout/attack_dataset.jsonl"
  require_file "$dataset"
  out="$ROOT/outputs/baselines/featurebench_nonobfuscated_attack_specific/$model_key/$attack/$baseline"
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

for model_dir in "$FCV_MODEL_DIR" "$SW_MODEL_DIR"; do
  require_file "$model_dir/model.pt"
  require_file "$model_dir/metadata.json"
  require_file "$model_dir/graph_cache_index.json"
done

for model_key in gemini3_flash_socal claude_sonnet46_socal; do
  run_one "$model_key" fcv_cwe78
  run_one "$model_key" swexploit_anthropic
done

echo '[featurebench-ours-attack-specific] all requested runs completed'
