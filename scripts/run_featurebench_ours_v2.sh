#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-$ROOT/data/models/structural_misalignment/featurebench_all_v2/hetero_gnn}"
HELDOUT_FILE="${HELDOUT_FILE:-$MODEL_DIR/heldout_instance_ids.txt}"
ATTACKS="${ATTACKS:-fcv_cwe78 swexploit_anthropic}"
SHARDS="${SHARDS:-8}"
PARALLEL="${PARALLEL:-4}"
STALE_TIMEOUT_SEC="${STALE_TIMEOUT_SEC:-7200}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[featurebench-ours-v2] missing required file: $1" >&2
    exit 1
  fi
}

run_model() {
  local model_key="$1"
  "$PYTHON" scripts/build_featurebench_mixed_eval_datasets.py \
    --dataset featurebench_full \
    --model-key "$model_key" \
    --heldout-file "$HELDOUT_FILE" \
    --attacks $ATTACKS

  local attack dataset out
  for attack in $ATTACKS; do
    dataset="$ROOT/outputs/attacks/$model_key/mixed/featurebench_full_none_vs_$attack/heldout/attack_dataset.jsonl"
    require_file "$dataset"
    out="$ROOT/outputs/baselines/featurebench_nonobfuscated_v2/$model_key/$attack/structural_misalignment_featurebench_all_v2"
    "$PYTHON" -u scripts/run_defense_sharded.py \
      --attack-results "$dataset" \
      --baseline structural_misalignment_featurebench_all_v2 \
      --fidelity-mode llm \
      --out "$out" \
      --shards "$SHARDS" \
      --parallel "$PARALLEL" \
      --stale-timeout-sec "$STALE_TIMEOUT_SEC" \
      --poll-interval-sec "$POLL_INTERVAL_SEC" \
      --isolate-repos \
      --cleanup-repo-copies
  done
}

require_file "$MODEL_DIR/model.pt"
require_file "$MODEL_DIR/metadata.json"
require_file "$MODEL_DIR/graph_cache_index.json"
require_file "$HELDOUT_FILE"

run_model gemini3_flash_socal
run_model claude_sonnet46_socal

echo '[featurebench-ours-v2] all requested runs completed'
