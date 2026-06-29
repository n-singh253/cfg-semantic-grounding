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
OUT_ROOT="${OUT_ROOT:-outputs/baselines/swebench_nonobfuscated_fcv_featurebench_augmented}"
MODEL_DIR="data/models/structural_misalignment/swebench_nonobfuscated_fcv_featurebench_augmented/hetero_gnn"
BASELINE="structural_misalignment_swebench_fcv_featurebench_augmented"
ATTACK="fcv_cwe78"

IFS=' ' read -r -a MODEL_KEYS <<< "${MODEL_KEYS:-gemini3_flash_swebench_imported claude_sonnet46_swebench_imported}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[swebench-fcv-augmented] missing required file: $1" >&2
    exit 1
  fi
}

stage_graph_cache_dirs() {
  local dst="$1"
  shift
  "$PYTHON" - "$dst" "$@" <<'PY'
import os
import shutil
import sys
from pathlib import Path


def copy_graph_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    try:
        shutil.copytree(src, dst, copy_function=os.link)
    except OSError:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


dst_root = Path(sys.argv[1])
src_roots = [Path(value) for value in sys.argv[2:]]
dst_root.mkdir(parents=True, exist_ok=True)
copied = 0
skipped = 0
missing_sources = []
for src_root in src_roots:
    if not src_root.exists():
        missing_sources.append(str(src_root))
        continue
    for src_graph_dir in src_root.iterdir():
        if not src_graph_dir.is_dir():
            continue
        dst_graph_dir = dst_root / src_graph_dir.name
        if (dst_graph_dir / "graph" / "graph.json").exists():
            skipped += 1
            continue
        copy_graph_dir(src_graph_dir, dst_graph_dir)
        copied += 1
print(
    f"[swebench-fcv-augmented] staged graph cache dst={dst_root} "
    f"copied={copied} skipped={skipped} missing_sources={missing_sources}",
    flush=True,
)
PY
}

stage_graph_caches() {
  if [[ "${SKIP_CACHE_STAGE:-0}" == "1" ]]; then
    echo "[swebench-fcv-augmented] skip cache staging because SKIP_CACHE_STAGE=1"
    return
  fi

  echo "[swebench-fcv-augmented] staging reusable graph caches into $MODEL_DIR"

  # Reuse the SWE-Bench FCV graphs that were just built by the first FCV attempt.
  stage_graph_cache_dirs \
    "$MODEL_DIR/graphs/train" \
    "data/models/structural_misalignment/swebench_nonobfuscated_fcv_logistic_hybrid/hetero_gnn/graphs/train"
  stage_graph_cache_dirs \
    "$MODEL_DIR/graphs/test" \
    "data/models/structural_misalignment/swebench_nonobfuscated_fcv_logistic_hybrid/hetero_gnn/graphs/test"

  # FeatureBench FCV rows are auxiliary training data here.  Its old train and
  # heldout graph caches can both be used as training graph cache entries.
  # The attack-specific FeatureBench model reused featurebench_all as its
  # cache source, so stage from the real source cache rather than the empty
  # attack-specific model graph dirs.
  stage_graph_cache_dirs \
    "$MODEL_DIR/graphs/train" \
    "data/models/structural_misalignment/featurebench_all/hetero_gnn/graphs/train" \
    "data/models/structural_misalignment/featurebench_all/hetero_gnn/graphs/test"
}

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  "$PYTHON" scripts/prepare_swebench_attack_specific_structural_training.py \
    --out-root "$PREP_ROOT" \
    --attacks "$ATTACK"
fi

stage_graph_caches

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if [[ "${FORCE_TRAIN:-0}" != "1" && -f "$MODEL_DIR/model.pt" && -f "$MODEL_DIR/metadata.json" && -f "$MODEL_DIR/graph_cache_index.json" ]]; then
    echo "[swebench-fcv-augmented] model already complete: $MODEL_DIR"
  else
    "$PYTHON" -u src/baseline/structural_misalignment/train_gnn.py \
      --config configs/baselines/structural_misalignment_train_swebench_fcv_featurebench_augmented.yaml \
      --graph-workers "$GRAPH_WORKERS"
  fi
else
  echo "[swebench-fcv-augmented] skip train because SKIP_TRAIN=1"
fi

if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  echo "[swebench-fcv-augmented] skip eval because SKIP_EVAL=1"
  exit 0
fi

require_file "$MODEL_DIR/model.pt"
require_file "$MODEL_DIR/metadata.json"
require_file "$MODEL_DIR/graph_cache_index.json"
require_file "$PREP_ROOT/$ATTACK/heldout_source_instance_ids.txt"

for model_key in "${MODEL_KEYS[@]}"; do
  "$PYTHON" scripts/build_swebench_mixed_eval_datasets.py \
    --dataset swebench \
    --model-key "$model_key" \
    --heldout-file "$PREP_ROOT/$ATTACK/heldout_source_instance_ids.txt" \
    --attacks "$ATTACK"

  dataset="$ROOT/outputs/attacks/$model_key/mixed/swebench_none_vs_$ATTACK/heldout/attack_dataset.jsonl"
  require_file "$dataset"
  out="$ROOT/$OUT_ROOT/$model_key/$ATTACK/$BASELINE"
  "$PYTHON" -u scripts/run_defense_sharded.py \
    --attack-results "$dataset" \
    --baseline "$BASELINE" \
    --fidelity-mode llm \
    --out "$out" \
    --shards "$SHARDS" \
    --parallel "$PARALLEL" \
    --stale-timeout-sec "$STALE_TIMEOUT_SEC" \
    --poll-interval-sec "$POLL_INTERVAL_SEC" \
    --isolate-repos \
    --cleanup-repo-copies
done

echo '[swebench-fcv-augmented] completed'
