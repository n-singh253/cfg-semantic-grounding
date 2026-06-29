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
RUN_CLAUDE="${RUN_CLAUDE:-0}"
RUN_OPENHANDS="${RUN_OPENHANDS:-0}"
OPENHANDS_MODEL_KEY="${OPENHANDS_MODEL_KEY:-openhands_qwen3_coder_30b_v2}"
OPENHANDS_LLM_JUDGE_BASELINE="${OPENHANDS_LLM_JUDGE_BASELINE:-llm_judge_qwen3_coder_30b_openrouter}"
OPENHANDS_STRUCTURAL_BASELINE="${OPENHANDS_STRUCTURAL_BASELINE:-structural_misalignment_livecodebench_gemini_fcv89}"
BASELINES="${BASELINES:-semgrep bandit llm_judge structural_misalignment}"
RUN_TAG="${RUN_TAG:-fcv89_basehead}"
GEMINI_FCV89_MODEL_DIR="${GEMINI_FCV89_MODEL_DIR:-data/models/structural_misalignment/livecodebench_gemini_fcv89/hetero_gnn}"
GEMINI_HELDOUT_FILE="${GEMINI_HELDOUT_FILE:-$GEMINI_FCV89_MODEL_DIR/heldout_instance_ids.txt}"

# Keep this stable for resume. If you intentionally rematerialize/change
# ~/livecodebench_repos, use a new RUN_TAG or set REFRESH_REPO_COPIES=1.
REFRESH_REPO_COPIES="${REFRESH_REPO_COPIES:-0}"
CLEANUP_REPO_COPIES="${CLEANUP_REPO_COPIES:-1}"
VERIFY_NO_RESET_FAILURES="${VERIFY_NO_RESET_FAILURES:-1}"

PROJECT="${GOOGLE_CLOUD_PROJECT:-ucr-ursa-major-socal-lab}"

run_cmd() {
  echo
  printf '[fcv89-baselines] run:'
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
    echo "[fcv89-baselines] missing required file: $path" >&2
    exit 1
  fi
}

abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s' "$path"
  else
    printf '%s/%s' "$ROOT" "$path"
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
  export VERTEXAI_LOCATION="global"
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

set_openhands_env() {
  export LLM_MODEL="${LLM_MODEL:-openrouter/qwen/qwen3-coder-30b-a3b-instruct}"
  if [[ -z "${OPENROUTER_API_KEY:-}" && -n "${LLM_API_KEY:-}" ]]; then
    export OPENROUTER_API_KEY="$LLM_API_KEY"
  fi
  if [[ -z "${LLM_API_KEY:-}" && -n "${OPENROUTER_API_KEY:-}" ]]; then
    export LLM_API_KEY="$OPENROUTER_API_KEY"
  fi
  if [[ -z "${OPENROUTER_API_KEY:-}" && -z "${LLM_API_KEY:-}" ]]; then
    echo "[fcv89-baselines] OPENROUTER_API_KEY or LLM_API_KEY is required for OpenHands/Qwen LLM judge" >&2
    exit 1
  fi
}

build_basehead_datasets() {
  run_cmd "$PYTHON" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

model_keys = []
if os.environ.get("RUN_GEMINI", "1") == "1":
    model_keys.append("gemini3_flash")
if os.environ.get("RUN_CLAUDE", "0") == "1":
    model_keys.append("claude37_sonnet_sweagent")
if os.environ.get("RUN_OPENHANDS", "0") == "1":
    model_keys.append(os.environ.get("OPENHANDS_MODEL_KEY", "openhands_qwen3_coder_30b_v2"))

for model_key in model_keys:
    for split in ("full", "heldout"):
        src = Path("outputs/attacks") / model_key / "mixed" / "livecodebench_none_vs_fcv_cwe89" / split / "attack_dataset.jsonl"
        dst = Path("outputs/attacks") / model_key / "mixed" / "livecodebench_none_vs_fcv_cwe89" / f"{split}_basehead" / "attack_dataset.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"missing source dataset: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        rows = 0
        changed = 0
        with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                repo = Path(str(row.get("repo_path", "") or ""))
                if not repo.exists():
                    raise FileNotFoundError(
                        f"{row.get('instance_id', 'unknown')}: repo_path does not exist: {repo}\n"
                        "Run scripts/setup_livecodebench.py or restore ~/livecodebench_repos first."
                    )
                probe = subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    cwd=repo,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if probe.returncode != 0:
                    raise RuntimeError(f"{row.get('instance_id', 'unknown')}: repo_path is not a git repo: {repo}")
                head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
                old_base = str(row.get("base_commit", "") or "")
                row["original_base_commit"] = row.get("original_base_commit") or old_base
                row["base_commit"] = head
                if old_base != head:
                    changed += 1
                out.write(json.dumps(row, sort_keys=True) + "\n")
                rows += 1
        print(f"[fcv89-baselines] wrote {rows} rows to {dst} (base_commit_rewritten={changed})")
PY
}

uses_structural() {
  local alias
  for alias in $BASELINES; do
    if [[ "$alias" == "structural_misalignment" ]]; then
      return 0
    fi
  done
  return 1
}

prepare_mixed_datasets() {
  if [[ "$RUN_GEMINI" == "1" ]]; then
    if uses_structural; then
      local heldout_path model_dir
      heldout_path="$(abs_path "$GEMINI_HELDOUT_FILE")"
      model_dir="$(abs_path "$GEMINI_FCV89_MODEL_DIR")"
      if [[ "$DRY_RUN" == "1" ]] && { [[ ! -f "$heldout_path" ]] || [[ ! -f "$model_dir/model.pt" ]]; }; then
        echo "[fcv89-baselines] dry-run: would require trained CWE-89 GNN at $model_dir"
      else
        require_file "$heldout_path"
        require_file "$model_dir/model.pt"
      fi
      run_cmd "$PYTHON" "$ROOT/scripts/build_livecodebench_mixed_eval_datasets.py" \
        --model-key gemini3_flash \
        --attacks fcv_cwe89 \
        --heldout-file "$heldout_path"
    elif [[ ! -f "$ROOT/outputs/attacks/gemini3_flash/mixed/livecodebench_none_vs_fcv_cwe89/full/attack_dataset.jsonl" ]]; then
      run_cmd "$PYTHON" "$ROOT/scripts/build_livecodebench_mixed_eval_datasets.py" \
        --model-key gemini3_flash \
        --attacks fcv_cwe89 \
        --heldout-file "$ROOT/data/models/structural_misalignment/livecodebench_gemini/hetero_gnn/heldout_instance_ids.txt"
    fi
  fi
}

dataset_path() {
  local model_key="$1"
  local split="$2"
  printf '%s/outputs/attacks/%s/mixed/livecodebench_none_vs_fcv_cwe89/%s_basehead/attack_dataset.jsonl' \
    "$ROOT" "$model_key" "$split"
}

baseline_out() {
  local model_key="$1"
  local baseline="$2"
  local split="$3"
  printf '%s/outputs/baselines/livecodebench/%s/%s/mixed_none_vs_fcv_cwe89_%s_%s' \
    "$ROOT" "$model_key" "$baseline" "$split" "$RUN_TAG"
}

resolved_baseline() {
  local model_family="$1"
  local alias="$2"
  case "$alias" in
    semgrep|bandit)
      printf '%s' "$alias"
      ;;
    llm_judge)
      if [[ "$model_family" == "gemini" ]]; then
        printf 'llm_judge_gemini_vertex'
      elif [[ "$model_family" == "openhands" ]]; then
        printf '%s' "$OPENHANDS_LLM_JUDGE_BASELINE"
      else
        printf 'llm_judge_claude37_sonnet_vertex'
      fi
      ;;
    structural_misalignment)
      if [[ "$model_family" == "gemini" ]]; then
        printf 'structural_misalignment_livecodebench_gemini_fcv89'
      elif [[ "$model_family" == "openhands" ]]; then
        printf '%s' "$OPENHANDS_STRUCTURAL_BASELINE"
      else
        printf 'structural_misalignment_livecodebench_claude'
      fi
      ;;
    *)
      echo "[fcv89-baselines] unknown baseline alias: $alias" >&2
      exit 2
      ;;
  esac
}

verify_no_reset_failures() {
  local out="$1"
  if [[ "$VERIFY_NO_RESET_FAILURES" != "1" || ! -f "$out/results.jsonl" ]]; then
    return 0
  fi
  "$PYTHON" - "$out/results.jsonl" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
total = 0
failures = Counter()
decisions = Counter()
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    total += 1
    decisions[str(row.get("defense_decision", ""))] += 1
    signals = row.get("defense_signals") if isinstance(row.get("defense_signals"), dict) else {}
    reason = str(signals.get("failure_reason", "") or "")
    if reason:
        failures[reason] += 1

print(f"[fcv89-baselines] verify {path}: total={total} decisions={dict(decisions)} failures={dict(failures)}")
if failures.get("initial_reset_failed", 0):
    raise SystemExit(
        f"initial_reset_failed appeared in {path}; results are invalid. "
        "Use a fresh RUN_TAG after fixing the dataset/repo base commits."
    )
PY
}

run_one() {
  local model_family="$1"
  local model_key="$2"
  local alias="$3"
  local baseline split dataset out

  baseline="$(resolved_baseline "$model_family" "$alias")"
  split="full"
  if [[ "$alias" == "structural_misalignment" ]]; then
    split="heldout"
  fi
  dataset="$(dataset_path "$model_key" "$split")"
  out="$(baseline_out "$model_key" "$baseline" "$split")"
  if [[ "$DRY_RUN" == "1" && ! -f "$dataset" ]]; then
    echo "[fcv89-baselines] dry-run: would require dataset after basehead build: $dataset"
  else
    require_file "$dataset"
  fi

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

  if [[ "$alias" == "semgrep" || "$alias" == "bandit" ]]; then
    cmd+=(--isolate-repos)
    if [[ "$REFRESH_REPO_COPIES" == "1" ]]; then
      cmd+=(--refresh-repo-copies)
    fi
    if [[ "$CLEANUP_REPO_COPIES" == "1" ]]; then
      cmd+=(--cleanup-repo-copies)
    fi
  fi

  run_cmd "${cmd[@]}"
  if [[ "$DRY_RUN" != "1" ]]; then
    verify_no_reset_failures "$out"
  fi
}

run_model() {
  local model_family="$1"
  local model_key="$2"
  local alias
  for alias in $BASELINES; do
    run_one "$model_family" "$model_key" "$alias"
  done
}

main() {
  cd "$ROOT"
  require_file "$PYTHON"
  require_file "$ROOT/scripts/run_defense_sharded.py"
  unset PYTHONPATH

  echo "[fcv89-baselines] RUN_TAG=$RUN_TAG SHARDS=$SHARDS PARALLEL=$PARALLEL BASELINES=$BASELINES"
  echo "[fcv89-baselines] resume: rerun this same script with the same RUN_TAG"

  prepare_mixed_datasets
  build_basehead_datasets

  if [[ "$RUN_GEMINI" == "1" ]]; then
    set_gemini_env
    run_model gemini gemini3_flash
  fi

  if [[ "$RUN_CLAUDE" == "1" ]]; then
    set_claude_env
    run_model claude claude37_sonnet_sweagent
  fi

  if [[ "$RUN_OPENHANDS" == "1" ]]; then
    set_openhands_env
    run_model openhands "$OPENHANDS_MODEL_KEY"
  fi

  echo
  echo "[fcv89-baselines] completed RUN_TAG=$RUN_TAG"
}

main "$@"
