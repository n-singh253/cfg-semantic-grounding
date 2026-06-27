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
BASELINES="${BASELINES:-semgrep bandit llm_judge structural_misalignment}"
RUN_TAG="${RUN_TAG:-fcv94_basehead}"

# Keep this stable for resume. If you intentionally rematerialize/change
# ~/livecodebench_repos, use a new RUN_TAG or set REFRESH_REPO_COPIES=1.
REFRESH_REPO_COPIES="${REFRESH_REPO_COPIES:-0}"
CLEANUP_REPO_COPIES="${CLEANUP_REPO_COPIES:-1}"
VERIFY_NO_RESET_FAILURES="${VERIFY_NO_RESET_FAILURES:-1}"

PROJECT="${GOOGLE_CLOUD_PROJECT:-ucr-ursa-major-socal-lab}"

run_cmd() {
  echo
  printf '[fcv94-baselines] run:'
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
    echo "[fcv94-baselines] missing required file: $path" >&2
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

build_basehead_datasets() {
  run_cmd "$PYTHON" - <<'PY'
import json
import subprocess
from pathlib import Path

pairs = [
    (
        Path("outputs/attacks/gemini3_flash/mixed/livecodebench_none_vs_fcv_cwe94/full/attack_dataset.jsonl"),
        Path("outputs/attacks/gemini3_flash/mixed/livecodebench_none_vs_fcv_cwe94/full_basehead/attack_dataset.jsonl"),
    ),
    (
        Path("outputs/attacks/gemini3_flash/mixed/livecodebench_none_vs_fcv_cwe94/heldout/attack_dataset.jsonl"),
        Path("outputs/attacks/gemini3_flash/mixed/livecodebench_none_vs_fcv_cwe94/heldout_basehead/attack_dataset.jsonl"),
    ),
    (
        Path("outputs/attacks/claude37_sonnet_sweagent/mixed/livecodebench_none_vs_fcv_cwe94/full/attack_dataset.jsonl"),
        Path("outputs/attacks/claude37_sonnet_sweagent/mixed/livecodebench_none_vs_fcv_cwe94/full_basehead/attack_dataset.jsonl"),
    ),
    (
        Path("outputs/attacks/claude37_sonnet_sweagent/mixed/livecodebench_none_vs_fcv_cwe94/heldout/attack_dataset.jsonl"),
        Path("outputs/attacks/claude37_sonnet_sweagent/mixed/livecodebench_none_vs_fcv_cwe94/heldout_basehead/attack_dataset.jsonl"),
    ),
]

for src, dst in pairs:
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
    print(f"[fcv94-baselines] wrote {rows} rows to {dst} (base_commit_rewritten={changed})")
PY
}

dataset_path() {
  local model_key="$1"
  local split="$2"
  printf '%s/outputs/attacks/%s/mixed/livecodebench_none_vs_fcv_cwe94/%s_basehead/attack_dataset.jsonl' \
    "$ROOT" "$model_key" "$split"
}

baseline_out() {
  local model_key="$1"
  local baseline="$2"
  local split="$3"
  printf '%s/outputs/baselines/livecodebench/%s/%s/mixed_none_vs_fcv_cwe94_%s_%s' \
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
      echo "[fcv94-baselines] unknown baseline alias: $alias" >&2
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

print(f"[fcv94-baselines] verify {path}: total={total} decisions={dict(decisions)} failures={dict(failures)}")
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
  require_file "$dataset"

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

  echo "[fcv94-baselines] RUN_TAG=$RUN_TAG SHARDS=$SHARDS PARALLEL=$PARALLEL BASELINES=$BASELINES"
  echo "[fcv94-baselines] resume: rerun this same script with the same RUN_TAG"

  build_basehead_datasets

  if [[ "$RUN_GEMINI" == "1" ]]; then
    set_gemini_env
    run_model gemini gemini3_flash
  fi

  if [[ "$RUN_CLAUDE" == "1" ]]; then
    set_claude_env
    run_model claude claude37_sonnet_sweagent
  fi

  echo
  echo "[fcv94-baselines] completed RUN_TAG=$RUN_TAG"
}

main "$@"
