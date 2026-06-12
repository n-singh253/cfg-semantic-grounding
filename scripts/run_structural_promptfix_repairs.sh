#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SHARDS="${SHARDS:-8}"
PARALLEL="${PARALLEL:-4}"
STALE_TIMEOUT_SEC="${STALE_TIMEOUT_SEC:-7200}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
DRY_RUN="${DRY_RUN:-0}"
CLEAN="${CLEAN:-0}"
CLEAN_ONLY="${CLEAN_ONLY:-0}"
KILL_STALE="${KILL_STALE:-1}"
RUN_GEMINI="${RUN_GEMINI:-1}"
RUN_CLAUDE="${RUN_CLAUDE:-1}"
FCV_ATTACK="${FCV_ATTACK:-fcv_cwe94}"

IDS_ROOT="$ROOT/outputs/diagnostics/structural_misalignment/filtered_clean_eval_results"

GEMINI_FCV_OUT="$ROOT/outputs/baselines/livecodebench/gemini3_flash/structural_misalignment_livecodebench_gemini/${FCV_ATTACK}_promptfix_badrepair"
GEMINI_SW_OUT="$ROOT/outputs/baselines/livecodebench/gemini3_flash/structural_misalignment_livecodebench_gemini/swexploit_gemini_vertex_promptfix_badrepair"
CLAUDE_FCV_OUT="$ROOT/outputs/baselines/livecodebench/claude37_sonnet_sweagent/structural_misalignment_livecodebench_claude/${FCV_ATTACK}_promptfix_badrepair"
CLAUDE_SW_OUT="$ROOT/outputs/baselines/livecodebench/claude37_sonnet_sweagent/structural_misalignment_livecodebench_claude/swexploit_gemini_vertex_promptfix_badrepair"

usage() {
  cat <<'EOF'
Usage: scripts/run_structural_promptfix_repairs.sh [--dry-run] [--clean-only] [--clean] [--no-clean]

Runs the four structural-misalignment prompt-fix repair evaluations:
  1. Gemini FCV confirmed-problematic rows
  2. Gemini SWExploit confirmed-problematic rows
  3. Claude FCV confirmed-problematic rows
  4. Claude SWExploit confirmed-problematic rows

By default, the script resumes into existing *_promptfix_badrepair output
directories. Use --clean or CLEAN=1 only when you intentionally want to remove
those repair outputs first. It does not modify the old completed eval directories.

Environment overrides:
  PYTHON                 Python executable, defaults to .venv/bin/python.
  SHARDS                 Shard count, defaults to 8.
  PARALLEL               Parallel shard workers, defaults to 4.
  STALE_TIMEOUT_SEC      Stale shard timeout, defaults to 7200.
  POLL_INTERVAL_SEC      Poll interval, defaults to 30.
  DRY_RUN=1              Print commands without running.
  CLEAN=1                Remove repair output dirs before running.
  CLEAN_ONLY=1           Remove repair output dirs and exit.
  KILL_STALE=0           Do not terminate stale repair shard processes before cleanup.
  RUN_GEMINI=0           Skip Gemini repairs.
  RUN_CLAUDE=0           Skip Claude repairs.
  FCV_ATTACK=fcv_cwe94   FCV attack dataset name, defaults to fcv_cwe94.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    --clean-only)
      CLEAN_ONLY=1
      CLEAN=1
      ;;
    --clean)
      CLEAN=1
      ;;
    --no-clean)
      CLEAN=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[promptfix-repair] missing required file: $path" >&2
    exit 1
  fi
}

require_nonempty_id_file() {
  local path="$1"
  require_file "$path"
  if [[ ! -s "$path" ]]; then
    echo "[promptfix-repair] id file is empty: $path" >&2
    exit 1
  fi
}

run_cmd() {
  echo
  printf '[promptfix-repair] run:'
  printf ' %q' "$@"
  echo
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

clean_dir() {
  local path="$1"
  if [[ "$CLEAN" != "1" && "$CLEAN_ONLY" != "1" ]]; then
    return 0
  fi
  echo "[promptfix-repair] remove: $path"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  for attempt in 1 2 3; do
    rm -rf "$path" && return 0
    echo "[promptfix-repair] remove failed for $path; retry $attempt/3 after short sleep" >&2
    sleep 2
  done
  echo "[promptfix-repair] failed to remove $path" >&2
  echo "[promptfix-repair] check for stale processes with: ps -eo pid,ppid,stat,etime,args | grep promptfix_fullrepair" >&2
  return 1
}

kill_stale_repair_processes() {
  if [[ "$KILL_STALE" != "1" ]]; then
    return 0
  fi
  local pids
  pids="$(
    ps -eo pid=,args= \
      | awk '/promptfix_(fullrepair|badrepair)/ && /src[.]eval[.]cli run_defense|run_defense_sharded[.]py/ {print $1}' \
      | sort -u
  )"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  echo "[promptfix-repair] terminating stale repair processes: $pids"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 5
  local survivors
  survivors="$(
    ps -eo pid=,args= \
      | awk '/promptfix_(fullrepair|badrepair)/ && /src[.]eval[.]cli run_defense|run_defense_sharded[.]py/ {print $1}' \
      | sort -u
  )"
  if [[ -n "$survivors" ]]; then
    echo "[promptfix-repair] force-killing stale repair processes: $survivors"
    # shellcheck disable=SC2086
    kill -9 $survivors 2>/dev/null || true
  fi
}

set_common_vertex_env() {
  export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/cfg-semantic-grounding/gemini_adc.json}"
  export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-ucr-ursa-major-congliu-lab}"
  export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
  export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
  export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
}

set_gemini_env() {
  set_common_vertex_env
  export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-global}"
  export CFG_GEMINI_VERTEX_SAFETY_THRESHOLD="${CFG_GEMINI_VERTEX_SAFETY_THRESHOLD:-BLOCK_NONE}"
  export CFG_GEMINI_VERTEX_MAX_CONCURRENT_CALLS="${CFG_GEMINI_VERTEX_MAX_CONCURRENT_CALLS:-2}"
  export CFG_GEMINI_VERTEX_TIMEOUT_MS="${CFG_GEMINI_VERTEX_TIMEOUT_MS:-90000}"
  export CFG_GEMINI_VERTEX_MAX_OUTPUT_TOKENS="${CFG_GEMINI_VERTEX_MAX_OUTPUT_TOKENS:-4096}"
}

set_claude_env() {
  set_common_vertex_env
  export VERTEXAI_LOCATION="us-east5"
  export ANTHROPIC_VERTEX_PROJECT_ID="${ANTHROPIC_VERTEX_PROJECT_ID:-$GOOGLE_CLOUD_PROJECT}"
  export ANTHROPIC_VERTEX_REGION="${ANTHROPIC_VERTEX_REGION:-us-east5}"
  export CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC="${CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC:-90}"
  export CFG_ANTHROPIC_VERTEX_MAX_CONCURRENT_CALLS="${CFG_ANTHROPIC_VERTEX_MAX_CONCURRENT_CALLS:-2}"
}

check_prereqs() {
  require_file "$PYTHON"
  require_file "$ROOT/scripts/run_defense_sharded.py"
  require_file "$ROOT/outputs/attacks/gemini3_flash/full/livecodebench_${FCV_ATTACK}/attack_dataset.jsonl"
  require_file "$ROOT/outputs/attacks/gemini3_flash/full/livecodebench_swexploit_gemini_vertex/attack_dataset.jsonl"
  require_file "$ROOT/outputs/attacks/claude37_sonnet_sweagent/full/livecodebench_${FCV_ATTACK}/attack_dataset.jsonl"
  require_file "$ROOT/outputs/attacks/claude37_sonnet_sweagent/full/livecodebench_swexploit_gemini_vertex/attack_dataset.jsonl"
  require_nonempty_id_file "$IDS_ROOT/gemini_fcv/removed_problematic_ids.txt"
  require_nonempty_id_file "$IDS_ROOT/gemini_swexploit/removed_problematic_ids.txt"
  require_nonempty_id_file "$IDS_ROOT/claude_fcv/removed_problematic_ids.txt"
  require_nonempty_id_file "$IDS_ROOT/claude_swexploit/removed_problematic_ids.txt"
  require_file "$ROOT/configs/baselines/structural_misalignment_livecodebench_gemini.yaml"
  require_file "$ROOT/configs/baselines/structural_misalignment_livecodebench_claude.yaml"
}

run_repair() {
  local attack_results="$1"
  local baseline="$2"
  local out_dir="$3"
  local id_file="$4"

  run_cmd "$PYTHON" "$ROOT/scripts/run_defense_sharded.py" \
    --attack-results "$attack_results" \
    --baseline "$baseline" \
    --fidelity-mode llm \
    --out "$out_dir" \
    --instance-id-file "$id_file" \
    --shards "$SHARDS" \
    --parallel "$PARALLEL" \
    --stale-timeout-sec "$STALE_TIMEOUT_SEC" \
    --poll-interval-sec "$POLL_INTERVAL_SEC" \
    --isolate-repos \
    --cleanup-repo-copies
}

main() {
  cd "$ROOT"
  check_prereqs
  kill_stale_repair_processes

  clean_dir "$GEMINI_FCV_OUT"
  clean_dir "$GEMINI_SW_OUT"
  clean_dir "$CLAUDE_FCV_OUT"
  clean_dir "$CLAUDE_SW_OUT"

  if [[ "$CLEAN_ONLY" == "1" ]]; then
    echo "[promptfix-repair] clean-only complete"
    exit 0
  fi

  if [[ "$RUN_GEMINI" == "1" ]]; then
    set_gemini_env
    run_repair \
      "$ROOT/outputs/attacks/gemini3_flash/full/livecodebench_${FCV_ATTACK}/attack_dataset.jsonl" \
      structural_misalignment_livecodebench_gemini \
      "$GEMINI_FCV_OUT" \
      "$IDS_ROOT/gemini_fcv/removed_problematic_ids.txt"

    run_repair \
      "$ROOT/outputs/attacks/gemini3_flash/full/livecodebench_swexploit_gemini_vertex/attack_dataset.jsonl" \
      structural_misalignment_livecodebench_gemini \
      "$GEMINI_SW_OUT" \
      "$IDS_ROOT/gemini_swexploit/removed_problematic_ids.txt"
  fi

  if [[ "$RUN_CLAUDE" == "1" ]]; then
    set_claude_env
    run_repair \
      "$ROOT/outputs/attacks/claude37_sonnet_sweagent/full/livecodebench_${FCV_ATTACK}/attack_dataset.jsonl" \
      structural_misalignment_livecodebench_claude \
      "$CLAUDE_FCV_OUT" \
      "$IDS_ROOT/claude_fcv/removed_problematic_ids.txt"

    run_repair \
      "$ROOT/outputs/attacks/claude37_sonnet_sweagent/full/livecodebench_swexploit_gemini_vertex/attack_dataset.jsonl" \
      structural_misalignment_livecodebench_claude \
      "$CLAUDE_SW_OUT" \
      "$IDS_ROOT/claude_swexploit/removed_problematic_ids.txt"
  fi

  echo
  echo "[promptfix-repair] all requested repair runs completed"
}

main "$@"
