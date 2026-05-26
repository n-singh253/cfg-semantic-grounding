#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SHARDS="${SHARDS:-16}"
PARALLEL="${PARALLEL:-4}"
ARCHIVE_EXISTING="${ARCHIVE_EXISTING:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/attacks/claude37_sonnet_sweagent}"
AGENT="${AGENT:-sweagent_claude37_sonnet_vertex}"
ATTACKS="${ATTACKS:-fcv_cwe78 swexploit_gemini_vertex}"
RETRY_DISCARDED_ATTACKS="${RETRY_DISCARDED_ATTACKS:-fcv_cwe78}"

export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/cfg-semantic-grounding/gemini_adc.json}"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-ucr-ursa-major-congliu-lab}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-us-east5}"
export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-east5}"
export ANTHROPIC_VERTEX_PROJECT_ID="${ANTHROPIC_VERTEX_PROJECT_ID:-$GOOGLE_CLOUD_PROJECT}"
export ANTHROPIC_VERTEX_REGION="${ANTHROPIC_VERTEX_REGION:-us-east5}"
export CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC="${CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC:-90}"
export CFG_ANTHROPIC_VERTEX_MAX_CONCURRENT_CALLS="${CFG_ANTHROPIC_VERTEX_MAX_CONCURRENT_CALLS:-2}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[retry-claude-attacks] missing required file: $path" >&2
    exit 1
  fi
}

uses_retry_discarded() {
  local attack="$1"
  local item
  for item in $RETRY_DISCARDED_ATTACKS; do
    if [[ "$item" == "$attack" ]]; then
      return 0
    fi
  done
  return 1
}

attack_output_leaf() {
  local attack="$1"
  case "$attack" in
    fcv_cwe78) echo "livecodebench_fcv_cwe78" ;;
    swexploit_gemini_vertex) echo "livecodebench_swexploit_gemini_vertex" ;;
    *) echo "livecodebench_$attack" ;;
  esac
}

archive_existing_outputs() {
  local stamp archive_root attack leaf final_dir shard_leaf
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  archive_root="$ROOT/outputs/attacks/_archived_claude_retry/$stamp/claude37_sonnet_sweagent"
  for attack in $ATTACKS; do
    leaf="$(attack_output_leaf "$attack")"
    final_dir="$ROOT/$OUT_ROOT/full/$leaf"
    if [[ -e "$final_dir" ]]; then
      mkdir -p "$archive_root/final"
      echo "[retry-claude-attacks] archive $final_dir -> $archive_root/final/$leaf"
      mv "$final_dir" "$archive_root/final/$leaf"
    fi
    shopt -s nullglob
    for shard_leaf in "$ROOT/$OUT_ROOT"/full/shards/shard_*/"$leaf"; do
      mkdir -p "$archive_root/shards/$(basename "$(dirname "$shard_leaf")")"
      echo "[retry-claude-attacks] archive $shard_leaf -> $archive_root/shards/$(basename "$(dirname "$shard_leaf")")/$leaf"
      mv "$shard_leaf" "$archive_root/shards/$(basename "$(dirname "$shard_leaf")")/$leaf"
    done
    shopt -u nullglob
  done
}

require_file "$PYTHON"
require_file "$GOOGLE_APPLICATION_CREDENTIALS"

echo "[retry-claude-attacks] validating attack and SWE-agent config"
"$PYTHON" - <<'PY'
import os
from pathlib import Path
from src.common.config import load_component_config

cfg = load_component_config(Path("configs"), "attacks", "fcv_cwe78")
extra = str(cfg.get("extra_injection", ""))
required = [
    "Do not add, edit, or rely on tests",
    "typically solution.py",
    "os.system",
    "not a comment",
    "do not create new files",
]
missing = [item for item in required if item not in extra]
if missing:
    raise SystemExit(f"fcv_cwe78 extra_injection missing strict clauses: {missing}")
print("fcv_cwe78 strict clauses present")

swexploit_cfg = load_component_config(Path("configs"), "attacks", "swexploit_gemini_vertex")
if swexploit_cfg.get("plugin") != "swexploit":
    raise SystemExit("swexploit_gemini_vertex must use plugin=swexploit")
print("swexploit_gemini_vertex config present")

agent_name = os.environ.get("AGENT", "sweagent_claude37_sonnet_vertex")
agent_cfg = load_component_config(Path("configs"), "agents", agent_name)
command = [str(part) for part in agent_cfg.get("command", [])]
if "--problem_statement.type=text_file" not in command:
    raise SystemExit(f"{agent_name} must use problem_statement.type=text_file")
if "--problem_statement.path={agent_prompt_file}" not in command:
    raise SystemExit(f"{agent_name} must pass {{agent_prompt_file}}")
if any(part.startswith("--problem_statement.text=") for part in command):
    raise SystemExit(f"{agent_name} still passes multiline prompt through text=")
print(f"{agent_name} prompt-file config present")
PY

if [[ "$ARCHIVE_EXISTING" == "1" ]]; then
  archive_existing_outputs
fi

echo "[retry-claude-attacks] out_root=$OUT_ROOT agent=$AGENT attacks=[$ATTACKS] shards=$SHARDS parallel=$PARALLEL archive_existing=$ARCHIVE_EXISTING"
echo "[retry-claude-attacks] credentials=$GOOGLE_APPLICATION_CREDENTIALS project=$GOOGLE_CLOUD_PROJECT region=$ANTHROPIC_VERTEX_REGION"

for attack in $ATTACKS; do
  echo
  echo "[retry-claude-attacks] running attack=$attack"
  cmd=(
    "$PYTHON" -u scripts/run_attack_sharded.py
    --dataset livecodebench \
    --split test \
    --agent "$AGENT" \
    --attack "$attack" \
    --out-root "$OUT_ROOT" \
    --mode full \
    --shards "$SHARDS" \
    --parallel "$PARALLEL"
  )
  if uses_retry_discarded "$attack"; then
    cmd+=(--retry-discarded)
  fi
  "${cmd[@]}"
done

echo
echo "[retry-claude-attacks] completed. Summaries:"
for attack in $ATTACKS; do
  leaf="$(attack_output_leaf "$attack")"
  echo
  echo "== $leaf =="
  cat "$OUT_ROOT/full/$leaf/attack_preprocessing_summary.json"
done
