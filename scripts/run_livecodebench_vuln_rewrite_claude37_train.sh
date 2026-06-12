#!/usr/bin/env bash
set -euo pipefail

# Build the training-only vulnerable rewrite attack dataset for LiveCodeBench.
#
# Required environment for Claude on Vertex:
#   GOOGLE_APPLICATION_CREDENTIALS
#   GOOGLE_CLOUD_PROJECT
#   VERTEXAI_LOCATION or ANTHROPIC_VERTEX_REGION
#
# Optional overrides:
#   SHARDS=8
#   PARALLEL=8
#   WORKERS=8        # convenience alias for SHARDS/PARALLEL when those are unset
#   LIMIT=10
#   OUT_ROOT=outputs/attacks/claude37_sonnet_sweagent
#   AGENT=sweagent_claude37_sonnet_vertex_portable
#   ATTACK=vuln_rewrite_claude37_sonnet_vertex
#   LCB_DATA_PATH=data/livecodebench_code_generation_lite_release_latest.jsonl
#   LCB_AUTO_SETUP=1
#   LCB_RELEASE=release_latest
#   LCB_SETUP_LIMIT=10
#   LCB_REPOS_ROOT=$HOME/livecodebench_repos
#   DRY_RUN=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DATASET="${DATASET:-livecodebench}"
SPLIT="${SPLIT:-test}"
AGENT="${AGENT:-sweagent_claude37_sonnet_vertex_portable}"
ATTACK="${ATTACK:-vuln_rewrite_claude37_sonnet_vertex}"
OUT_ROOT="${OUT_ROOT:-outputs/attacks/claude37_sonnet_sweagent}"
WORKERS="${WORKERS:-}"
if [[ -n "$WORKERS" ]]; then
  SHARDS="${SHARDS:-$WORKERS}"
  PARALLEL="${PARALLEL:-$WORKERS}"
else
  SHARDS="${SHARDS:-8}"
  PARALLEL="${PARALLEL:-8}"
fi
MODE="${MODE:-full}"
CONFIG_DIR="${CONFIG_DIR:-configs}"
FIDELITY_MODE="${FIDELITY_MODE:-llm}"
LCB_RELEASE="${LCB_RELEASE:-release_latest}"
LCB_AUTO_SETUP="${LCB_AUTO_SETUP:-1}"
LCB_DATA_PATH="${LCB_DATA_PATH:-${LIVE_CODEBENCH_DATA_PATH:-}}"
LCB_REPOS_ROOT="${LCB_REPOS_ROOT:-${LIVE_CODEBENCH_REPOS_ROOT:-}}"
LCB_DEFAULT_DATA_PATH="data/livecodebench_code_generation_lite_${LCB_RELEASE}.jsonl"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[vuln-rewrite] missing Python executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" || ! -f "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  echo "[vuln-rewrite] GOOGLE_APPLICATION_CREDENTIALS must point to an existing ADC JSON file" >&2
  exit 1
fi

if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
  echo "[vuln-rewrite] GOOGLE_CLOUD_PROJECT is required" >&2
  exit 1
fi

if [[ -z "${ANTHROPIC_VERTEX_REGION:-}" && -z "${VERTEXAI_LOCATION:-}" ]]; then
  echo "[vuln-rewrite] set ANTHROPIC_VERTEX_REGION or VERTEXAI_LOCATION for Claude on Vertex" >&2
  exit 1
fi

export ANTHROPIC_VERTEX_PROJECT_ID="${ANTHROPIC_VERTEX_PROJECT_ID:-$GOOGLE_CLOUD_PROJECT}"
export ANTHROPIC_VERTEX_REGION="${ANTHROPIC_VERTEX_REGION:-${VERTEXAI_LOCATION}}"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-$ANTHROPIC_VERTEX_REGION}"
export CFG_SWEAGENT_NO_CACHE_CONFIG="${CFG_SWEAGENT_NO_CACHE_CONFIG:-$ROOT_DIR/configs/sweagent_no_cache.yaml}"

if [[ -z "${SWEAGENT_BIN:-}" ]]; then
  if command -v sweagent >/dev/null 2>&1; then
    SWEAGENT_BIN="$(command -v sweagent)"
  elif [[ -x "$HOME/.cache/cfg-semantic-grounding/sweagent-venv-py311/bin/sweagent" ]]; then
    SWEAGENT_BIN="$HOME/.cache/cfg-semantic-grounding/sweagent-venv-py311/bin/sweagent"
  else
    echo "[vuln-rewrite] SWEAGENT_BIN is not set and sweagent was not found on PATH" >&2
    echo "[vuln-rewrite] Install SWE-Agent or set SWEAGENT_BIN=/path/to/sweagent" >&2
    exit 1
  fi
fi
export SWEAGENT_BIN

if [[ -z "${SWEAGENT_DEFAULT_CONFIG:-}" ]]; then
  if [[ -f "$HOME/.cache/cfg-semantic-grounding/SWE-agent/config/default.yaml" ]]; then
    SWEAGENT_DEFAULT_CONFIG="$HOME/.cache/cfg-semantic-grounding/SWE-agent/config/default.yaml"
  else
    echo "[vuln-rewrite] SWEAGENT_DEFAULT_CONFIG is not set" >&2
    echo "[vuln-rewrite] Set SWEAGENT_DEFAULT_CONFIG=/path/to/SWE-agent/config/default.yaml" >&2
    exit 1
  fi
fi
export SWEAGENT_DEFAULT_CONFIG

if [[ "$DATASET" == "livecodebench" ]]; then
  LCB_DATA_PATH="${LCB_DATA_PATH:-$LCB_DEFAULT_DATA_PATH}"
  lcb_source="$LCB_DATA_PATH"
  if [[ "$lcb_source" != /* ]]; then
    lcb_source="$ROOT_DIR/$lcb_source"
  fi

  if [[ ! -f "$lcb_source" ]]; then
    if [[ "$LCB_AUTO_SETUP" != "1" ]]; then
      echo "[vuln-rewrite] LiveCodeBench data not found: $lcb_source" >&2
      echo "[vuln-rewrite] Set LCB_DATA_PATH=/path/to/livecodebench.jsonl or LCB_AUTO_SETUP=1" >&2
      exit 1
    fi

    setup_cmd=(
      "$PYTHON_BIN" scripts/setup_livecodebench.py
      --release "$LCB_RELEASE"
      --output "$LCB_DATA_PATH"
    )
    if [[ -n "$LCB_REPOS_ROOT" ]]; then
      setup_cmd+=(--repos-root "$LCB_REPOS_ROOT")
    fi
    if [[ -n "${LCB_SETUP_LIMIT:-}" ]]; then
      setup_cmd+=(--limit "$LCB_SETUP_LIMIT")
    fi
    echo "[vuln-rewrite] preparing LiveCodeBench: ${setup_cmd[*]}"
    "${setup_cmd[@]}"
  fi
fi

cmd=(
  "$PYTHON_BIN" -u scripts/run_attack_sharded.py
  --dataset "$DATASET"
  --split "$SPLIT"
  --agent "$AGENT"
  --attack "$ATTACK"
  --out-root "$OUT_ROOT"
  --mode "$MODE"
  --config-dir "$CONFIG_DIR"
  --fidelity-mode "$FIDELITY_MODE"
  --shards "$SHARDS"
  --parallel "$PARALLEL"
)

if [[ "$DATASET" == "livecodebench" ]]; then
  cmd+=(--dataset-data-path "$LCB_DATA_PATH")
fi

if [[ -n "${LIMIT:-}" ]]; then
  cmd+=(--limit "$LIMIT")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry-run)
fi

echo "[vuln-rewrite] dataset=$DATASET split=$SPLIT agent=$AGENT attack=$ATTACK"
echo "[vuln-rewrite] out_root=$OUT_ROOT mode=$MODE shards=$SHARDS parallel=$PARALLEL"
if [[ "$DATASET" == "livecodebench" ]]; then
  echo "[vuln-rewrite] livecodebench_data=$LCB_DATA_PATH auto_setup=$LCB_AUTO_SETUP release=$LCB_RELEASE"
  if [[ -n "$LCB_REPOS_ROOT" ]]; then
    echo "[vuln-rewrite] livecodebench_repos_root=$LCB_REPOS_ROOT"
  fi
fi
echo "[vuln-rewrite] claude_region=$ANTHROPIC_VERTEX_REGION project=$ANTHROPIC_VERTEX_PROJECT_ID"
echo "[vuln-rewrite] sweagent_bin=$SWEAGENT_BIN"
echo "[vuln-rewrite] sweagent_default_config=$SWEAGENT_DEFAULT_CONFIG"
echo "[vuln-rewrite] command: ${cmd[*]}"
exec "${cmd[@]}"
