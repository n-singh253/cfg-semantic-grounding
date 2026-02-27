#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${ROOT_DIR}/.venv"
WITH_LLM=0
WITH_STATIC_TOOLS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-llm)
      WITH_LLM=1
      shift
      ;;
    --with-static-tools)
      WITH_STATIC_TOOLS=1
      shift
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: bash scripts/bootstrap_env.sh [--with-llm] [--with-static-tools] [--venv /path/to/venv]"
      exit 1
      ;;
  esac
done

echo "[bootstrap] root=${ROOT_DIR}"
echo "[bootstrap] venv=${VENV_PATH}"

python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${ROOT_DIR}"
python "${ROOT_DIR}/scripts/init_structural_model.py"

if [[ "${WITH_LLM}" -eq 1 ]]; then
  python -m pip install -e "${ROOT_DIR}[llm]"
fi

if [[ "${WITH_STATIC_TOOLS}" -eq 1 ]]; then
  python -m pip install bandit semgrep
fi

echo
echo "[bootstrap] done"
echo "Activate with: source ${VENV_PATH}/bin/activate"
echo "Check prereqs: python ${ROOT_DIR}/scripts/check_prereqs.py --dataset toy"
