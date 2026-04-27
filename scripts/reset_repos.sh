#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOS_ROOT="${CFG_SWEBENCH_REPOS_DIR:-${ROOT_DIR}/data/repos/swebench_lite}"
DRY_RUN=false

log() {
  printf '[reset-repos] %s\n' "$*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --repos-root)
      REPOS_ROOT="$2"
      shift 2
      ;;
    *)
      REPOS_ROOT="$1"
      shift
      ;;
  esac
done

if [[ ! -d "$REPOS_ROOT" ]]; then
  log "error | repos_root=${REPOS_ROOT} reason=not_found"
  exit 1
fi

log "start | repos_root=${REPOS_ROOT} dry_run=${DRY_RUN}"

ok=0
skipped=0
errors=0

for repo_dir in "$REPOS_ROOT"/*/; do
  [[ -d "$repo_dir/.git" ]] || { ((skipped++)) || true; continue; }

  if [[ "$DRY_RUN" == true ]]; then
    log "would-reset | repo=${repo_dir}"
    ((ok++)) || true
    continue
  fi

  if git -C "$repo_dir" checkout -- . >/dev/null 2>&1 && git -C "$repo_dir" clean -fd -q >/dev/null 2>&1; then
    log "reset | repo=${repo_dir}"
    ((ok++)) || true
  else
    log "error | repo=${repo_dir} reason=reset_failed"
    ((errors++)) || true
  fi
done

log "done | ok=${ok} skipped=${skipped} errors=${errors}"
[[ $errors -eq 0 ]]
