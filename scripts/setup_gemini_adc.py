#!/usr/bin/env python3
"""Store Gemini/Vertex ADC credentials outside git-tracked paths.

Pass the JSON on stdin. The script validates the shape, writes with chmod 600,
and prints export commands without echoing secret values.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Dict


DEFAULT_PATH = Path.home() / ".config" / "cfg-semantic-grounding" / "gemini_adc.json"
REPO_FALLBACK = Path(__file__).resolve().parents[1] / ".secrets" / "gemini_adc.json"


def _load_stdin_json() -> Dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        raise SystemExit("No JSON received on stdin.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("Credential payload must be a JSON object.")
    cred_type = str(payload.get("type", ""))
    if cred_type not in {"authorized_user", "service_account"}:
        raise SystemExit("Expected credential type 'authorized_user' or 'service_account'.")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Store Gemini ADC credentials safely.")
    parser.add_argument("--path", default=str(DEFAULT_PATH), help="Destination path; defaults outside this repo.")
    parser.add_argument("--repo-fallback", action="store_true", help="Write to ignored .secrets/gemini_adc.json instead.")
    parser.add_argument("--project", default="", help="Optional GOOGLE_CLOUD_PROJECT override.")
    parser.add_argument("--location", default="global", help="GOOGLE_CLOUD_LOCATION value.")
    args = parser.parse_args()

    payload = _load_stdin_json()
    dest = REPO_FALLBACK if args.repo_fallback else Path(args.path).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(dest, stat.S_IRUSR | stat.S_IWUSR)

    project = args.project or str(payload.get("quota_project_id") or "")
    print(f"Wrote ADC credentials to: {dest}")
    print("Add these exports before running attacks:")
    print(f"export GOOGLE_APPLICATION_CREDENTIALS={dest}")
    if project:
        print(f"export GOOGLE_CLOUD_PROJECT={project}")
        print(f"export VERTEXAI_PROJECT={project}")
    print(f"export GOOGLE_CLOUD_LOCATION={args.location}")
    print(f"export VERTEXAI_LOCATION={args.location}")
    print("export GOOGLE_GENAI_USE_VERTEXAI=true")
    print("export MSWEA_MODEL_NAME=vertex_ai/gemini-3-flash-preview")
    print("export MSWEA_COST_TRACKING=ignore_errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
