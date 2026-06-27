#!/usr/bin/env python3
"""Build the portable Llama Guard dataset handoff requested for GPU evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


DATASETS = [
    # SWE-Bench non-obfuscated.
    ("swebench_nonobfuscated_gemini_fcv78", "gemini3_flash_swebench_imported", "swebench_none_vs_fcv_cwe78"),
    ("swebench_nonobfuscated_gemini_swexploit", "gemini3_flash_swebench_imported", "swebench_none_vs_swexploit_anthropic"),
    ("swebench_nonobfuscated_claude_fcv78", "claude_sonnet46_swebench_imported", "swebench_none_vs_fcv_cwe78"),
    ("swebench_nonobfuscated_claude_swexploit", "claude_sonnet46_swebench_imported", "swebench_none_vs_swexploit_anthropic"),
    # FeatureBench non-obfuscated.
    ("featurebench_nonobfuscated_gemini_fcv78", "gemini3_flash_socal", "featurebench_full_none_vs_fcv_cwe78"),
    ("featurebench_nonobfuscated_gemini_swexploit", "gemini3_flash_socal", "featurebench_full_none_vs_swexploit_anthropic"),
    ("featurebench_nonobfuscated_claude_fcv78", "claude_sonnet46_socal", "featurebench_full_none_vs_fcv_cwe78"),
    ("featurebench_nonobfuscated_claude_swexploit", "claude_sonnet46_socal", "featurebench_full_none_vs_swexploit_anthropic"),
    # LiveCodeBench Base64-obfuscated.
    ("livecodebench_obfuscated_gemini_fcv78", "gemini3_flash", "livecodebench_none_vs_fcv_cwe78_base64_obfuscated"),
    ("livecodebench_obfuscated_gemini_swexploit", "gemini3_flash", "livecodebench_none_vs_swexploit_base64_obfuscated"),
    ("livecodebench_obfuscated_claude_fcv78", "claude37_sonnet_sweagent", "livecodebench_none_vs_fcv_cwe78_base64_obfuscated"),
    ("livecodebench_obfuscated_claude_swexploit", "claude37_sonnet_sweagent", "livecodebench_none_vs_swexploit_base64_obfuscated"),
    # SWE-Bench Lite Base64-obfuscated.
    ("swebench_obfuscated_gemini_fcv78", "gemini3_flash", "swebench_lite_none_vs_fcv_cwe78_base64_obfuscated"),
    ("swebench_obfuscated_gemini_swexploit", "gemini3_flash", "swebench_lite_none_vs_swexploit_base64_obfuscated"),
    ("swebench_obfuscated_claude_fcv78", "claude37_sonnet_sweagent", "swebench_lite_none_vs_fcv_cwe78_base64_obfuscated"),
    ("swebench_obfuscated_claude_swexploit", "claude37_sonnet_sweagent", "swebench_lite_none_vs_swexploit_base64_obfuscated"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _inline_row(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    prompt_path = _resolve(root, str(row.get("attack_artifact_path", ""))) / "adv_prompt.txt"
    patch_path = _resolve(root, str(row.get("patch_artifacts", {}).get("adv_patch_path", "")))
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Missing prompt artifact: {prompt_path}")
    if not patch_path.is_file():
        raise FileNotFoundError(f"Missing patch artifact: {patch_path}")
    return {
        **row,
        "portable_prompt": prompt_path.read_text(encoding="utf-8"),
        "portable_adv_patch": patch_path.read_text(encoding="utf-8"),
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _result_relpath(dataset_id: str) -> str:
    parts = dataset_id.split("_")
    family = "_".join(parts[:2])
    agent = "gemini" if "_gemini_" in dataset_id else "claude"
    attack = "fcv78" if dataset_id.endswith("_fcv78") else "swexploit"
    return f"portable_llama_guard/{family}/{agent}/{attack}"


def _readme(total_rows: int) -> str:
    return f"""# Portable Llama Guard datasets

This bundle contains 16 mixed evaluation datasets ({total_rows:,} rows total). Every
dataset includes clean rows (`mixed_eval_label=0`) and one attack type
(`mixed_eval_label=1`). Prompts and adversarial patches are embedded as
`portable_prompt` and `portable_adv_patch`; the original local paths remain only
as provenance and are not used by the portable runner.

## Run from a cfg-semantic-grounding checkout

Extract this ZIP, then from the repository root run:

```bash
.venv/bin/python -u PATH_TO_EXTRACTED_BUNDLE/run_portable_llama_guard_bundle.py \\
  PATH_TO_EXTRACTED_BUNDLE
```

The run is resumable and loads the model only once. Results are written beneath
`outputs/baselines/portable_llama_guard/`. Use `--only featurebench`,
`--only obfuscated_gemini`, or another manifest-ID substring to run a subset.

The current project configuration evaluates `input_field=prompt` (the default in
`configs/baselines/llama_guard.yaml`). The bundle also contains patches so the
teammate can explicitly test `--input-field patch` or `--input-field
prompt_and_patch` if desired.

This runner is intentionally classification-only: it does not apply patches or
run repository tests. Therefore export SWE-Bench with the requested accuracy
metric, not the operational metric:

```bash
.venv/bin/python scripts/export_paper_table.py \\
  --root outputs/baselines \\
  --swebench-primary-metric accuracy
```

To send results back, ZIP the directory
`outputs/baselines/portable_llama_guard/`. Do not run multiple processes on the
same output directory; use `--only` filters with separate GPUs/output roots if
parallel execution is needed.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("deliverables/llama_guard_mixed_datasets_20260623.zip"),
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    stage = args.out.with_suffix("")
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "datasets").mkdir(parents=True)

    manifest_entries: list[dict[str, Any]] = []
    total_rows = 0
    for dataset_id, model_key, mixed_name in DATASETS:
        source = root / "outputs" / "attacks" / model_key / "mixed" / mixed_name / "full" / "attack_dataset.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"Missing mixed dataset: {source}")
        destination = stage / "datasets" / f"{dataset_id}.jsonl"
        counts: Counter[str] = Counter()
        with destination.open("w", encoding="utf-8") as handle:
            for row in _read_rows(source):
                portable = _inline_row(root, row)
                handle.write(json.dumps(portable, sort_keys=True) + "\n")
                counts["rows"] += 1
                counts["clean" if int(row.get("mixed_eval_label", 0)) == 0 else "attack"] += 1
                counts[str(row.get("attack_name", "unknown"))] += 1
        total_rows += counts["rows"]
        entry = {
            "id": dataset_id,
            "path": str(destination.relative_to(stage)),
            "result_relpath": _result_relpath(dataset_id),
            "source": str(source.relative_to(root)),
            "rows": counts["rows"],
            "clean_rows": counts["clean"],
            "attack_rows": counts["attack"],
            "attack_name_counts": {
                key: value for key, value in sorted(counts.items()) if key not in {"rows", "clean", "attack"}
            },
            "sha256": _sha256(destination),
        }
        manifest_entries.append(entry)
        print(
            f"[bundle] {dataset_id}: rows={entry['rows']} clean={entry['clean_rows']} attack={entry['attack_rows']}",
            flush=True,
        )

    manifest = {
        "bundle_version": 1,
        "purpose": "Portable classifier-only Llama Guard evaluation",
        "total_datasets": len(manifest_entries),
        "total_rows": total_rows,
        "datasets": manifest_entries,
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (stage / "README.md").write_text(_readme(total_rows), encoding="utf-8")
    shutil.copy2(root / "scripts" / "run_portable_llama_guard_bundle.py", stage)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(path, Path(stage.name) / path.relative_to(stage))
    print(f"[bundle] wrote {args.out} ({args.out.stat().st_size / (1024 * 1024):.1f} MiB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
