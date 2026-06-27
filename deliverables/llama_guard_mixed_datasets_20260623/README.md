# Portable Llama Guard datasets

This bundle contains 16 mixed evaluation datasets (8,508 rows total). Every
dataset includes clean rows (`mixed_eval_label=0`) and one attack type
(`mixed_eval_label=1`). Prompts and adversarial patches are embedded as
`portable_prompt` and `portable_adv_patch`; the original local paths remain only
as provenance and are not used by the portable runner.

## Run from a cfg-semantic-grounding checkout

Extract this ZIP, then from the repository root run:

```bash
.venv/bin/python -u PATH_TO_EXTRACTED_BUNDLE/run_portable_llama_guard_bundle.py \
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
.venv/bin/python scripts/export_paper_table.py \
  --root outputs/baselines \
  --swebench-primary-metric accuracy
```

To send results back, ZIP the directory
`outputs/baselines/portable_llama_guard/`. Do not run multiple processes on the
same output directory; use `--only` filters with separate GPUs/output roots if
parallel execution is needed.
