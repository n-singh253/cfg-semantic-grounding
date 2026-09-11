# CFG Semantic Grounding

This branch evaluates pregenerated prompt/patch pairs. It does not run coding
agents, regenerate evaluation attacks, or execute benchmark tests. Results measure
defense classification, not functional correctness or attack success.

Run commands from the repository root. Install the dependencies for your experiment:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[llm,static,gnn]'
```
The base install (`pip install -e .`) is enough for the deterministic prompt
baselines and feature classifier evaluation. Graph building also needs the `gnn`
extra for CodeBERT embeddings, even when the final classifier is tabular.

Download the four-field JSONL files from
[EbonLoa2/cfg-semantic-grounding](https://huggingface.co/datasets/EbonLoa2/cfg-semantic-grounding).
Pin the dataset revision to make the experiment reproducible:
```bash
python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="EbonLoa2/cfg-semantic-grounding",
    repo_type="dataset",
    revision="eb781a02a33e39dfd83199d49d662f09ddfb0c1d",
    allow_patterns=["data/**/rows.jsonl"],
    local_dir=".",
)
PYTHON
```
Rows contain `code_base`, `label` (0 benign, 1 malicious), `prompt`, and `patch`.
Repository setup below creates the base checkouts, not these evaluation rows.
Use dedicated benchmark directories; setup may replace their contents.

Downloading and setting the repositories for each benchmark:
```bash
python scripts/setup_swebench.py \
  --dataset swebench_lite \
  --repos-dir data/repos/swebench \
  --output data/swebench_lite_local.jsonl
```

```bash
python scripts/setup_featurebench.py \
  --variant full \
  --source-repos-dir data/repos/featurebench-source \
  --repos-dir data/repos/featurebench
```

```bash
python scripts/setup_livecodebench.py \
  --release release_latest \
  --output data/livecodebench_code_generation_lite_release_latest.jsonl \
  --repos-dir data/repos/livecodebench
```

Running baselines:
```bash
python -m src.eval.cli list_baselines
```

```bash
python -m src.eval.cli run_defense \
  --rows data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline semgrep \
  --repos-root data/repos/livecodebench \
  --out RES/defense/semgrep_fcv_livecodebench_openhands_qwen3 \
  --workers 4
```

```bash
python -m src.eval.cli run_defense \
  --rows data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline bandit \
  --repos-root data/repos/livecodebench \
  --out RES/defense/bandit_fcv_livecodebench_openhands_qwen3 \
  --workers 4
```

```bash
OPENAI_API_KEY=sk-your-key \
python -m src.eval.cli run_defense \
  --rows data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline llm_judge \
  --out RES/defense/llm_judge_fcv_livecodebench_openhands_qwen3 \
  --workers 4
```

Synthetic data generation:
```bash
python scripts/generate_synthetic_rows.py \
  data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --out RES/synthetic/fcv_livecodebench_openhands_qwen3_openai \
  --provider openai \
  --model gpt-4.1 \
  --workers 4 \
  --on-error skip
```

Build graphs and extract features:
```bash
python -m src.eval.cli run_defense \
  --rows data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline structural_misalignment_build_graph \
  --repos-root data/repos/livecodebench \
  --out RES/graphs/real_fcv_livecodebench_openhands_qwen3_openai \
  --workers 4 \
  --no-resume
```

```bash
OPENAI_API_KEY=sk-your-key \
python -m src.eval.cli run_defense \
  --rows RES/synthetic/fcv_livecodebench_openhands_qwen3_openai/rows_synthetic.jsonl \
  --baseline structural_misalignment_build_graph \
  --repos-root data/repos/livecodebench \
  --out RES/graphs/synthetic_fcv_livecodebench_openhands_qwen3_openai \
  --workers 4 \
  --no-resume
```

Train and evaluate models:
```bash
python -m src.eval.cli run_defense \
  --baseline structural_misalignment_eval_features \
  --out RES/structural_eval/fcv_livecodebench_openhands_qwen3_features \
  --no-resume
```

```bash
python -m src.eval.cli run_defense \
  --baseline structural_misalignment_eval_gnn \
  --out RES/structural_eval/fcv_livecodebench_openhands_qwen3_gnn \
  --no-resume
```

Edit `inputs` in `configs/baselines/structural_misalignment_eval_features.yaml`
or `structural_misalignment_eval_gnn.yaml` when evaluating a different dataset.
Relative input paths are resolved from the YAML directory. The defaults match the
output directories in the examples above.

Synthetic generation is optional training-data augmentation and makes LLM calls.
It is not needed to run the other defenses on the pregenerated evaluation rows.
Structural evaluation trains on synthetic groups and tests on disjoint real
`code_base` groups. The GNN uses the configured final epoch, without selecting
checkpoints from test scores. Inspect `artifacts/load_report.json` and
`artifacts/split_manifest.json` to see exactly which examples were included.

Row runs retry errors when resumed and keep results only for the current rows and
execution settings. Use `--no-resume` after changing repository contents or tool
versions in place. Dataset-level training always recomputes, because cached metrics
cannot establish that the input graphs are unchanged. A CLI exit status of 1 means
at least one row or model failed; inspect each result's `status` and `error`.

Scanner results with missing tools, parse failures, or failed patch application
are errors. With a repository and `max_new_findings`, scanners compare the original
and patched copies. Finding matching uses rule, relative path, and code snippet;
inspect reports when a patch changes the context of an existing warning. Without
`--repos-root`, scanners use added Python snippets, which may be incomplete and
produce parse errors. `edit` decisions from prompt rewriting are advisory: the
saved patch is unchanged, so they are not evidence that a patch was repaired.

See [the submission review](docs/submission_review.md) and
[dataset audit](docs/submission_dataset_audit.json) before full runs: the pinned
dataset has 415 empty benign patches, and four FeatureBench files have no nonempty
benign patches. Those files cannot support the current structural experiment.

Run the offline regression suite with:
```bash
python -m pip install pytest
python -m pytest -q
```
