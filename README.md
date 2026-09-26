# SIGIL

This repository evaluates defenses against adversarial instructions in software-development tasks. It starts from previously generated prompt/patch pairs for SWE-bench and LiveCodeBench; it does not run coding agents or generate the original attack patches. The available defenses are Bandit, Semgrep, an LLM judge, Llama Guard, and structural misalignment models trained on prompt-to-code graphs.

## Installation

Run commands from the repository root. Python 3.12 and Git are recommended. The supplied Conda environment installs the scanner, LLM, and graph dependencies, including a CUDA-enabled PyTorch build:

```bash
conda env create -f environment.yml
conda activate cfg-semantic-grounding
python -m src.eval.cli list_baselines
```

For OpenAI-backed baselines and graph generation, set `OPENAI_API_KEY` in your shell. Do not put credentials in config files or commit them. Llama Guard additionally requires access to `meta-llama/Llama-Guard-4-12B` on Hugging Face; authenticate with `hf auth login` before running it. Model downloads may require substantial disk space.

## Input Data

Experiment inputs are under `data/synthesized_results/{Non-Obfuscated,Obfuscated}/<run>/rows.jsonl`. Each JSON line contains a `code_base`, `prompt`, `patch`, and binary `injection` field. The distributed data may also include `adv` (whether the patch implements the injection) and, for obfuscated runs, `obfuscated`. Row-level defenses read the prompt and patch and retain `injection` as the row label; they do **not** calculate accuracy or use `adv`/`obfuscated` to filter examples. Apply any experiment-specific filtering when analyzing the recorded decisions.

For example, this file contains one LiveCodeBench run:

```text
data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl
```

`--rows` accepts a JSONL file or a directory containing row files. Use `--recursive` to search nested directories, `--limit N` for a smoke test, and `--no-resume` to recompute an output directory. Each command below writes to a distinct directory under the Git-ignored `outputs/` tree.

## Benchmark Repositories

Bandit, Semgrep, and graph construction need local benchmark repositories. Prepare only the benchmark you intend to evaluate:

```bash
python scripts/setup_swebench.py \
  --dataset swebench_lite \
  --repos-dir outputs/repos/SWEBench \
  --workers 8

python scripts/setup_livecodebench.py \
  --release release_latest \
  --repos-dir outputs/repos/LiveCodeBench \
  --workers 8
```

Pass the corresponding `--repos-dir` as `--repos-root` when running a defense. Each setup script also writes a local benchmark manifest alongside its prepared repositories; the defense input remains the experiment's `rows.jsonl`, **not** that manifest. These directories must be dedicated to the benchmark: repository-backed stages reset Git checkouts and may remove untracked files.

## Run Baselines

The CLI loads a baseline from `configs/baselines/<name>.yaml`. The following examples use the LiveCodeBench run above:

```bash
python -m src.eval.cli run_defense \
  --rows data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline bandit \
  --repos-root outputs/repos/LiveCodeBench \
  --out outputs/defense/bandit_fcv_lcb \
  --workers 4

python -m src.eval.cli run_defense \
  --rows data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline semgrep \
  --repos-root outputs/repos/LiveCodeBench \
  --out outputs/defense/semgrep_fcv_lcb \
  --workers 4

python -m src.eval.cli run_defense \
  --rows data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline llm_judge \
  --out outputs/defense/llm_judge_fcv_lcb \
  --workers 4

python -m src.eval.cli run_defense \
  --rows data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline llama_guard \
  --out outputs/defense/llama_guard_fcv_lcb \
  --workers 1
```

Bandit and Semgrep compare findings before and after applying each patch to an isolated copy of its repository. Their commands and decision thresholds are configurable in `configs/baselines/`. The LLM judge's provider, model, and judgment mode are configured there as well. Llama Guard uses the `prompt_and_patch` input format by default and runs with one effective worker per process.

Row-level outputs contain `results.jsonl` (one decision, status, and signal record per row), `integration_spec.json` (resolved run configuration), and `artifacts/` (method-specific reports, prompts, or model responses). An `error` decision is a failed evaluation, not an accept or reject. Runs resume by default when the row and baseline configuration hashes match.

## Structural Misalignment

The structural pipeline has three stages: build graphs from real rows, generate synthetic adversarial graph examples from successfully parsed benign graphs, then train and evaluate a feature classifier or GNN. The synthetic stage inserts prompt and code **nodes**; it does not produce an executable synthetic patch or a new `rows.jsonl`.

1. Build real graphs and extract structural features:

   ```bash
   python -m src.eval.cli run_defense \
     --rows data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
     --baseline structural_misalignment_build_graph \
     --repos-root outputs/repos/LiveCodeBench \
     --out outputs/graphs/real_fcv_lcb \
     --workers 4
   ```

2. Generate synthetic examples from the benign graph results:

   ```bash
   python scripts/generate_synthetic_graphs.py \
     outputs/graphs/real_fcv_lcb/results.jsonl \
     --out outputs/graphs/synthetic_fcv_lcb \
     --provider openai \
     --model gpt-4o-mini \
     --workers 4
   ```

3. Edit `inputs.data_rows`, `inputs.data_graph_dir`, and `inputs.synthetic_graph_dir` in both `configs/baselines/structural_misalignment_eval_features.yaml` and `configs/baselines/structural_misalignment_eval_gnn.yaml`. Paths in these configs resolve relative to `configs/baselines/`, so use the following values for this example (replacing the checked-in machine-specific values):

   ```yaml
   inputs:
     data_rows: ../../data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl
     data_graph_dir: ../../outputs/graphs/real_fcv_lcb
     synthetic_graph_dir: ../../outputs/graphs/synthetic_fcv_lcb
   ```

   Then run either or both evaluators:

   ```bash
   python -m src.eval.cli run_defense \
     --baseline structural_misalignment_eval_features \
     --out outputs/eval/features_fcv_lcb

   python -m src.eval.cli run_defense \
     --baseline structural_misalignment_eval_gnn \
     --out outputs/eval/gnn_fcv_lcb
   ```

Graph construction parses prompt subtasks and patch CFG/code nodes, links subtasks to code, embeds nodes with CodeBERT, and writes graph and feature artifacts. The build configuration selects the parsers, linker, LLM, and embedding device. Synthetic generation requires successful benign graphs with usable prompt and code nodes; `--on-error skip` allows it to continue past individual generation failures.

Structural evaluation splits by `code_base` so related examples do not cross train and test sets. It trains on synthetic graphs and tests on held-out real graphs for code bases represented in both collections. Its `results.jsonl` contains model-level results; `artifacts/split_manifest.json` and `artifacts/load_report.json` document the split and excluded examples. A small `--limit` smoke test may not leave enough eligible code bases for training.
