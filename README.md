# CFG Semantic Grounding

```bash
conda env create -f environment.yml
conda activate cfg-semantic-grounding
```

```bash
hf download EbonLoa2/cfg-semantic-grounding \
  --repo-type dataset \
  --include "data/synthesized_results/**" \
  --local-dir /proj/arise/arise/hj2742/cfg-semantic-grounding-submission
```

Downloading and setting the repositories for all three benchmarks:
```bash
python scripts/setup_swebench.py --repos-dir /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/repos/SWEBench --workers 8
python scripts/setup_featurebench.py --source-repos-dir /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/repos/FeatureBench/source_repos --repos-dir /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/repos/FeatureBench/instance_repos --workers 8
python scripts/setup_livecodebench.py --repos-dir /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/repos/LiveCodeBench --workers 8
```

Running baselines:
```bash
python -m src.eval.cli list_baselines
```

```bash
python -m src.eval.cli run_defense \
  --rows /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline semgrep \
  --repos-root /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/repos/LiveCodeBench \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/RES/defense/semgrep_fcv_livecodebench_openhands_qwen3 \
  --workers 4
```

```bash
python -m src.eval.cli run_defense \
  --rows /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline bandit \
  --repos-root /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/repos/LiveCodeBench \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/RES/defense/bandit_fcv_livecodebench_openhands_qwen3 \
  --workers 4
```

```bash
OPENAI_API_KEY=sk-your-key \
python -m src.eval.cli run_defense \
  --rows /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline llm_judge \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/RES/defense/llm_judge_fcv_livecodebench_openhands_qwen3 \
  --workers 4
```

Build real graphs and extract features:
```bash
HF_HOME=/proj/arise/arise/hj2742/.cache/huggingface \
OPENAI_API_KEY=sk-your-key \
python -m src.eval.cli run_defense \
  --rows /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline structural_misalignment_build_graph \
  --repos-root /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/repos/LiveCodeBench \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/RES/graphs/real_fcv_livecodebench_openhands_qwen3_openai \
  --workers 4 \
  --no-resume
```

Generate synthetic graphs from the benign real results:
```bash
HF_HOME=/proj/arise/arise/hj2742/.cache/huggingface \
OPENAI_API_KEY=sk-your-key \
python scripts/generate_synthetic_graphs.py \
  /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/RES/graphs/real_fcv_livecodebench_openhands_qwen3_openai/results.jsonl \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/RES/graphs/synthetic_fcv_livecodebench_openhands_qwen3_openai \
  --provider openai \
  --model gpt-4o-mini \
  --workers 4
```

Train and evaluate models:
```bash
python -m src.eval.cli run_defense \
  --baseline structural_misalignment_eval_features \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/RES/structural_eval/fcv_livecodebench_openhands_qwen3_features \
  --no-resume
```

```bash
python -m src.eval.cli run_defense \
  --baseline structural_misalignment_eval_gnn \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding-submission/RES/structural_eval/fcv_livecodebench_openhands_qwen3_gnn \
  --no-resume
```
