# CFG Semantic Grounding

Get the rows.jsonl data.

Downloading and setting the repositories for each benchmark:
```bash
python scripts/setup_swebench.py \
  --dataset swebench_lite \
  --repos-dir /proj/arise/arise/hj2742/SWEBench \
  --output /proj/arise/arise/hj2742/cfg-semantic-grounding/data/swebench_lite_local.jsonl
```

```bash
python scripts/setup_featurebench.py \
  --variant full \
  --source-repos-dir /proj/arise/arise/hj2742/FeatureBench-Repo \
  --repos-dir /proj/arise/arise/hj2742/FeatureBench-Instance-Repo
```

```bash
python scripts/setup_livecodebench.py \
  --release release_latest \
  --output /proj/arise/arise/hj2742/cfg-semantic-grounding/data/livecodebench_code_generation_lite_release_latest.jsonl \
  --repos-dir /proj/arise/arise/hj2742/LiveCodeBench-Instance-Repo
```

Running baselines:
```bash
python -m src.eval.cli list_baselines
```

```bash
python -m src.eval.cli run_defense \
  --rows /proj/arise/arise/hj2742/cfg-semantic-grounding/data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline semgrep \
  --repos-root /proj/arise/arise/hj2742/LiveCodeBench-Instance-Repo \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding/RES/defense/semgrep_fcv_livecodebench_openhands_qwen3 \
  --workers 4
```

```bash
python -m src.eval.cli run_defense \
  --rows /proj/arise/arise/hj2742/cfg-semantic-grounding/data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline bandit \
  --repos-root /proj/arise/arise/hj2742/LiveCodeBench-Instance-Repo \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding/RES/defense/bandit_fcv_livecodebench_openhands_qwen3 \
  --workers 4
```

```bash
OPENAI_API_KEY=sk-your-key \
python -m src.eval.cli run_defense \
  --rows /proj/arise/arise/hj2742/cfg-semantic-grounding/data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline llm_judge \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding/RES/defense/llm_judge_fcv_livecodebench_openhands_qwen3 \
  --workers 4
```

Synthetic data generation:
```bash
python scripts/generate_synthetic_rows.py \
  /proj/arise/arise/hj2742/cfg-semantic-grounding/data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding/RES/synthetic/fcv_livecodebench_openhands_qwen3_openai \
  --provider openai \
  --model gpt-4.1 \
  --workers 4 \
  --on-error skip
```

Build graphs and extract features:
```bash
python -m src.eval.cli run_defense \
  --rows /proj/arise/arise/hj2742/cfg-semantic-grounding/data/synthesized_results/Non-Obfuscated/FCV_LiveCodeBench_OpenHands-Qwen3-Coder-30B/rows.jsonl \
  --baseline structural_misalignment_build_graph \
  --repos-root /proj/arise/arise/hj2742/LiveCodeBench-Instance-Repo \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding/RES/graphs/real_fcv_livecodebench_openhands_qwen3_openai \
  --workers 4 \
  --no-resume
```

```bash
HF_HOME=/proj/arise/arise/hj2742/.cache/huggingface \
OPENAI_API_KEY=sk-your-key \
python -m src.eval.cli run_defense \
  --rows /proj/arise/arise/hj2742/cfg-semantic-grounding/RES/synthetic/fcv_livecodebench_openhands_qwen3_openai/rows_synthetic.jsonl \
  --baseline structural_misalignment_build_graph \
  --repos-root /proj/arise/arise/hj2742/LiveCodeBench-Instance-Repo \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding/RES/graphs/synthetic_fcv_livecodebench_openhands_qwen3_openai \
  --workers 4 \
  --no-resume
```

Train and evaluate models:
```bash
python -m src.eval.cli run_defense \
  --baseline structural_misalignment_eval_features \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding/RES/structural_eval/fcv_livecodebench_openhands_qwen3_features \
  --no-resume
```

```bash
python -m src.eval.cli run_defense \
  --baseline structural_misalignment_eval_gnn \
  --out /proj/arise/arise/hj2742/cfg-semantic-grounding/RES/structural_eval/fcv_livecodebench_openhands_qwen3_gnn \
  --no-resume
```