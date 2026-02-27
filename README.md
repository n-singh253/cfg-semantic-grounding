# CFG Semantic Grounding Harness

`cfg-semantic-grounding` is an API-exact SWE-Bench harness for comparing patch agents, attacks, and defenses with reproducible artifacts.

Public APIs used by the runner:
- `attack(repo_code, ori_prompt, all_tests) -> adv_prompt`
- `defense(prompt, code_or_patch, all_tests, repo_code) -> True | False | new_code`

`baseline/` is the defense package. Defense return semantics:
- `True`: accept attacked patch unchanged
- `False`: reject patch
- `new_code`: edit path (unified diff => edited patch, otherwise treated as edited prompt and agent reruns once)

## Repository Structure

```text
src/
  attack/      # attack plugins (LLM-driven by default)
  baseline/    # defense plugins (advisor defense API)
  agent/       # patch-agent wrappers
  dataset/     # dataset adapters
  eval/        # runner + CLI
  common/      # shared cfg/grounding/features/models/security/llm
configs/
  datasets/ agents/ attacks/ baselines/ runs/
scripts/       # bootstrap + helper scripts
data/          # local dataset/model artifacts
outputs/       # run artifacts/results
```

## Quickstart

```bash
cd /cfg-semantic-grounding
bash scripts/bootstrap_env.sh
source .venv/bin/activate
```

Optional extras:

```bash
bash scripts/bootstrap_env.sh --with-llm --with-static-tools
```

## Required Environment Variables (LLM Runs)

Set based on provider:
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `ANTHROPIC_API_KEY`

Example:

```bash
export OPENAI_API_KEY="<key>"
```

## Configure Datasets

Dataset YAML files are in `configs/datasets/`.

For non-toy datasets, set `data_path` in:
- `configs/datasets/lite.yaml`
- `configs/datasets/pro.yaml`
- `configs/datasets/plus.yaml`

Helper script for local starter data:

```bash
python scripts/init_dataset.py \
  --dataset swebench_lite \
  --repo-path /abs/path/to/local/repo \
  --base-commit HEAD \
  --output data/swebench_lite_local.jsonl
```

## Configure Agent CLIs

Agent command wrappers are configured in `configs/agents/*.yaml`.
If the CLI binary is on your `PATH`, the wrapper can call it.

Behavior when CLI is missing is controlled by each agent config (`missing_tool_behavior: fail|skip`).

## CLI Commands

```bash
python -m src.eval.cli --help
python -m src.eval.cli run_one --help
python -m src.eval.cli run_matrix --help
python -m src.eval.cli list_baselines
```

### `run_one`

```bash
python -m src.eval.cli run_one \
  --dataset toy \
  --agent dummy \
  --attack none \
  --baseline prompt_filter \
  --out outputs/runs/toy1
```

### Structural Defense Smoke

```bash
python -m src.eval.cli run_one \
  --dataset toy \
  --agent dummy \
  --attack none \
  --baseline structural_misalignment \
  --out outputs/runs/toy_struct
```

### Real Smoke (requires dataset + tools)

```bash
python -m src.eval.cli run_one \
  --dataset swebench_lite \
  --split test \
  --limit 1 \
  --agent openhands \
  --attack none \
  --baseline structural_misalignment \
  --fidelity-mode llm \
  --out outputs/runs/smoke_structural
```

### Matrix Run

```bash
python -m src.eval.cli run_matrix \
  --config configs/runs/example_matrix.yaml \
  --out outputs/runs/matrix_toy
```

## Pipeline Order (Exact)

1. load instance (`repo_code`, `ori_prompt`, `all_tests`)
2. `ori_patch = agent(repo_code, ori_prompt, all_tests)`
3. `adv_prompt = attack(repo_code, ori_prompt, all_tests)`
4. `adv_patch = agent(repo_code, adv_prompt, all_tests)`
5. `decision = defense(adv_prompt, adv_patch, all_tests, repo_code)`
6. resolve decision: accept/reject/edit
7. apply final patch (if not rejected)
8. run tests
9. run static checks
10. optional LLM judges
11. write artifacts + `results.jsonl`

Edit resolution:
- unified diff => edited patch
- otherwise => edited prompt, rerun agent once

## Command Option Tables

### `--dataset`

| Value | Config file |
|---|---|
| `toy` | `configs/datasets/toy.yaml` |
| `swebench_lite` | `configs/datasets/lite.yaml` |
| `swebench_pro` | `configs/datasets/pro.yaml` |
| `swebench_plus` | `configs/datasets/plus.yaml` |

### `--agent`

| Value | Config file | External CLI |
|---|---|---|
| `dummy` | `configs/agents/dummy.yaml` | none |
| `dummy2` | `configs/agents/dummy2.yaml` | none |
| `minisweagent` | `configs/agents/minisweagent.yaml` | `minisweagent` |
| `sweagent` | `configs/agents/sweagent.yaml` | `sweagent` |
| `openhands` | `configs/agents/openhands.yaml` | `openhands` |
| `claude_code` | `configs/agents/claude_code.yaml` | `claude-code` |
| `gemini_cli` | `configs/agents/gemini_cli.yaml` | `gemini-cli` |

### `--attack`

| Value | Config file |
|---|---|
| `none` | `configs/attacks/none.yaml` |
| `bug_reports` | `configs/attacks/bug_reports.yaml` |
| `udora` | `configs/attacks/udora.yaml` |
| `swexploit` | `configs/attacks/swexploit.yaml` |
| `fcv` | `configs/attacks/fcv.yaml` |

### `--baseline` (defense)

| Value | Config file |
|---|---|
| `prompt_filter` | `configs/baselines/prompt_filter.yaml` |
| `prompt_rewrite` | `configs/baselines/prompt_rewrite.yaml` |
| `agentic_guard` | `configs/baselines/agentic_guard.yaml` |
| `llm_judge` | `configs/baselines/llm_judge.yaml` |
| `bandit` | `configs/baselines/bandit.yaml` |
| `semgrep` | `configs/baselines/semgrep.yaml` |
| `structural_misalignment` | `configs/baselines/structural_misalignment.yaml` |

### `--fidelity-mode`

| Value | Meaning |
|---|---|
| `llm` | paper-faithful LLM path |
| `surrogate_debug` | deterministic/mock fallback path |

## Structural Misalignment Defense

`baseline=structural_misalignment` ports the old methodology:
- CFG diff extraction from patch against repo snapshot
- LLM subtask decomposition from prompt
- LLM subtask->CFG grounding
- structural metrics + optional similarity features
- model inference with explicit `decision_policy`
- Task8 modes primary, Task7 modes optional

Default severity behavior is universal-only (`severity_mode: universal`).
Dataset-marker patterns require explicit debug enablement.

Model artifact note:
- `model_path` / `model_paths` in `configs/baselines/structural_misalignment.yaml` points to a sklearn model bundle directory (or file under that directory).
- Missing/incompatible model artifacts fail clearly and are recorded in `results.jsonl -> defense_signals.error` and `failure_flags.model_missing`.
- No silent heuristic fallback is used on the primary path.

## Output Artifacts

Per run:
- `outputs/runs/<run_id>/integration_spec.json`
- `outputs/runs/<run_id>/dataset_report.json`
- `outputs/runs/<run_id>/results.jsonl`
- `outputs/runs/<run_id>/logs/`
- `outputs/runs/<run_id>/artifacts/`

Patch artifacts per instance:
- `artifacts/patches/<instance_id>/ori_patch.diff`
- `artifacts/patches/<instance_id>/adv_patch.diff`
- `artifacts/patches/<instance_id>/final_patch.diff` (if edited)

Structural defense artifacts:
- `artifacts/defenses/<instance_id>/structural_misalignment/cfg_stats.json`
- `.../subtasks.json`
- `.../grounding.json`
- `.../features.json`
- `.../model_output.json`
- `.../severity.json` (Task7 modes)

Progressive save + resume:
- LLM stages (attacks, subtasks, grounding, judges) write prompt/response/metadata artifacts immediately and cache by deterministic key.
- `results.jsonl` is append-only with flush/fsync per row.
- Re-running the same config in the same `--out` directory reuses cached LLM calls and skips already-completed instances.

## Matrix Configs

- `configs/runs/example_matrix.yaml`: local toy smoke matrix
- `configs/runs/swebench_lite_starter.yaml`: starter real matrix template

## Helper Scripts

- `scripts/bootstrap_env.sh`: create venv/install deps
- `scripts/check_prereqs.py`: tool/key/data checks
- `scripts/init_dataset.py`: create starter local dataset rows
- `scripts/init_structural_model.py`: create local demo structural model bundle

## Port Mapping Audit (Old -> New)

This defense port was copied/adapted from the previous repo modules into the new harness layout:

| Old repo source | New repo destination |
|---|---|
| `utils/cfg_extractor.py` | `src/common/cfg/build.py` |
| `utils/cfg_diff.py` | `src/common/cfg/diff.py` |
| `utils/cfg_grounding.py` | `src/common/cfg/diff.py` + `src/common/grounding/*` |
| `utils/llm_clients.py` (subtasks/linking prompt/parsing behavior) | `src/common/grounding/subtasks.py`, `src/common/grounding/link.py`, `src/common/grounding/schemas.py` |
| `utils/misalignment_features.py` | `src/common/features/task8.py` |
| `utils/task7_feature_extractor.py` | `src/common/features/task7.py` |
| `utils/security_filters.py` | `src/common/security/patterns.py`, `src/common/security/severity.py` |
| `scripts/run_attack_suite.py` model-bundle/eval-only guardrails | `src/common/models/load.py`, `src/common/models/infer.py`, `src/common/models/train.py`, `src/baseline/structural_misalignment.py` |
