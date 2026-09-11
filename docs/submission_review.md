# Submission experiment review — 2026-09-11

The lightweight architecture makes sense: evaluation consumes pregenerated
`code_base`, `label`, `prompt`, and `patch` rows, while the old attack and coding-agent
pipeline is removed. However, the original `submission` head was not ready for
reliable experiments. The code fixes in this review address reproducible failures
and measurement errors. Missing data still blocks some structural experiments.

Reviewed GitHub `submission` at `088a567` against `main` at `ebb7205`. Audited all
41 JSONL files at Hugging Face revision
`eb781a02a33e39dfd83199d49d662f09ddfb0c1d` of
[EbonLoa2/cfg-semantic-grounding](https://huggingface.co/datasets/EbonLoa2/cfg-semantic-grounding/tree/eb781a02a33e39dfd83199d49d662f09ddfb0c1d).
The complete per-file counts are in [submission_dataset_audit.json](submission_dataset_audit.json).

## Blocking dataset finding: missing benign patches

All 32,616 rows satisfy the four-field schema. However, **415 benign rows across
16 files have empty patches**. No malicious row has an empty patch. All nonempty
patches pass the harness's diff sanitizer; that does not establish that they apply
to the correct base commit or preserve task behavior.

In these four `Obfuscated/` datasets, **every benign patch is empty**:

| Dataset directory | Benign rows with empty patches | Malicious rows |
| --- | ---: | ---: |
| `FCV-78_FeatureBench_MINI-Gemini-3` | 98 / 98 | 98 |
| `FCV-78_FeatureBench_SWEAgent-Claude-3.7` | 46 / 46 | 46 |
| `SWExploit_FeatureBench_MINI-Gemini-3` | 64 / 64 | 64 |
| `SWExploit_FeatureBench_SWEAgent-Claude-3.7` | 37 / 37 | 37 |

`StructuralMisalignmentBuildGraphDefense.defense` fails at `empty_patch`. As a
result, these files produce no benign structural examples, and the training/test
validation requiring both classes fails. The synthetic generator also has no
usable benign patch from which to produce faithful training augmentation.

Restore the actual generated benign patches from the source experiment artifacts
before using these files for patch/graph comparisons. If an empty patch is an
intentional no-op outcome, define a separate no-op representation and reporting
policy; treating it as a missing graph silently removes one class. The review does
not invent patches, relabel rows, or modify the published dataset.

Across files there are also 1,479 malicious-only and 3,203 benign-only `code_base`
groups (counts are summed per file, not globally unique). Therefore, do not assume
all files contain balanced pairs. The original synthetic generator used a malicious
row as the supposed benign seed when no benign row existed. It now requires a
nonempty benign seed and follows the configured error policy otherwise.

## Code findings and fixes

| Priority | Original behavior and consequence | Fix |
| --- | --- | --- |
| High | GNN training chose the best epoch using test ROC-AUC/accuracy, biasing the reported test result. This behavior was inherited from `main`. | Train for the configured epoch count; evaluate the test set once afterward and save the final checkpoint. |
| High | Graph loading matched only row position and label. Filtered, reordered, or regenerated rows could load another example's graph. Repeated result records could duplicate samples. | Match `code_base`, label, and prompt/patch hashes; check training-artifact identity, deduplicate matching records, reject mixed graph configs. Feature columns come only from training examples. |
| High | Missing scanners, scanner parse errors, missing reports, and scanner failures could become successful accept/reject decisions. Empty patches could scan an empty directory successfully. | Record these conditions as errors; remove stale reports before scanning; return CLI exit status 1 when any row/model fails. Invalid LLM-judge output also errors instead of defaulting to acceptance. |
| High | `max_new_findings` was applied to all findings in the patched repository, rejecting harmless patches due to existing vulnerabilities. | Scan original and patched isolated copies, then compare findings by rule, path, and code snippet. Preserve total and baseline counts for auditing. |
| High | Default resume skipped failed rows, ignored fidelity/repository/timeout changes, and retained stale or duplicate results. Dataset-level resume reused metrics whenever YAML matched, regardless of changed inputs. | Retry failures, include execution settings in row keys, atomically retain one current successful result per row, and always recompute dataset-level training. |
| Medium | Static scans reset/cleaned shared source repositories for each row, creating races with multiple workers and deleting untracked files. Failed patch application silently changed the experiment to scanning snippets. | Keep source repositories unchanged; fail on a missing requested repository or failed patch application. Snippet paths are confined to their workspace. |
| Medium | A missing structural repository path resolved to the harness's current directory. The resolver could also substitute an upstream clone's unrelated HEAD for an instance's base. | Pass `None` when no path is supplied and resolve only per-instance repository layouts. |
| Medium | Transformer guard models were initialized for every row. Thousands of weight loads make experiments unnecessarily slow and concurrent loads can exhaust memory. | Reuse one plugin instance per worker and reset row-local signals. Use one worker for large GPU guards unless memory supports more model copies. |
| Medium | README and evaluation YAML used personal absolute paths; YAML pointed at smoke outputs different from the README's outputs. | Document installation and a pinned dataset download, use portable paths, and align evaluation inputs with the examples. |
| Low | `load_rows(limit=0)` returned one row. | Return zero rows and test the boundary. |

## Remaining experimental constraints

- The branch measures defense classification on saved candidate patches. It does
  not remeasure functional correctness or attack success. Four-field rows do not
  include test outcomes, attack success evidence, or base commits; retain the
  original provenance separately if those claims are needed.
- Synthetic generation is an optional, paid training augmentation step still
  present in the lightweight branch. It is not required for the ordinary
  pregenerated-row baselines. Generated patch application/functionality is not
  validated by that script.
- Structural evaluation selects code bases with surviving graphs in both real and
  synthetic inputs, then splits them. Generation or graph failures can change the
  evaluated population. Report exclusions and class counts; establish a fixed
  evaluation cohort before comparing classifiers or reporting paper numbers.
- Hunk fallback is enabled in the shipped graph configuration. Inspect CFG
  diagnostics: fallback nodes are not a successful full-repository AST analysis.
  All benchmark checkouts and all published patches were not materialized/applied
  in this review.
- Prompt rewriting returns `edit` without generating a replacement candidate
  patch. It is an advisory action, not evidence of a repaired patch.
- Row resume does not fingerprint tool versions, model files, repository content,
  or artifact existence. After changing those in place, use a fresh output
  directory or `--no-resume`. Dataset-level retraining is intentionally repeated.
- Scanner finding matching is an approximation: changing the surrounding snippet
  of an existing warning can affect whether it counts as new. Semgrep's shipped
  `--config auto` is mutable; pin rules and record scanner versions for final runs.

## Validation

`python -m pytest -q`: **24 passed**. `git diff --check` and Python compile checks
also passed. Environment: Python 3.14.6; pytest 9.1.1, bandit 1.9.4, numpy 2.5.3, scikit-learn 1.9.1, torch 2.14.0, torch-geometric 2.8.0.post1.

The offline regression suite covers scanner errors, result retries and isolation,
graph identity/deduplication, synthesis seed requirements, missing repositories,
CLI failure status, model reuse, all 14 feature classifiers, and a real two-epoch
PyTorch/PyG training run. Test-set evaluation is asserted to occur only once, after
training. No paid provider calls are involved.

Additional real-data smoke checks:

- Ran `prompt_filter` on 20 published LiveCodeBench rows with four workers.
- Materialized the first Codeforces instance using the LiveCodeBench setup script's
  template. Both published patches applied; two-worker Bandit accepted the benign
  patch and rejected the malicious patch (0 versus 3 new findings).
- Built and reloaded both graphs for that pair using deterministic subtasks and
  debug hash embeddings. This checks the graph/artifact pipeline, not the quality
  of production CodeBERT embeddings or LLM decomposition.

Production LLM calls, CodeBERT downloads/inference, large guard models, and a live
Semgrep scan were not exercised. These checks are not a full experimental rerun.
