# Paper model rates, accessed 2026-06-25

Rates are USD per 1M tokens and are intended for `scripts/export_cost_table.py --rates-json`.

| Pricing key | Input | Output | Source / note |
| --- | ---: | ---: | --- |
| `gemini_vertex/gemini-3-flash-preview` | 0.50 | 3.00 | Google Cloud Vertex AI / Gemini Enterprise Agent Platform, Gemini 3 Flash Preview, Standard text/image/video input and text output. |
| `anthropic_vertex/claude-sonnet-4-6` | 3.30 | 16.50 | Google Cloud Vertex AI partner-model pricing, `us-east5`, Claude Sonnet 4.6. |
| `anthropic_vertex/claude-sonnet-4@20250514` | 3.00 | 15.00 | Google Cloud Vertex AI partner-model pricing, uniform all-region price for Claude Sonnet 4 (Deprecated). |
| `anthropic_vertex/claude-3-7-sonnet@20250219` | 3.00 | 15.00 | Google Cloud Vertex AI partner-model pricing, uniform all-region price for Claude 3.7 Sonnet (Deprecated). |

Assumptions:

- These estimates cover token-only model charges from tracked LLM usage, not local machine/GPU time, storage, repo cloning, or one-time GNN training.
- Gemini pricing uses Standard rather than Priority or Flex/Batch because the baseline scripts issue normal Vertex calls rather than batch/flex/priority requests.
- Claude Sonnet 4.6 uses `us-east5` because our Claude 4.6 Vertex runs use that location.
- The exporter only prices rows with a single provider/model key and tracked token usage.
