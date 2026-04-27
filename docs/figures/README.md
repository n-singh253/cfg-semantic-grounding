# Figure Sources (Paper Export)

These are publication-friendly Graphviz sources for the pipeline figures:

- `pipeline_core.dot` (main pipeline with data contracts)
- `pipeline_options.dot` (current config option space)
- `pipeline_paper_style.dot` (actor-driven paper-style summary figure)
- `pipeline_stylized_overview.dot` (stepwise stylized overview inspired by SWExploit figure style)
- `pipeline_figure5.dot` (two-phase decomposition with defense decision branching)
- `pipeline_semantic_misalignment.dot` (semantic misalignment defense internals with module I/O contracts)

If Graphviz is installed, export with:

```bash
dot -Tsvg docs/figures/pipeline_core.dot -o docs/figures/pipeline_core.svg
dot -Tpdf docs/figures/pipeline_core.dot -o docs/figures/pipeline_core.pdf
dot -Tsvg docs/figures/pipeline_options.dot -o docs/figures/pipeline_options.svg
dot -Tpdf docs/figures/pipeline_options.dot -o docs/figures/pipeline_options.pdf
dot -Tsvg docs/figures/pipeline_paper_style.dot -o docs/figures/pipeline_paper_style.svg
dot -Tpdf docs/figures/pipeline_paper_style.dot -o docs/figures/pipeline_paper_style.pdf
dot -Tsvg docs/figures/pipeline_stylized_overview.dot -o docs/figures/pipeline_stylized_overview.svg
dot -Tpdf docs/figures/pipeline_stylized_overview.dot -o docs/figures/pipeline_stylized_overview.pdf
dot -Tsvg docs/figures/pipeline_figure5.dot -o docs/figures/pipeline_figure5.svg
dot -Tpdf docs/figures/pipeline_figure5.dot -o docs/figures/pipeline_figure5.pdf
dot -Tsvg docs/figures/pipeline_semantic_misalignment.dot -o docs/figures/pipeline_semantic_misalignment.svg
dot -Tpdf docs/figures/pipeline_semantic_misalignment.dot -o docs/figures/pipeline_semantic_misalignment.pdf
```

Use `*.pdf` directly in LaTeX and `*.svg` for docs/web.
