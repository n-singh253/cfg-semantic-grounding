"""Baseline/defense plugin package."""

from src.baseline import (  # noqa: F401
    agentic_guard,
    llm_judge,
    prompt_filter,
    prompt_rewrite,
    static_bandit,
    static_semgrep,
)
from src.baseline.structural_misalignment import build_graph_plugin as structural_misalignment_build_graph  # noqa: F401
from src.baseline.structural_misalignment import eval_plugin as structural_misalignment_eval  # noqa: F401

try:
    from src.baseline import llama_guard  # noqa: F401
except Exception:
    pass  # llama_guard requires torch and transformers

try:
    from src.baseline import llama_prompt_guard  # noqa: F401
except Exception:
    pass  # llama_prompt_guard requires torch and transformers
