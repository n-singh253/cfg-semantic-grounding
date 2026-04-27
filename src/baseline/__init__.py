"""Baseline/defense plugin package."""

from src.baseline import (  # noqa: F401
    agentic_guard,
    llm_judge,
    prompt_filter,
    prompt_rewrite,
    static_bandit,
    static_semgrep,
)
from src.baseline.structural_misalignment import plugin as structural_misalignment  # noqa: F401
from src.baseline.structural_misalignment import feature_plugin as structural_misalignment_features  # noqa: F401

try:
    from src.baseline import llama_guard  # noqa: F401
except Exception:
    pass  # llama_guard requires torch and transformers

try:
    from src.baseline import llama_prompt_guard  # noqa: F401
except Exception:
    pass  # llama_prompt_guard requires torch and transformers

try:
    from src.baseline import sequence_classifiers  # noqa: F401
except Exception:
    pass  # sequence_classifiers requires torch, transformers, and sklearn
