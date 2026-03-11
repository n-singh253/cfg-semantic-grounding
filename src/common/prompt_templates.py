"""Shared prompt templates used across agent wrappers."""

from __future__ import annotations

AGENT_PATCH_PROMPT_TEMPLATE = (
    "You are solving a SWE-Bench style repository issue.\n"
    "You do NOT have tool access (no file editing, shell, or search tools).\n"
    "Do not mention tool limitations. Produce the patch directly.\n"
    "Instance: {instance_id}\n"
    "Repo: {repo_id}\n"
    "Base commit: {base_commit}\n"
    "Repository path: {repo_path}\n\n"
    "Issue / Prompt:\n"
    "{prompt}\n\n"
    "Available tests:\n"
    "{tests}\n\n"
    "Requirements:\n"
    "1) Produce a minimal patch that fixes the issue and passes tests.\n"
    "2) Do not change unrelated behavior.\n"
    "3) Return ONLY one valid unified diff patch.\n"
    "4) Do not use markdown fences and do not include explanation text.\n"
)
