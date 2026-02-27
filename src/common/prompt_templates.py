"""Shared prompt templates used across agent wrappers."""

from __future__ import annotations

AGENT_PATCH_PROMPT_TEMPLATE = (
    "You are solving a SWE-Bench style repository issue.\n"
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
    "3) Return ONLY a valid unified diff patch.\n"
)

