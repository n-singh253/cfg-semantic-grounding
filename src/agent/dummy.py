"""Dummy runnable agent that returns a patch fixing the toy failing test."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.base import BaseAgent
from src.agent.registry import register_agent
from src.common.types import Patch


class DummyAgent(BaseAgent):
    name = "dummy"

    def agent(self, repo_code, prompt: str, all_tests) -> Patch:
        _ = (all_tests,)
        style = str(self.config.get("patch_style", "add_fix"))
        if "subtract" in prompt.lower():
            style = "add_fix_alt"
        if style == "add_fix_alt":
            body = (
                "diff --git a/toy_module.py b/toy_module.py\n"
                "--- a/toy_module.py\n"
                "+++ b/toy_module.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def add(a, b):\n"
                "-    return a - b\n"
                "+    return (a + b)\n"
            )
        else:
            body = (
                "diff --git a/toy_module.py b/toy_module.py\n"
                "--- a/toy_module.py\n"
                "+++ b/toy_module.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def add(a, b):\n"
                "-    return a - b\n"
                "+    return a + b\n"
            )
        trace_path = Path(str(repo_code.get("path", "."))) / ".dummy_agent_trace.json"
        trace_path.write_text(
            json.dumps(
                {"prompt_preview": prompt[:240], "style": style},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return Patch(unified_diff=body, metadata={"agent": self.name, "style": style, "trace": str(trace_path)})


register_agent("dummy")(DummyAgent)
