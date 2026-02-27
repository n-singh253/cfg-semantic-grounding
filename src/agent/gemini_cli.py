"""gemini-cli command wrapper."""

from __future__ import annotations

from src.agent.base import BaseAgent
from src.agent.cli_wrapper import run_cli_agent
from src.agent.registry import register_agent
from src.common.types import Patch


class GeminiCLIWrapper(BaseAgent):
    name = "gemini_cli"

    def agent(self, repo_code, prompt, all_tests) -> Patch:
        return run_cli_agent(
            agent_name=self.name,
            config=self.config,
            repo_code=repo_code,
            prompt=prompt,
            all_tests=all_tests,
        )


register_agent("gemini_cli")(GeminiCLIWrapper)
