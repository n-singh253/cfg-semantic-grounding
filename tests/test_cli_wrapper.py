from pathlib import Path

from src.agent.cli_wrapper import _git_diff, _openhands_fatal_generation_error
from src.common.subprocess import run_command


def _init_git_repo(path: Path) -> None:
    run_command(["git", "init"], cwd=path)
    run_command(["git", "config", "user.email", "test@example.com"], cwd=path)
    run_command(["git", "config", "user.name", "Test User"], cwd=path)
    run_command(["git", "add", "."], cwd=path)
    run_command(["git", "commit", "-m", "init"], cwd=path)


def test_git_diff_includes_untracked_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    _init_git_repo(repo)

    (repo / "agent_code").mkdir()
    (repo / "agent_code" / "__init__.py").write_text("value = 2\n", encoding="utf-8")

    diff = _git_diff(repo)

    assert "diff --git a/agent_code/__init__.py b/agent_code/__init__.py" in diff
    assert "new file mode" in diff
    assert "+value = 2" in diff


def test_openhands_fatal_generation_error_detects_provider_billing_failure():
    stdout = (
        '--JSON Event--\n{"kind": "ConversationErrorEvent", "code": "APIError", '
        '"detail": "litellm.APIError: OpenrouterException - '
        '{\\"error\\":{\\"message\\":\\"Insufficient credits\\",\\"code\\":402}}"}'
    )

    assert _openhands_fatal_generation_error(stdout) == "provider_insufficient_credits"


def test_openhands_fatal_generation_error_detects_direct_fallback_billing_failure():
    error = (
        "RuntimeError: LLM call failed for agent_direct_fallback:openhands after retries. "
        "provider=openrouter, model=qwen/qwen3-coder-30b-a3b-instruct, "
        "error=Error code: 402 - {'error': {'message': 'Insufficient credits', 'code': 402}}"
    )

    assert _openhands_fatal_generation_error(error) == "provider_insufficient_credits"
