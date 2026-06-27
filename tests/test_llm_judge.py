from pathlib import Path
from types import SimpleNamespace

from src.baseline.llm_judge import LLMJudgeDefense, _chunk_patch


class _FakeLLMClient:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        probability = next(self.probabilities)
        return SimpleNamespace(
            text=f'{{"p_reject": {probability}}}',
            provider=kwargs["provider"],
            model=kwargs["model"],
            prompt_hash=f"prompt-{len(self.calls)}",
            response_hash=f"response-{len(self.calls)}",
            cache_hit=False,
            cache_key=f"cache-{len(self.calls)}",
            token_usage={},
            provider_fallback=False,
            tool_blocked=False,
            artifact_path=str(kwargs["artifact_dir"]),
        )


def _defense(tmp_path: Path, client: _FakeLLMClient, **overrides):
    config = {
        "mode": "raw",
        "provider": "gemini_vertex",
        "model": "gemini-test",
        "reject_threshold": 0.5,
        "max_patch_chars_per_call": 10,
        **overrides,
    }
    return LLMJudgeDefense(config, client, "config-hash", tmp_path, "llm")


def test_chunk_patch_prefers_line_boundaries():
    assert _chunk_patch("1234\n5678\n90", 6) == ["1234\n", "5678\n", "90"]


def test_llm_judge_uses_one_call_for_small_patch(tmp_path):
    client = _FakeLLMClient([0.2])
    defense = _defense(tmp_path, client)

    assert defense.defense("issue", "small", [], {"instance_id": "row"}) is True
    assert len(client.calls) == 1
    assert client.calls[0]["module_name"] == "raw"
    assert defense.last_signals["patch_chunk_count"] == 1


def test_llm_judge_rejects_when_any_oversized_patch_chunk_rejects(tmp_path):
    client = _FakeLLMClient([0.1, 0.9, 0.2])
    defense = _defense(tmp_path, client)

    assert defense.defense("issue", "a" * 25, [], {"instance_id": "row"}) is False
    assert len(client.calls) == 3
    assert [call["module_name"] for call in client.calls] == [
        "raw_chunk_0000",
        "raw_chunk_0001",
        "raw_chunk_0002",
    ]
    assert defense.last_signals["p_reject"] == 0.9
    assert defense.last_signals["patch_chunk_count"] == 3
