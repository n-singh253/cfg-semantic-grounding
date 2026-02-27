"""Shared LLM client with cache/resume and provenance-friendly metadata."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.common.artifact_store import atomic_write_json, atomic_write_text
from src.common.hashing import sha256_json, sha256_text


@dataclass
class LLMCallResult:
    text: str
    provider: str
    model: str
    temperature: float
    seed: Optional[int]
    token_usage: Dict[str, Any]
    prompt_hash: str
    response_hash: str
    cache_key: str
    cache_hit: bool
    provider_fallback: bool
    tool_blocked: bool
    error: str
    call_count: int
    artifact_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMClient:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        *,
        instance_id: str,
        module_kind: str,
        module_name: str,
        module_config_hash: str,
        fidelity_mode: str,
        provider: str,
        model: str,
        prompt: str,
        artifact_dir: Path,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_retries: int = 2,
        backoff_sec: float = 1.0,
        allow_provider_fallback: bool = False,
        surrogate_fn: Optional[Callable[[str], str]] = None,
    ) -> LLMCallResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        prompt_hash = sha256_text(prompt)
        key_payload = {
            "instance_id": instance_id,
            "module_kind": module_kind,
            "module_name": module_name,
            "module_config_hash": module_config_hash,
            "fidelity_mode": fidelity_mode,
            "provider": provider,
            "model": model,
            "prompt_hash": prompt_hash,
        }
        key_hash = sha256_json(key_payload)
        cache_file = self.cache_root / module_kind / module_name / f"{instance_id}_{key_hash}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            cached["cache_hit"] = True
            return LLMCallResult(**cached)

        if fidelity_mode == "surrogate_debug":
            result = self._surrogate_result(
                provider=provider,
                model=model,
                prompt=prompt,
                prompt_hash=prompt_hash,
                key_hash=key_hash,
                artifact_dir=artifact_dir,
                surrogate_fn=surrogate_fn,
            )
            atomic_write_json(cache_file, result.to_dict())
            return result

        call_count = 0
        last_error = ""
        for attempt in range(max_retries + 1):
            call_count += 1
            try:
                text, usage = self._provider_call(
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    seed=seed,
                )
                response_hash = sha256_text(text)
                blocked = self._is_blocked_text(text)
                result = LLMCallResult(
                    text=text,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    token_usage=usage,
                    prompt_hash=prompt_hash,
                    response_hash=response_hash,
                    cache_key=key_hash,
                    cache_hit=False,
                    provider_fallback=False,
                    tool_blocked=blocked,
                    error="",
                    call_count=call_count,
                    artifact_path=str(artifact_dir),
                )
                self._write_artifacts(artifact_dir, prompt, text, result.to_dict())
                atomic_write_json(cache_file, result.to_dict())
                return result
            except Exception as exc:  # pragma: no cover - provider/env dependent.
                last_error = str(exc)
                if attempt < max_retries:
                    time.sleep(backoff_sec * (2**attempt))
                    continue

        if allow_provider_fallback:
            result = self._surrogate_result(
                provider=provider,
                model=model,
                prompt=prompt,
                prompt_hash=prompt_hash,
                key_hash=key_hash,
                artifact_dir=artifact_dir,
                surrogate_fn=surrogate_fn,
                error=last_error,
                provider_fallback=True,
            )
            atomic_write_json(cache_file, result.to_dict())
            return result

        raise RuntimeError(
            f"LLM call failed for {module_kind}:{module_name} after retries. "
            f"provider={provider}, model={model}, error={last_error}"
        )

    @staticmethod
    def _is_blocked_text(text: str) -> bool:
        lowered = (text or "").lower()
        blocked_tokens = ["safety", "policy", "cannot comply", "can't comply", "refuse"]
        return any(token in lowered for token in blocked_tokens)

    def _surrogate_result(
        self,
        *,
        provider: str,
        model: str,
        prompt: str,
        prompt_hash: str,
        key_hash: str,
        artifact_dir: Path,
        surrogate_fn: Optional[Callable[[str], str]],
        error: str = "",
        provider_fallback: bool = True,
    ) -> LLMCallResult:
        fn = surrogate_fn or (lambda p: p)
        text = fn(prompt)
        result = LLMCallResult(
            text=text,
            provider=provider,
            model=model,
            temperature=0.0,
            seed=None,
            token_usage={},
            prompt_hash=prompt_hash,
            response_hash=sha256_text(text),
            cache_key=key_hash,
            cache_hit=False,
            provider_fallback=provider_fallback,
            tool_blocked=False,
            error=error,
            call_count=1,
            artifact_path=str(artifact_dir),
        )
        self._write_artifacts(artifact_dir, prompt, text, result.to_dict())
        return result

    @staticmethod
    def _write_artifacts(artifact_dir: Path, prompt: str, response: str, metadata: Dict[str, Any]) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(artifact_dir / "prompt.txt", prompt)
        atomic_write_text(artifact_dir / "response.txt", response)
        atomic_write_json(artifact_dir / "metadata.json", metadata)

    def _provider_call(
        self,
        *,
        provider: str,
        model: str,
        prompt: str,
        temperature: float,
        seed: Optional[int],
    ) -> tuple[str, Dict[str, Any]]:
        normalized = provider.strip().lower()
        if normalized == "openai":
            return self._call_openai(model=model, prompt=prompt, temperature=temperature, seed=seed)
        if normalized == "gemini":
            return self._call_gemini(model=model, prompt=prompt, temperature=temperature)
        if normalized in {"anthropic", "claude"}:
            return self._call_anthropic(model=model, prompt=prompt, temperature=temperature)
        raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _call_openai(
        *,
        model: str,
        prompt: str,
        temperature: float,
        seed: Optional[int],
    ) -> tuple[str, Dict[str, Any]]:
        from openai import OpenAI  # pragma: no cover - optional dependency.

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=api_key)
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        resp = client.chat.completions.create(**payload)
        text = (resp.choices[0].message.content or "").strip()
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        return text, usage

    @staticmethod
    def _call_gemini(*, model: str, prompt: str, temperature: float) -> tuple[str, Dict[str, Any]]:
        import google.generativeai as genai  # pragma: no cover - optional dependency.

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(model_name=model)
        resp = gen_model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        text = getattr(resp, "text", "") or ""
        usage = {}
        usage_meta = getattr(resp, "usage_metadata", None)
        if usage_meta is not None:
            usage = {
                "prompt_token_count": getattr(usage_meta, "prompt_token_count", None),
                "candidates_token_count": getattr(usage_meta, "candidates_token_count", None),
                "total_token_count": getattr(usage_meta, "total_token_count", None),
            }
        return text.strip(), usage

    @staticmethod
    def _call_anthropic(*, model: str, prompt: str, temperature: float) -> tuple[str, Dict[str, Any]]:
        from anthropic import Anthropic  # pragma: no cover - optional dependency.

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts = []
        for block in getattr(resp, "content", []):
            if getattr(block, "type", "") == "text":
                text_parts.append(getattr(block, "text", ""))
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", None),
                "output_tokens": getattr(resp.usage, "output_tokens", None),
            }
        return "\n".join(text_parts).strip(), usage
