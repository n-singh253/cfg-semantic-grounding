"""Shared LLM client with cache/resume and provenance-friendly metadata."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shlex
import threading
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.common.artifact_store import atomic_write_json, atomic_write_text
from src.common.hashing import sha256_json, sha256_text


_PROVIDER_SEMAPHORES: Dict[str, threading.BoundedSemaphore] = {}
_PROVIDER_SEMAPHORES_LOCK = threading.Lock()


def _provider_call_process_worker(queue: Any, kwargs: Dict[str, Any]) -> None:
    try:
        client = LLMClient(Path(os.environ.get("CFG_LLM_SUBPROCESS_CACHE_ROOT", "/tmp/cfg-llm-provider-call")))
        queue.put({"ok": True, "value": client._provider_call(**kwargs)})
    except BaseException as exc:  # pragma: no cover - subprocess/provider dependent.
        queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


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
                text, usage = self._provider_call_with_hard_timeout(
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    seed=seed,
                )
                if not (text or "").strip():
                    usage_summary = json.dumps(usage, sort_keys=True)[:1500] if usage else "{}"
                    raise RuntimeError(f"LLM provider returned empty response; usage={usage_summary}")
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

    def _provider_call_with_hard_timeout(
        self,
        *,
        provider: str,
        model: str,
        prompt: str,
        temperature: float,
        seed: Optional[int],
    ) -> tuple[str, Dict[str, Any]]:
        timeout_sec = self._hard_timeout_seconds(provider)
        call_kwargs = {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "seed": seed,
        }
        semaphore = self._provider_semaphore(provider)
        if semaphore is not None:
            with semaphore:
                return self._provider_call_with_hard_timeout_inner(timeout_sec=timeout_sec, call_kwargs=call_kwargs)
        return self._provider_call_with_hard_timeout_inner(timeout_sec=timeout_sec, call_kwargs=call_kwargs)

    def _provider_call_with_hard_timeout_inner(
        self,
        *,
        timeout_sec: float,
        call_kwargs: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        if timeout_sec <= 0:
            return self._provider_call(
                **call_kwargs,
            )

        methods = mp.get_all_start_methods()
        context = mp.get_context("fork" if "fork" in methods else methods[0])
        queue: Any = context.Queue(maxsize=1)
        process = context.Process(target=_provider_call_process_worker, args=(queue, call_kwargs))
        process.start()
        try:
            process.join(timeout_sec)
            if process.is_alive():
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
                raise TimeoutError(f"LLM provider call exceeded hard timeout of {timeout_sec:g}s")
            if queue.empty():
                raise RuntimeError(f"LLM provider subprocess exited without a result; exitcode={process.exitcode}")
            result = queue.get()
            if not result.get("ok"):
                error = str(result.get("error", "unknown subprocess error"))
                tb = str(result.get("traceback", ""))[:2000]
                raise RuntimeError(f"LLM provider subprocess failed: {error}\n{tb}")
            value = result["value"]
            return value[0], value[1]
        finally:
            queue.cancel_join_thread()
            queue.close()

    @staticmethod
    def _hard_timeout_seconds(provider: str) -> float:
        normalized = provider.strip().upper().replace("-", "_")
        provider_key = f"CFG_{normalized}_HARD_TIMEOUT_SEC"
        raw_value = os.environ.get(provider_key) or os.environ.get("CFG_LLM_HARD_TIMEOUT_SEC")
        if not raw_value:
            return 0.0
        try:
            return max(0.0, float(raw_value))
        except ValueError:
            return 0.0

    @staticmethod
    def _provider_concurrency_limit(provider: str) -> int:
        normalized = provider.strip().upper().replace("-", "_")
        provider_key = f"CFG_{normalized}_MAX_CONCURRENT_CALLS"
        raw_value = os.environ.get(provider_key) or os.environ.get("CFG_LLM_MAX_CONCURRENT_CALLS")
        if not raw_value:
            return 0
        try:
            return max(0, int(raw_value))
        except ValueError:
            return 0

    @staticmethod
    def _provider_semaphore(provider: str) -> threading.BoundedSemaphore | None:
        limit = LLMClient._provider_concurrency_limit(provider)
        if limit <= 0:
            return None
        normalized = provider.strip().lower().replace("-", "_")
        key = f"{normalized}:{limit}"
        with _PROVIDER_SEMAPHORES_LOCK:
            semaphore = _PROVIDER_SEMAPHORES.get(key)
            if semaphore is None:
                semaphore = threading.BoundedSemaphore(limit)
                _PROVIDER_SEMAPHORES[key] = semaphore
            return semaphore

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
        if normalized == "vllm":
            return self._call_vllm(model=model, prompt=prompt, temperature=temperature, seed=seed)
        if normalized == "gemini":
            return self._call_gemini(model=model, prompt=prompt, temperature=temperature)
        if normalized in {"gemini_vertex", "vertex", "vertex_ai"}:
            return self._call_gemini_vertex(model=model, prompt=prompt, temperature=temperature)
        if normalized == "gemini_cli":
            return self._call_gemini_cli(model=model, prompt=prompt, temperature=temperature)
        if normalized in {"anthropic_vertex", "claude_vertex", "vertex_anthropic"}:
            return self._call_anthropic_vertex(model=model, prompt=prompt, temperature=temperature)
        if normalized in {"anthropic", "claude"}:
            return self._call_anthropic(model=model, prompt=prompt, temperature=temperature)
        raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _call_vllm(
        *,
        model: str,
        prompt: str,
        temperature: float,
        seed: Optional[int],
    ) -> tuple[str, Dict[str, Any]]:
        from openai import OpenAI  # pragma: no cover - optional dependency.

        base_url = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
        api_key = os.environ.get("VLLM_API_KEY", "local-dev-key")
        client = OpenAI(base_url=base_url, api_key=api_key)
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
            
        base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )
        client_kwargs: Dict[str, Any] = {"api_key": api_key}

        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
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
    def _extract_gemini_text(resp: Any) -> str:
        text = getattr(resp, "text", "") or ""
        if text:
            return str(text).strip()

        parts_text: list[str] = []
        for candidate in getattr(resp, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", "") or ""
                if part_text:
                    parts_text.append(str(part_text))
        return "\n".join(parts_text).strip()

    @staticmethod
    def _gemini_empty_response_diagnostics(resp: Any) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        prompt_feedback = getattr(resp, "prompt_feedback", None)
        if prompt_feedback is not None:
            diagnostics["prompt_feedback"] = str(prompt_feedback)
        candidate_summaries = []
        for candidate in getattr(resp, "candidates", []) or []:
            candidate_summaries.append(
                {
                    "finish_reason": str(getattr(candidate, "finish_reason", "")),
                    "safety_ratings": str(getattr(candidate, "safety_ratings", "")),
                }
            )
        if candidate_summaries:
            diagnostics["candidates"] = candidate_summaries
        return diagnostics

    @staticmethod
    def _gemini_safety_settings(types_module: Any) -> list[Any]:
        threshold_name = (
            os.environ.get("CFG_GEMINI_VERTEX_SAFETY_THRESHOLD")
            or os.environ.get("CFG_GEMINI_SAFETY_THRESHOLD")
            or os.environ.get("CFG_LLM_SAFETY_THRESHOLD")
            or ""
        ).strip()
        if not threshold_name:
            return []

        threshold = getattr(types_module.HarmBlockThreshold, threshold_name, None)
        if threshold is None:
            threshold = getattr(types_module.HarmBlockThreshold, threshold_name.upper(), None)
        if threshold is None:
            raise RuntimeError(f"Unsupported Gemini safety threshold: {threshold_name}")

        categories = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
            "HARM_CATEGORY_JAILBREAK",
        ]
        settings = []
        for category_name in categories:
            category = getattr(types_module.HarmCategory, category_name, None)
            if category is not None:
                settings.append(types_module.SafetySetting(category=category, threshold=threshold))
        return settings

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
        usage = {}
        usage_meta = getattr(resp, "usage_metadata", None)
        if usage_meta is not None:
            usage = {
                "prompt_token_count": getattr(usage_meta, "prompt_token_count", None),
                "candidates_token_count": getattr(usage_meta, "candidates_token_count", None),
                "total_token_count": getattr(usage_meta, "total_token_count", None),
            }
        text = LLMClient._extract_gemini_text(resp)
        if not text:
            diagnostics = LLMClient._gemini_empty_response_diagnostics(resp)
            if diagnostics:
                usage["empty_response_diagnostics"] = diagnostics
        return text, usage

    @staticmethod
    def _call_gemini_vertex(*, model: str, prompt: str, temperature: float) -> tuple[str, Dict[str, Any]]:
        from google import genai  # pragma: no cover - optional dependency.
        from google.genai import types  # pragma: no cover - optional dependency.

        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEXAI_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("VERTEXAI_LOCATION") or "global"
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set for Gemini Vertex calls")
        if credentials_path and not Path(credentials_path).expanduser().exists():
            raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS points to a missing file")

        timeout_ms = int(
            os.environ.get("CFG_GEMINI_VERTEX_TIMEOUT_MS")
            or os.environ.get("CFG_LLM_TIMEOUT_MS")
            or "90000"
        )
        max_output_tokens_env = os.environ.get("CFG_GEMINI_VERTEX_MAX_OUTPUT_TOKENS") or os.environ.get(
            "CFG_LLM_MAX_OUTPUT_TOKENS"
        )
        max_output_tokens = int(max_output_tokens_env) if max_output_tokens_env else None
        response_mime_type = os.environ.get("CFG_GEMINI_VERTEX_RESPONSE_MIME_TYPE") or os.environ.get(
            "CFG_LLM_RESPONSE_MIME_TYPE"
        )
        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(api_version="v1", timeout=timeout_ms),
        )
        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "http_options": types.HttpOptions(api_version="v1", timeout=timeout_ms),
        }
        safety_settings = LLMClient._gemini_safety_settings(types)
        if max_output_tokens is not None:
            generation_config["max_output_tokens"] = max_output_tokens
        if response_mime_type:
            generation_config["response_mime_type"] = response_mime_type
        if safety_settings:
            generation_config["safety_settings"] = safety_settings
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**generation_config),
        )
        usage = {}
        usage_meta = getattr(resp, "usage_metadata", None)
        if usage_meta is not None:
            usage = {
                "prompt_token_count": getattr(usage_meta, "prompt_token_count", None),
                "candidates_token_count": getattr(usage_meta, "candidates_token_count", None),
                "total_token_count": getattr(usage_meta, "total_token_count", None),
            }
        text = LLMClient._extract_gemini_text(resp)
        if not text:
            diagnostics = LLMClient._gemini_empty_response_diagnostics(resp)
            if diagnostics:
                usage["empty_response_diagnostics"] = diagnostics
        return text, usage

    @staticmethod
    def _call_gemini_cli(*, model: str, prompt: str, temperature: float) -> tuple[str, Dict[str, Any]]:
        """Call Gemini via the locally-authenticated gemini CLI (no API key)."""
        import shutil
        import subprocess
        import tempfile

        gemini_bin = shutil.which("gemini")
        if not gemini_bin:
            raise RuntimeError("gemini CLI is not installed or not on PATH")

        # Write prompt to temp file then pass via shell expansion to avoid
        # argument length limits on large prompts.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write(prompt)
            tmp_path = tmp.name

        try:
            # Use shell=True with cat to handle prompts exceeding ARG_MAX.
            shell_cmd = f'{shlex.quote(gemini_bin)} -p "$(cat {shlex.quote(tmp_path)})" -m {shlex.quote(model)}'
            result = subprocess.run(
                shell_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        finally:
            os.unlink(tmp_path)

        if result.returncode != 0:
            raise RuntimeError(
                f"gemini CLI exited {result.returncode}: {(result.stderr or '')[:500]}"
            )
        text = (result.stdout or "").strip()
        if not text:
            raise RuntimeError("gemini CLI returned empty output")
        return text, {"provider": "gemini_cli", "model": model}

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

    @staticmethod
    def _call_anthropic_vertex(*, model: str, prompt: str, temperature: float) -> tuple[str, Dict[str, Any]]:
        from anthropic import AnthropicVertex  # pragma: no cover - optional dependency.

        project_id = (
            os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("VERTEXAI_PROJECT")
        )
        region = os.environ.get("ANTHROPIC_VERTEX_REGION") or os.environ.get("VERTEXAI_LOCATION")
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not project_id:
            raise RuntimeError("ANTHROPIC_VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT is not set")
        if not region:
            raise RuntimeError("ANTHROPIC_VERTEX_REGION or VERTEXAI_LOCATION is not set")
        if credentials_path and not Path(credentials_path).expanduser().exists():
            raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS points to a missing file")

        timeout_sec = float(
            os.environ.get("CFG_ANTHROPIC_VERTEX_TIMEOUT_SEC")
            or os.environ.get("CFG_LLM_TIMEOUT_SEC")
            or "90"
        )
        client = AnthropicVertex(region=region, project_id=project_id, timeout=timeout_sec)
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
