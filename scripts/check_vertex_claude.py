#!/usr/bin/env python3
"""Diagnose Claude-on-Vertex auth/connectivity for the attack harness."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _print_env() -> None:
    print("[env]")
    for key in (
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "VERTEXAI_PROJECT",
        "VERTEXAI_LOCATION",
        "MSWEA_MODEL_NAME",
    ):
        value = os.environ.get(key, "")
        if key == "GOOGLE_APPLICATION_CREDENTIALS" and value:
            path = Path(value).expanduser()
            print(f"{key}={path} exists={path.exists()}")
        else:
            print(f"{key}={value or '<unset>'}")


def _project_and_location() -> tuple[str, str]:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEXAI_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("VERTEXAI_LOCATION") or "global"
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT/VERTEXAI_PROJECT is not set")
    return project, location


def _check_anthropic_vertex(model: str, prompt: str, timeout: int) -> bool:
    print("\n[anthropic-vertex]")
    try:
        from anthropic import AnthropicVertex

        project, location = _project_and_location()
        client = AnthropicVertex(project_id=project, region=location, timeout=timeout)
        message = client.messages.create(
            model=model,
            max_tokens=16,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            getattr(block, "text", "")
            for block in getattr(message, "content", [])
            if getattr(block, "type", "") == "text"
        ).strip()
        print(f"ok text={text[:120]!r}")
        return True
    except Exception as exc:
        print(f"failed {type(exc).__name__}: {exc}")
        return False


def _check_litellm(model: str, prompt: str, timeout: int) -> bool:
    print("\n[litellm]")
    try:
        import litellm

        response = litellm.completion(
            model=f"vertex_ai/{model}",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=16,
            timeout=timeout,
            num_retries=0,
        )
        text = response["choices"][0]["message"]["content"].strip()
        print(f"ok text={text[:120]!r}")
        return True
    except Exception as exc:
        print(f"failed {type(exc).__name__}: {exc}")
        return False


def _check_mini(model: str, timeout: int) -> bool:
    print("\n[mini]")
    mini_path = shutil.which("mini")
    if mini_path is None:
        fallback = Path(sys.executable).parent / "mini"
        if fallback.exists():
            mini_path = str(fallback)
    if mini_path is None:
        print("failed: mini not found on PATH")
        return False

    with tempfile.TemporaryDirectory(prefix="vertex-claude-mini-check-") as td:
        work = Path(td)
        (work / "answer.txt").write_text("old\n", encoding="utf-8")
        subprocess.run(["git", "init"], cwd=work, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(
            ["git", "-c", "user.name=check", "-c", "user.email=check@example.invalid", "add", "answer.txt"],
            cwd=work,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["git", "-c", "user.name=check", "-c", "user.email=check@example.invalid", "commit", "-m", "init"],
            cwd=work,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        cmd = [
            str(mini_path),
            "-t",
            "Change answer.txt so it contains exactly: ok",
            "-m",
            f"vertex_ai/{model}",
            "--agent-class",
            "default",
            "-l",
            "0.05",
            "-c",
            "mini.yaml",
            "-c",
            "model.set_cache_control=null",
            "-c",
            "agent.step_limit=4",
            "-c",
            f"model.model_kwargs.timeout={timeout}",
            "-c",
            "model.model_kwargs.num_retries=0",
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=work,
                text=True,
                capture_output=True,
                timeout=max(timeout * 3, 90),
                check=False,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            result = subprocess.CompletedProcess(
                cmd,
                124,
                stdout=(exc.stdout or ""),
                stderr=(exc.stderr or ""),
            )
            timed_out = True
        diff = subprocess.run(["git", "diff", "--", "answer.txt"], cwd=work, text=True, capture_output=True)
        print(f"returncode={result.returncode}")
        if timed_out:
            print("timed_out=true")
        if result.stderr.strip():
            print(f"stderr_tail={result.stderr.strip()[-1000:]}")
        if result.stdout.strip():
            print(f"stdout_tail={result.stdout.strip()[-1000:]}")
        if diff.stdout.strip():
            print("git_diff_present=true")
            return result.returncode == 0
        print("git_diff_present=false")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Claude Vertex connectivity.")
    parser.add_argument("--model", default="claude-3-7-sonnet@20250219")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--skip-mini", action="store_true")
    args = parser.parse_args()

    prompt = "Reply with exactly: ok"
    _print_env()
    anthropic_ok = _check_anthropic_vertex(args.model, prompt, args.timeout)
    litellm_ok = _check_litellm(args.model, prompt, args.timeout)
    mini_ok = True if args.skip_mini else _check_mini(args.model, args.timeout)

    print("\n[summary]")
    print(f"anthropic_vertex={anthropic_ok}")
    print(f"litellm={litellm_ok}")
    print(f"mini={mini_ok}")
    return 0 if anthropic_ok and litellm_ok and mini_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
