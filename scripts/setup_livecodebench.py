#!/usr/bin/env python3
"""Materialize LiveCodeBench code-generation rows as local git repos.

The harness evaluates patch agents, so each LiveCodeBench problem is represented
as a tiny repository containing solution.py plus public tests. Private tests are
stored outside those repositories and referenced by path/hash only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPOS_ROOT = Path.home() / "livecodebench_repos"
PRIVATE_ROOT = DATA_DIR / "livecodebench_private"


RUN_PUBLIC_TESTS = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import base64
import importlib.util
import inspect
import subprocess
import sys
import typing
from pathlib import Path


PUBLIC_TEST_CASES = json.loads(base64.b64decode("__PUBLIC_TEST_CASES_B64__").decode("utf-8"))
FUNC_NAME = "__FUNC_NAME__"


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _parse_jsonish(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _load_solution_module(root: Path):
    spec = importlib.util.spec_from_file_location("lcb_solution", root / "solution.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import solution.py")
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(
        {
            "List": typing.List,
            "Dict": typing.Dict,
            "Set": typing.Set,
            "Tuple": typing.Tuple,
            "Optional": typing.Optional,
        }
    )
    sys.modules["lcb_solution"] = module
    spec.loader.exec_module(module)
    return module


def _run_functional_case(root: Path, case: dict) -> tuple[bool, dict]:
    if not FUNC_NAME:
        return False, {"reason": "missing_func_name"}
    module = _load_solution_module(root)
    solution_cls = getattr(module, "Solution", None)
    if solution_cls is None:
        return False, {"reason": "missing_solution_class"}
    method = getattr(solution_cls(), FUNC_NAME, None)
    if method is None:
        return False, {"reason": "missing_method", "func_name": FUNC_NAME}

    parsed_input = _parse_jsonish(case.get("input", ""))
    expected = _parse_jsonish(case.get("output", ""))
    positional_count = len(
        [
            param
            for param in inspect.signature(method).parameters.values()
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
            and param.default is param.empty
        ]
    )
    if isinstance(parsed_input, list) and positional_count != 1:
        args = parsed_input
    else:
        args = [parsed_input]

    actual = method(*args)
    return actual == expected, {"expected": expected, "actual": actual}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    solution = root / "solution.py"
    if not solution.exists():
        print("solution.py is missing", file=sys.stderr)
        return 1

    failures = []
    skipped = 0
    for idx, case in enumerate(PUBLIC_TEST_CASES, start=1):
        testtype = str(case.get("testtype", "stdin")).lower()
        if testtype == "functional":
            try:
                ok, details = _run_functional_case(root, case)
            except Exception as exc:
                ok, details = False, {"reason": "exception", "message": repr(exc)}
            if not ok:
                failures.append({"case": idx, "testtype": testtype, **details})
            continue
        if testtype != "stdin":
            skipped += 1
            continue
        proc = subprocess.run(
            [sys.executable, str(solution)],
            input=str(case.get("input", "")),
            text=True,
            capture_output=True,
            cwd=root,
            timeout=20,
            check=False,
        )
        expected = str(case.get("output", ""))
        if proc.returncode != 0 or _norm(proc.stdout) != _norm(expected):
            failures.append(
                {
                    "case": idx,
                    "returncode": proc.returncode,
                    "expected": expected,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            )

    if failures:
        print(json.dumps({"failures": failures[:5], "failure_count": len(failures)}, indent=2))
        return 1
    print(json.dumps({"passed": len(PUBLIC_TEST_CASES) - skipped - len(failures), "skipped": skipped}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._")
    return value or "unknown"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_jsonish(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def _run_git(args: List[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _iter_dataset_rows(release: str, max_rows: int | None = None):
    from huggingface_hub import hf_hub_download

    release_files = {
        "release_v1": ["test.jsonl"],
        "release_v2": ["test.jsonl", "test2.jsonl"],
        "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
        "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
        "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
        "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
        "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
        "v1": ["test.jsonl"],
        "v2": ["test2.jsonl"],
        "v3": ["test3.jsonl"],
        "v4": ["test4.jsonl"],
        "v5": ["test5.jsonl"],
        "v6": ["test6.jsonl"],
    }
    for start in range(1, 7):
        for end in range(start + 1, 7):
            release_files[f"v{start}_v{end}"] = [
                "test.jsonl" if idx == 1 else f"test{idx}.jsonl"
                for idx in range(start, end + 1)
            ]

    if release not in release_files:
        supported = ", ".join(sorted(release_files))
        raise RuntimeError(f"Unsupported LiveCodeBench release {release!r}; expected one of: {supported}")

    yielded = 0
    for filename in release_files[release]:
        path = hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename=filename,
            repo_type="dataset",
        )
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                yielded += 1
                if max_rows is not None and yielded >= max(0, max_rows):
                    return


def _solution_template(starter_code: str) -> str:
    starter = starter_code.strip()
    if starter:
        return starter + "\n"
    return '''"""LiveCodeBench solution entrypoint.

Implement solve() so solution.py reads stdin and writes stdout.
"""

import sys


def solve() -> None:
    data = sys.stdin.read()
    raise NotImplementedError("Implement the solution for this problem.")


if __name__ == "__main__":
    solve()
'''


def _problem_statement(row: Dict[str, Any]) -> str:
    title = str(row.get("question_title") or row.get("title") or "LiveCodeBench problem")
    content = str(row.get("question_content") or row.get("question") or "")
    starter = str(row.get("starter_code") or "").strip()
    parts = [
        f"# {title}",
        "",
        content,
        "",
        "Edit solution.py to solve the problem. Keep the public tests passing.",
    ]
    if starter:
        parts.extend(["", "Starter code is already present in solution.py."])
    return "\n".join(parts).strip() + "\n"


def _metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _parse_jsonish(row.get("metadata"), {})
    return metadata if isinstance(metadata, dict) else {}


def _write_repo(repo_path: Path, row: Dict[str, Any], public_cases: List[Dict[str, Any]]) -> str:
    repo_path.mkdir(parents=True, exist_ok=True)
    (repo_path / "tests").mkdir(exist_ok=True)
    (repo_path / "README.md").write_text(_problem_statement(row), encoding="utf-8")
    (repo_path / "solution.py").write_text(_solution_template(str(row.get("starter_code") or "")), encoding="utf-8")

    public_json = json.dumps(public_cases, ensure_ascii=True)
    public_b64 = base64.b64encode(public_json.encode("utf-8")).decode("ascii")
    func_name = str(_metadata(row).get("func_name") or "")
    test_script = (
        RUN_PUBLIC_TESTS.replace("__PUBLIC_TEST_CASES_B64__", public_b64)
        .replace("__FUNC_NAME__", func_name)
    )
    test_path = repo_path / "tests" / "run_public_tests.py"
    test_path.write_text(test_script, encoding="utf-8")
    test_path.chmod(0o755)

    if not (repo_path / ".git").exists():
        _run_git(["init"], cwd=repo_path)
    _run_git(["add", "README.md", "solution.py", "tests/run_public_tests.py"], cwd=repo_path)
    try:
        _run_git(
            [
                "-c",
                "user.name=cfg-semantic-grounding",
                "-c",
                "user.email=cfg-semantic-grounding@example.invalid",
                "commit",
                "-m",
                "Initialize LiveCodeBench problem",
            ],
            cwd=repo_path,
        )
    except subprocess.CalledProcessError:
        pass
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_path, check=True, text=True, stdout=subprocess.PIPE)
    return result.stdout.strip()


def build_rows(dataset: Iterable[Dict[str, Any]], release: str, out_path: Path, limit: int | None) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    private_dir = PRIVATE_ROOT / release
    private_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for raw in dataset:
            row = dict(raw)
            platform = _slug(str(row.get("platform") or "unknown"))
            question_id = _slug(str(row.get("question_id") or row.get("id") or written))
            instance_id = _slug(f"lcb_{release}_{platform}_{question_id}")
            repo_path = REPOS_ROOT / release / instance_id

            public_cases = _parse_jsonish(row.get("public_test_cases"), [])
            if not isinstance(public_cases, list):
                public_cases = []
            private_tests_raw = row.get("private_test_cases") or ""
            private_tests_path = private_dir / f"{instance_id}.json"
            private_tests_path.write_text(
                json.dumps({"private_test_cases": private_tests_raw}, ensure_ascii=True),
                encoding="utf-8",
            )

            base_commit = _write_repo(repo_path, row, public_cases)
            record = {
                "instance_id": instance_id,
                "repo_id": f"livecodebench/{platform}/{question_id}",
                "repo_path": str(repo_path),
                "base_commit": base_commit,
                "problem_statement": _problem_statement(row),
                "test_command": ["python3", "tests/run_public_tests.py"],
                "variant": "code_generation_lite",
                "release": release,
                "platform": row.get("platform", ""),
                "question_id": row.get("question_id", ""),
                "question_title": row.get("question_title", ""),
                "contest_id": row.get("contest_id", ""),
                "contest_date": str(row.get("contest_date", "")),
                "difficulty": row.get("difficulty", ""),
                "public_test_count": len(public_cases),
                "func_name": _metadata(row).get("func_name", ""),
                "private_tests_path": str(private_tests_path),
                "private_tests_hash": _sha256_text(str(private_tests_raw)),
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            written += 1
            if limit is not None and written >= max(0, limit):
                break
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare LiveCodeBench for the patch-agent harness.")
    parser.add_argument("--release", default="release_latest")
    parser.add_argument("--output", default="data/livecodebench_code_generation_lite_release_latest.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    print(f"[1/2] Loading LiveCodeBench code_generation_lite ({args.release})")
    ds = _iter_dataset_rows(args.release, args.limit)
    print(f"[2/2] Writing harness rows to {out_path}")
    written = build_rows(ds, args.release, out_path, args.limit)
    print(f"       Wrote {written} rows")
    print(f"       Repos: {REPOS_ROOT / args.release}")
    print(f"       Private test references: {PRIVATE_ROOT / args.release}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
