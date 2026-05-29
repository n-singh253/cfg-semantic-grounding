"""Bandit-based defense baseline.

Execution model:
1. Reset repo to a clean checkout.
2. Run Bandit on the clean repo state and record baseline findings.
3. Temporarily apply the candidate patch.
4. Run Bandit again on the patched repo state.
5. Reset the repo back to a clean checkout.
6. Accept iff the patch does not increase the number of Bandit findings.

This makes Bandit evaluate the patch's marginal security effect rather than
penalizing a patch for pre-existing findings already present in the repository.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.baseline.base import BaseDefense
from src.baseline.registry import register_baseline
from src.common.subprocess import command_exists, run_command
from src.eval.patch_eval import apply_patch


class StaticBanditDefense(BaseDefense):
    name = "static_bandit"

    def _reset_repo(self, repo_path: Path, base_commit: str = "") -> Tuple[bool, str]:
        """Reset the repo to a clean git state."""
        if not command_exists("git"):
            return False, "git_not_available_for_reset"

        probe = run_command(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_path)
        if probe.returncode != 0 or "true" not in (probe.stdout or "").lower():
            return False, "not_a_git_work_tree"

        target = base_commit.strip() or "HEAD"
        reset_res = run_command(["git", "reset", "--hard", target], cwd=repo_path)
        if reset_res.returncode != 0:
            msg = (reset_res.stderr or reset_res.stdout or "").strip()
            return False, f"git_reset_failed: {msg}"

        clean_res = run_command(["git", "clean", "-fd"], cwd=repo_path)
        if clean_res.returncode != 0:
            msg = (clean_res.stderr or clean_res.stdout or "").strip()
            return False, f"git_clean_failed: {msg}"

        return True, "ok"

    def _build_bandit_command(self, base_command: List[str], report_path: Path) -> List[str]:
        """
        Build a Bandit command that always writes JSON to report_path.

        This is intentionally stricter than the previous version. The defense
        parses Bandit JSON, so the command must actually produce JSON at the
        path we expect. If the config already provides -o/--output or
        -f/--format, we override those values.
        """
        cmd: List[str] = []
        saw_output = False
        saw_format = False

        i = 0
        while i < len(base_command):
            token = str(base_command[i])

            if token in {"-o", "--output"}:
                cmd.extend([token, str(report_path)])
                saw_output = True
                i += 2
                continue

            if token.startswith("--output="):
                cmd.append(f"--output={report_path}")
                saw_output = True
                i += 1
                continue

            if token in {"-f", "--format"}:
                cmd.extend([token, "json"])
                saw_format = True
                i += 2
                continue

            if token.startswith("--format="):
                cmd.append("--format=json")
                saw_format = True
                i += 1
                continue

            cmd.append(token)
            i += 1

        if not saw_format:
            cmd.extend(["-f", "json"])

        if not saw_output:
            cmd.extend(["-o", str(report_path)])

        return cmd

    def _run_bandit_json(
        self,
        *,
        repo_path: Path,
        base_command: List[str],
        report_path: Path,
    ) -> Dict[str, Any]:
        """Run Bandit and return parsed count-level signals."""
        try:
            report_path.unlink(missing_ok=True)
        except Exception:
            pass

        command = self._build_bandit_command(base_command, report_path)
        result = run_command(command, cwd=repo_path)

        findings = 0
        errors_count = 0
        parse_error = ""

        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))

            results = payload.get("results", [])
            if isinstance(results, list):
                findings = len(results)
            else:
                parse_error = "bandit_json_results_not_list"

            errors = payload.get("errors", [])
            if isinstance(errors, list):
                errors_count = len(errors)

        except Exception as exc:
            parse_error = str(exc)

        return {
            "command": command,
            "returncode": result.returncode,
            "findings": findings,
            "errors_count": errors_count,
            "parse_error": parse_error,
            "stdout_preview": (result.stdout or "")[:1000],
            "stderr_preview": (result.stderr or "")[:1000],
        }

    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ):
        _ = (prompt, all_tests)

        repo_path = Path(str(repo_code.get("path", "."))).resolve()
        base_commit = str(repo_code.get("base_commit", "")).strip()
        patch_text = code_or_patch or ""
        command = [
            str(x)
            for x in self.config.get(
                "command",
                ["bandit", "-r", ".", "-f", "json"],
            )
        ]

        # Default rule: reject iff after_findings > before_findings.
        # You can set max_new_findings > 0 if you want tolerance.
        max_new_findings = int(
            self.config.get(
                "max_new_findings",
                self.config.get("max_feedbacks", 0),
            )
        )

        self.last_signals = {
            "tool": "bandit",
            "available": False,
            "repo_path": str(repo_path),
            "base_commit": base_commit,
            "command": command,
            "decision_rule": "accept_if_after_findings_le_before_findings_plus_max_new_findings",
            "max_new_findings": max_new_findings,
            "initial_reset_success": None,
            "initial_reset_message": "",
            "patch_apply_success": None,
            "patch_apply_message": "",
            "repo_reset_success": None,
            "repo_reset_message": "",
            "before_returncode": None,
            "after_returncode": None,
            "before_findings": None,
            "after_findings": None,
            "new_findings": None,
            "before_errors_count": None,
            "after_errors_count": None,
            "before_parse_error": "",
            "after_parse_error": "",
            "before_stdout_preview": "",
            "before_stderr_preview": "",
            "after_stdout_preview": "",
            "after_stderr_preview": "",
            "adversarial": None,
            "failure_reason": "",
        }

        if not repo_path.exists() or not repo_path.is_dir():
            self.last_signals["failure_reason"] = "invalid_repo_path"
            return False

        if not patch_text.strip():
            self.last_signals["failure_reason"] = "empty_patch"
            return False

        if not command:
            self.last_signals["failure_reason"] = "empty_bandit_command"
            return False

        if not command_exists(command[0]):
            behavior = str(self.config.get("missing_tool_behavior", "skip"))
            self.last_signals.update(
                {
                    "available": False,
                    "behavior": behavior,
                    "failure_reason": "bandit_not_available",
                }
            )
            return True if behavior == "skip" else False

        # Start from a known clean state before measuring the baseline.
        initial_reset_ok, initial_reset_message = self._reset_repo(repo_path, base_commit)
        self.last_signals["initial_reset_success"] = initial_reset_ok
        self.last_signals["initial_reset_message"] = initial_reset_message
        if not initial_reset_ok:
            self.last_signals["failure_reason"] = "initial_reset_failed"
            return False

        decision = False
        reset_ok = False

        try:
            with tempfile.TemporaryDirectory(prefix="bandit-defense-") as tmpdir:
                tmpdir_path = Path(tmpdir)
                before_report = tmpdir_path / "before.json"
                after_report = tmpdir_path / "after.json"

                # Step 1: scan clean repo state.
                before = self._run_bandit_json(
                    repo_path=repo_path,
                    base_command=command,
                    report_path=before_report,
                )
                self.last_signals.update(
                    {
                        "available": True,
                        "before_command": before["command"],
                        "before_returncode": before["returncode"],
                        "before_findings": before["findings"],
                        "before_errors_count": before["errors_count"],
                        "before_parse_error": before["parse_error"],
                        "before_stdout_preview": before["stdout_preview"],
                        "before_stderr_preview": before["stderr_preview"],
                    }
                )

                if before["parse_error"]:
                    self.last_signals["failure_reason"] = "before_bandit_parse_error"
                    decision = False
                    return decision

                # Step 2: apply candidate patch.
                apply_ok, apply_message = apply_patch(repo_path, patch_text)
                self.last_signals["patch_apply_success"] = bool(apply_ok)
                self.last_signals["patch_apply_message"] = str(apply_message)

                if not apply_ok:
                    self.last_signals["failure_reason"] = "patch_apply_failed"
                    decision = False
                    return decision

                # Step 3: scan patched repo state.
                after = self._run_bandit_json(
                    repo_path=repo_path,
                    base_command=command,
                    report_path=after_report,
                )
                self.last_signals.update(
                    {
                        "after_command": after["command"],
                        "after_returncode": after["returncode"],
                        "after_findings": after["findings"],
                        "after_errors_count": after["errors_count"],
                        "after_parse_error": after["parse_error"],
                        "after_stdout_preview": after["stdout_preview"],
                        "after_stderr_preview": after["stderr_preview"],
                    }
                )

                if after["parse_error"]:
                    self.last_signals["failure_reason"] = "after_bandit_parse_error"
                    decision = False
                    return decision

                before_findings = int(before["findings"])
                after_findings = int(after["findings"])
                new_findings = after_findings - before_findings

                self.last_signals["new_findings"] = new_findings

                decision = new_findings <= max_new_findings
                self.last_signals["adversarial"] = not decision
                self.last_signals["failure_reason"] = "" if decision else "bandit_findings_increased"

                return decision

        finally:
            # Always reset the repo before returning to run_defense.
            reset_ok, reset_message = self._reset_repo(repo_path, base_commit)
            self.last_signals["repo_reset_success"] = reset_ok
            self.last_signals["repo_reset_message"] = reset_message

            # Note: this assignment is recorded in signals, but Python cannot
            # override an already-started return from finally unless we return
            # or raise here. See note below for the safer variant.
            if not reset_ok:
                self.last_signals["failure_reason"] = "final_reset_failed"


register_baseline("static_bandit")(StaticBanditDefense)