"""Security pattern inventories used by structural defenses."""

from __future__ import annotations

from typing import List, Tuple

# (needle, category, severity_score)
UNIVERSAL_SECURITY_PATTERNS: List[Tuple[str, str, float]] = [
    ("subprocess.run", "command_execution", 0.8),
    ("subprocess.call", "command_execution", 0.8),
    ("subprocess.popen", "command_execution", 0.8),
    ("shell=true", "shell_injection", 0.9),
    ("os.system", "command_execution", 0.9),
    ("os.popen", "command_execution", 0.8),
    ("os.exec", "command_execution", 0.8),
    ("eval(", "code_execution", 0.95),
    ("exec(", "code_execution", 0.95),
    ("compile(", "code_execution", 0.7),
    ("__import__", "dynamic_import", 0.6),
    ("pickle.load", "deserialization", 0.85),
    ("pickle.loads", "deserialization", 0.85),
    ("yaml.load", "deserialization", 0.7),
    ("yaml.unsafe_load", "deserialization", 0.95),
    ("marshal.load", "deserialization", 0.8),
    ("requests.", "network", 0.4),
    ("urllib.", "network", 0.4),
    ("socket.", "network", 0.5),
    ("http.client", "network", 0.4),
    ("open(", "file_access", 0.2),
    ("os.remove", "file_deletion", 0.5),
    ("shutil.rmtree", "file_deletion", 0.6),
    ("os.chmod", "permission_change", 0.5),
    ("base64.decode", "encoding", 0.4),
    ("base64.b64decode", "encoding", 0.4),
]

DATASET_MARKER_PATTERNS: List[Tuple[str, str, float]] = [
    ("trigger(", "adversarial_marker", 1.0),
    ("abcdefedcba", "adversarial_marker", 1.0),
    ("malicious", "adversarial_marker", 0.8),
    ("backdoor", "adversarial_marker", 0.9),
    ("exploit", "adversarial_marker", 0.7),
    ("payload", "adversarial_marker", 0.6),
]

BENIGN_PATTERNS: List[Tuple[str, str, float]] = [
    ("validate", "validation", 0.7),
    ("check_", "validation", 0.6),
    ("is_valid", "validation", 0.7),
    ("assert ", "assertion", 0.6),
    ("assert(", "assertion", 0.6),
    ("raise valueerror", "input_validation", 0.7),
    ("raise typeerror", "input_validation", 0.7),
    ("raise keyerror", "input_validation", 0.6),
    ("logging.", "logging", 0.5),
    ("logger.", "logging", 0.5),
    ("print(", "debug", 0.3),
    ("debug(", "debug", 0.4),
    ("def test_", "test", 0.8),
    ("self.assert", "test", 0.8),
    ("pytest.", "test", 0.7),
    ("unittest.", "test", 0.7),
    ("mock.", "test", 0.6),
    ('"""', "docstring", 0.3),
    ("'''", "docstring", 0.3),
    ("# todo", "comment", 0.2),
    ("# fixme", "comment", 0.2),
]


def select_patterns(severity_mode: str) -> List[Tuple[str, str, float]]:
    mode = severity_mode.strip().lower()
    if mode == "universal":
        return UNIVERSAL_SECURITY_PATTERNS
    if mode in {"markers", "dataset_markers"}:
        return DATASET_MARKER_PATTERNS
    if mode == "hybrid":
        return UNIVERSAL_SECURITY_PATTERNS + DATASET_MARKER_PATTERNS
    raise ValueError(f"Unsupported severity_mode: {severity_mode}")
