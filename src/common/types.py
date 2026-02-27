"""Canonical data types shared across datasets, agents, attacks, defenses, and eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class Patch:
    unified_diff: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepoSnapshot:
    repo_id: str
    path: str
    base_commit: str


@dataclass
class TestSpec:
    name: str
    command: List[str]
    cwd: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProblemInstance:
    dataset: str
    split: str
    instance_id: str
    prompt: str
    repo_snapshot: RepoSnapshot
    tests: List[TestSpec]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackResult:
    adv_prompt: str
    signals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DefenseResult:
    decision: Literal["accept", "reject", "edit"]
    edited_prompt: Optional[str] = None
    edited_patch: Optional[Patch] = None
    signals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetLoadResult:
    instances: List[ProblemInstance]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


DefenseReturn = Union[bool, str]
