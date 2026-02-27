"""Defense/baseline registry."""

from __future__ import annotations

from src.common.registry import Registry

BASELINE_REGISTRY = Registry(kind="baseline")


def register_baseline(name: str):
    return BASELINE_REGISTRY.register(name)


def get_baseline(name: str):
    return BASELINE_REGISTRY.get(name)


def list_baselines():
    return list(BASELINE_REGISTRY.names())
