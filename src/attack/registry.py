"""Attack registry."""

from __future__ import annotations

from src.common.registry import Registry

ATTACK_REGISTRY = Registry(kind="attack")


def register_attack(name: str):
    return ATTACK_REGISTRY.register(name)


def get_attack(name: str):
    return ATTACK_REGISTRY.get(name)


def list_attacks():
    return list(ATTACK_REGISTRY.names())
