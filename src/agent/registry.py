"""Agent registry."""

from __future__ import annotations

from src.common.registry import Registry

AGENT_REGISTRY = Registry(kind="agent")


def register_agent(name: str):
    return AGENT_REGISTRY.register(name)


def get_agent(name: str):
    return AGENT_REGISTRY.get(name)


def list_agents():
    return list(AGENT_REGISTRY.names())
