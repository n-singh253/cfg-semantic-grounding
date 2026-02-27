"""Dataset adapter registry."""

from __future__ import annotations

from src.common.registry import Registry

DATASET_REGISTRY = Registry(kind="dataset")


def register_dataset(name: str):
    return DATASET_REGISTRY.register(name)


def get_dataset(name: str):
    return DATASET_REGISTRY.get(name)


def list_datasets():
    return list(DATASET_REGISTRY.names())
