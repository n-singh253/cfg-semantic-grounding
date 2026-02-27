"""Simple plugin registry used across dataset/agent/attack/baseline modules."""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterable, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        key = name.strip().lower()

        def _decorator(item: T) -> T:
            if key in self._items:
                raise ValueError(f"Duplicate {self.kind} registration: {key}")
            self._items[key] = item
            return item

        return _decorator

    def get(self, name: str) -> T:
        key = name.strip().lower()
        if key not in self._items:
            known = ", ".join(sorted(self._items))
            raise KeyError(f"Unknown {self.kind} '{name}'. Known: [{known}]")
        return self._items[key]

    def names(self) -> Iterable[str]:
        return sorted(self._items)
