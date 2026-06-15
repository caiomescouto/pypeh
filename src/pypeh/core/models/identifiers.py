from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ulid import ULID


@dataclass(frozen=True)
class IdentifierContext:
    resource_class: type
    label: str | None = None
    described_by: str | None = None


class IdentifierProvider(Protocol):
    def mint(self, context: IdentifierContext) -> str: ...


class UlidIdentifierProvider:
    def __init__(self, suffix_strategy: Callable[[], str] | None = None):
        self.suffix_strategy = suffix_strategy or generate_ulid

    def mint(self, context: IdentifierContext) -> str:
        return self.suffix_strategy()


def generate_ulid() -> str:
    return str(ULID())
