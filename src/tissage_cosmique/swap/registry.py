"""Global registry mapping computation functions to emulators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..emulators.base import Emulator


@dataclass
class SwapEntry:
    """An emulator registered to replace a computation function."""

    emulator: Emulator
    param_names: list[str]
    enabled: bool = True


_registry: dict[Callable[..., Any], SwapEntry] = {}


def register(
    fn: Callable[..., Any],
    emulator: Emulator,
    param_names: list[str],
    *,
    enabled: bool = True,
) -> None:
    """Register an emulator to replace a computation function."""
    _registry[fn] = SwapEntry(emulator=emulator, param_names=param_names, enabled=enabled)


def unregister(fn: Callable[..., Any]) -> None:
    """Remove an emulator registration."""
    _registry.pop(fn, None)


def get_entry(fn: Callable[..., Any]) -> SwapEntry | None:
    """Look up the swap entry for a function."""
    return _registry.get(fn)


def is_registered(fn: Callable[..., Any]) -> bool:
    """Check if a function has a registered emulator."""
    return fn in _registry


def enable(fn: Callable[..., Any]) -> None:
    """Enable the emulator swap for a function."""
    entry = _registry.get(fn)
    if entry is not None:
        entry.enabled = True


def disable(fn: Callable[..., Any]) -> None:
    """Disable the emulator swap for a function (calls fall through to original)."""
    entry = _registry.get(fn)
    if entry is not None:
        entry.enabled = False


def enable_all() -> None:
    """Enable emulator swaps for all registered functions."""
    for entry in _registry.values():
        entry.enabled = True


def disable_all() -> None:
    """Disable emulator swaps for all registered functions."""
    for entry in _registry.values():
        entry.enabled = False


def list_registered() -> dict[str, SwapEntry]:
    """Return a dict of {function_qualname: SwapEntry} for all registrations."""
    return {fn.__qualname__: entry for fn, entry in _registry.items()}


def reset() -> None:
    """Clear the entire swap registry."""
    _registry.clear()
