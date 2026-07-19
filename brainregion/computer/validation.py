"""Shared validation primitives for computer-use contracts and locators.

Public package-internal API: ``contracts.py`` and ``locator.py`` import these
instead of reaching into each other's private helpers, keeping validation
single-source. Internal modules use relative imports (``from .validation``)
never the package root, to avoid circular imports through ``__init__``.
"""

from __future__ import annotations

import re
from typing import Any


HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def required_text(value: Any, name: str, *, max_length: int = 1000) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    if len(text) > max_length:
        raise ValueError(f"{name} cannot exceed {max_length} characters")
    return text


def identifier(value: Any, name: str) -> str:
    text = required_text(value, name, max_length=200)
    if any(char.isspace() for char in text):
        raise ValueError(f"{name} cannot contain whitespace")
    return text


def sha256_digest(value: Any, name: str) -> str:
    text = str(value or "").strip().casefold()
    if not HEX_SHA256.fullmatch(text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def strict_fields(data: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{name} unknown field(s): {sorted(unknown)}")


def attributes(value: Any, name: str) -> tuple[tuple[str, Any], ...]:
    """Normalize a JSON-object attribute map into a sorted tuple of (key, scalar) pairs."""
    if value in (None, ""):
        return ()
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    if len(value) > 32:
        raise ValueError(f"{name} cannot contain more than 32 entries")
    output: list[tuple[str, Any]] = []
    for raw_key, raw_value in value.items():
        key = identifier(raw_key, f"{name} key")
        if not isinstance(raw_value, (str, int, float, bool, type(None))):
            raise ValueError(f"{name}.{key} must be a JSON scalar")
        if isinstance(raw_value, str) and len(raw_value) > 2000:
            raise ValueError(f"{name}.{key} cannot exceed 2000 characters")
        output.append((key, raw_value))
    return tuple(sorted(output))
