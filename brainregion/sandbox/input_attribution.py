"""Content-free attribution for input tokens sent by the sandbox main model."""

from __future__ import annotations

from typing import Any, Iterable

from brainregion.core.context_loader import estimate_context_tokens
from brainregion.runtime import normalize_usage

INPUT_CATEGORIES = (
    "system",
    "scaffold",
    "task",
    "control_feedback",
    "memory_context",
    "expert_context",
    "region_context",
    "checkpoint",
    "model_transcript",
    "tool_transcript",
    "visual",
    "error_feedback",
    "other",
    "protocol",
)
_CATEGORY_SET = frozenset(INPUT_CATEGORIES)
_CATEGORY_KEY = "_brainregion_input_category"
_ESTIMATES_KEY = "_brainregion_input_estimates"
_CHARACTERS_KEY = "_brainregion_input_characters"
_MESSAGE_OVERHEAD_TOKENS = 4
_REQUEST_OVERHEAD_TOKENS = 3


def attributed_message(role: str, content: str, category: str) -> dict[str, Any]:
    """Create one internal message with content-free attribution metadata."""
    category = _category(category)
    text = str(content or "")
    return {
        "role": role,
        "content": text,
        _CATEGORY_KEY: category,
    }


def compound_message(
    role: str,
    parts: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Join prompt parts without duplicating their content in internal metadata."""
    rendered: list[str] = []
    estimates: dict[str, int] = {}
    characters: dict[str, int] = {}
    for raw_category, raw_text in parts:
        category = _category(raw_category)
        text = str(raw_text or "")
        rendered.append(text)
        estimates[category] = estimates.get(category, 0) + estimate_context_tokens(text)
        characters[category] = characters.get(category, 0) + len(text)
    if not rendered:
        raise ValueError("compound input message requires at least one part")
    return {
        "role": role,
        "content": "".join(rendered),
        _ESTIMATES_KEY: estimates,
        _CHARACTERS_KEY: characters,
    }


def provider_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove BrainRegion-only metadata before crossing the provider boundary."""
    return [
        {
            key: value
            for key, value in message.items()
            if not str(key).startswith("_brainregion_")
        }
        for message in messages
    ]


def capture_input_attribution(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Capture estimated token weights for the exact messages about to be sent."""
    categories: dict[str, dict[str, int]] = {}
    for message in messages:
        estimates = message.get(_ESTIMATES_KEY)
        characters = message.get(_CHARACTERS_KEY)
        if isinstance(estimates, dict) and isinstance(characters, dict):
            touched = set(estimates) | set(characters)
            for raw_category in touched:
                category = _category(raw_category)
                _accumulate(
                    categories,
                    category,
                    estimated_tokens=_non_negative_int(estimates.get(raw_category)),
                    characters=_non_negative_int(characters.get(raw_category)),
                    messages=1,
                )
            continue
        category = _category(message.get(_CATEGORY_KEY) or "other")
        content = str(message.get("content") or "")
        _accumulate(
            categories,
            category,
            estimated_tokens=estimate_context_tokens(content),
            characters=len(content),
            messages=1,
        )

    protocol_tokens = len(messages) * _MESSAGE_OVERHEAD_TOKENS + _REQUEST_OVERHEAD_TOKENS
    _accumulate(
        categories,
        "protocol",
        estimated_tokens=protocol_tokens,
        characters=0,
        messages=len(messages),
    )
    return _report(categories, calls=1, provider_reported_calls=0, actual_input_tokens=0)


def reconcile_input_attribution(
    captured: dict[str, Any],
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    """Allocate provider input tokens by estimated category share."""
    normalized = normalize_usage(usage)
    actual = normalized["input_tokens"]
    categories = _copy_categories(captured)
    allocation = _allocate_actual_tokens(
        {category: values["estimated_tokens"] for category, values in categories.items()},
        actual,
    )
    for category, values in categories.items():
        values["actual_input_tokens"] = allocation.get(category, 0)
    return _report(
        categories,
        calls=1,
        provider_reported_calls=int(actual > 0),
        actual_input_tokens=actual,
        cached_input_tokens=normalized["cached_tokens"],
    )


def merge_input_attributions(reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-call reports without retaining prompt or tool-result content."""
    categories: dict[str, dict[str, int]] = {}
    calls = 0
    reported_calls = 0
    actual = 0
    cached = 0
    for report in reports:
        if not isinstance(report, dict):
            continue
        calls += _non_negative_int(report.get("calls"))
        reported_calls += _non_negative_int(report.get("provider_reported_calls"))
        actual += _non_negative_int(report.get("actual_input_tokens"))
        cached += _non_negative_int(report.get("cached_input_tokens"))
        for raw_category, raw_values in (report.get("categories") or {}).items():
            if not isinstance(raw_values, dict):
                continue
            category = _category(raw_category)
            _accumulate(
                categories,
                category,
                estimated_tokens=_non_negative_int(raw_values.get("estimated_tokens")),
                characters=_non_negative_int(raw_values.get("characters")),
                messages=_non_negative_int(raw_values.get("messages")),
                actual_input_tokens=_non_negative_int(raw_values.get("actual_input_tokens")),
            )
    return _report(
        categories,
        calls=calls,
        provider_reported_calls=reported_calls,
        actual_input_tokens=actual,
        cached_input_tokens=cached,
    )


def _allocate_actual_tokens(estimates: dict[str, int], actual: int) -> dict[str, int]:
    if actual <= 0:
        return {category: 0 for category in estimates}
    total = sum(max(0, value) for value in estimates.values())
    if total <= 0:
        return {**{category: 0 for category in estimates}, "protocol": actual}
    allocation: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    assigned = 0
    for category in sorted(estimates):
        numerator = actual * max(0, estimates[category])
        quotient, remainder = divmod(numerator, total)
        allocation[category] = quotient
        assigned += quotient
        remainders.append((remainder, category))
    for _, category in sorted(remainders, key=lambda item: (-item[0], item[1]))[: actual - assigned]:
        allocation[category] += 1
    return allocation


def _report(
    categories: dict[str, dict[str, int]],
    *,
    calls: int,
    provider_reported_calls: int,
    actual_input_tokens: int,
    cached_input_tokens: int = 0,
) -> dict[str, Any]:
    estimated_total = sum(values.get("estimated_tokens", 0) for values in categories.values())
    normalized_categories: dict[str, dict[str, Any]] = {}
    for category in INPUT_CATEGORIES:
        values = categories.get(category)
        if not values:
            continue
        estimated = _non_negative_int(values.get("estimated_tokens"))
        actual = _non_negative_int(values.get("actual_input_tokens"))
        normalized_categories[category] = {
            "estimated_tokens": estimated,
            "actual_input_tokens": actual,
            "estimated_share": estimated / estimated_total if estimated_total else None,
            "actual_share": actual / actual_input_tokens if actual_input_tokens else None,
            "characters": _non_negative_int(values.get("characters")),
            "messages": _non_negative_int(values.get("messages")),
        }
    return {
        "method": "estimated_category_share_v1",
        "calls": _non_negative_int(calls),
        "provider_reported_calls": _non_negative_int(provider_reported_calls),
        "estimated_input_tokens": estimated_total,
        "actual_input_tokens": _non_negative_int(actual_input_tokens),
        "cached_input_tokens": _non_negative_int(cached_input_tokens),
        "allocation_status": (
            "provider_aligned" if actual_input_tokens > 0 else "estimate_only"
        ),
        "categories": normalized_categories,
        "contains_content": False,
        "contains_reasoning": False,
        "contains_tool_results": False,
    }


def _copy_categories(report: dict[str, Any]) -> dict[str, dict[str, int]]:
    categories: dict[str, dict[str, int]] = {}
    for raw_category, raw_values in (report.get("categories") or {}).items():
        if not isinstance(raw_values, dict):
            continue
        category = _category(raw_category)
        categories[category] = {
            "estimated_tokens": _non_negative_int(raw_values.get("estimated_tokens")),
            "actual_input_tokens": 0,
            "characters": _non_negative_int(raw_values.get("characters")),
            "messages": _non_negative_int(raw_values.get("messages")),
        }
    return categories


def _accumulate(
    categories: dict[str, dict[str, int]],
    category: str,
    *,
    estimated_tokens: int = 0,
    actual_input_tokens: int = 0,
    characters: int = 0,
    messages: int = 0,
) -> None:
    values = categories.setdefault(
        category,
        {"estimated_tokens": 0, "actual_input_tokens": 0, "characters": 0, "messages": 0},
    )
    values["estimated_tokens"] += _non_negative_int(estimated_tokens)
    values["actual_input_tokens"] += _non_negative_int(actual_input_tokens)
    values["characters"] += _non_negative_int(characters)
    values["messages"] += _non_negative_int(messages)


def _category(value: Any) -> str:
    category = str(value or "").strip().casefold()
    if category not in _CATEGORY_SET:
        raise ValueError(f"unknown input attribution category: {category!r}")
    return category


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "INPUT_CATEGORIES",
    "attributed_message",
    "capture_input_attribution",
    "compound_message",
    "merge_input_attributions",
    "provider_messages",
    "reconcile_input_attribution",
]
