"""Model usage and cost helpers for runtime telemetry.

Prices change over time, so this module treats built-ins as defaults and lets
project config override them via ``model_prices``. Values are USD per 1M tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from brainregion import defaults


@dataclass(frozen=True)
class ModelPrice:
    model: str
    input_usd_per_1m: float
    output_usd_per_1m: float
    source: str = "builtin"


_BUILTIN_PRICES: dict[str, ModelPrice] = {
    "claude-opus-4-8": ModelPrice("claude-opus-4-8", 5.0, 25.0),
    "claude-opus-4-7": ModelPrice("claude-opus-4-7", 5.0, 25.0),
    "claude-opus-4-6": ModelPrice("claude-opus-4-6", 5.0, 25.0),
    "claude-sonnet-4-6": ModelPrice("claude-sonnet-4-6", 3.0, 15.0),
    "claude-haiku-4-5": ModelPrice("claude-haiku-4-5", 1.0, 5.0),
    "claude-fable-5": ModelPrice("claude-fable-5", 10.0, 50.0),
    "gpt-4o": ModelPrice("gpt-4o", 2.5, 10.0),
    "gpt-5": ModelPrice("gpt-5", 5.0, 15.0),
    "gpt-5.5": ModelPrice("gpt-5.5", 5.0, 15.0),
}


def canonical_model_name(model: str | None) -> str:
    name = (model or "").strip()
    if "/" in name:
        return name.rsplit("/", 1)[-1]
    return name


def _float_value(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in data:
            continue
        try:
            return float(data[key])
        except (TypeError, ValueError):
            continue
    return None


def load_model_prices(config: dict[str, Any] | None = None) -> dict[str, ModelPrice]:
    prices = dict(_BUILTIN_PRICES)
    config_prices = config
    if config_prices is None:
        try:
            config_prices = defaults.apply().get("model_prices") or {}
        except Exception:
            config_prices = {}
    if not isinstance(config_prices, dict):
        return prices

    for model, raw in config_prices.items():
        if not isinstance(raw, dict):
            continue
        in_price = _float_value(raw, "input_usd_per_1m", "input_per_1m", "input")
        out_price = _float_value(raw, "output_usd_per_1m", "output_per_1m", "output")
        if in_price is None or out_price is None:
            continue
        key = canonical_model_name(str(model))
        prices[key] = ModelPrice(key, in_price, out_price, source="config")
    return prices


def price_for_model(model: str, *, config: dict[str, Any] | None = None) -> ModelPrice | None:
    prices = load_model_prices(config)
    return prices.get(model) or prices.get(canonical_model_name(model))


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    cached_tokens = int(prompt_details.get("cached_tokens") or usage.get("cached_tokens") or 0)
    reasoning_tokens = int(completion_details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def estimate_cost_usd(model: str, usage: dict[str, Any] | None, *, config: dict[str, Any] | None = None) -> tuple[float | None, str]:
    normalized = normalize_usage(usage)
    price = price_for_model(model, config=config)
    if price is None:
        return None, "missing_price"
    cost = (
        normalized["input_tokens"] * price.input_usd_per_1m
        + normalized["output_tokens"] * price.output_usd_per_1m
    ) / 1_000_000
    return cost, price.source


def model_usage_payload(
    *,
    provider: str | None,
    model: str,
    resolved_model: str | None,
    endpoint_id: str | None,
    usage: dict[str, Any] | None,
    cost_usd: float | None,
    latency_ms: float | None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    estimated_cost, price_source = estimate_cost_usd(resolved_model or model, usage)
    final_cost = cost_usd if cost_usd is not None else estimated_cost
    return {
        "provider": provider,
        "model": model,
        "resolved_model": resolved_model or model,
        "canonical_model": canonical_model_name(resolved_model or model),
        "endpoint_id": endpoint_id,
        "usage": normalize_usage(usage),
        "raw_usage": usage or {},
        "cost_usd": final_cost,
        "cost_source": "provider" if cost_usd is not None else price_source,
        "latency_ms": latency_ms,
        "status": status,
        "error": error,
    }
