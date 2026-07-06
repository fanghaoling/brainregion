from __future__ import annotations

from brainregion.runtime.pricing import (
    canonical_model_name,
    estimate_cost_usd,
    model_usage_payload,
    normalize_usage,
    price_for_model,
)


def test_canonical_model_name_strips_provider_prefix():
    assert canonical_model_name("modelbridge_anthropic/claude-opus-4-8") == "claude-opus-4-8"
    assert canonical_model_name("anthropic/claude-opus-4-8") == "claude-opus-4-8"
    assert canonical_model_name("gpt-5.5") == "gpt-5.5"


def test_normalize_usage_keeps_token_categories():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 125,
        "prompt_tokens_details": {"cached_tokens": 40},
        "completion_tokens_details": {"reasoning_tokens": 7},
    }

    assert normalize_usage(usage) == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 125,
        "cached_tokens": 40,
        "reasoning_tokens": 7,
    }


def test_estimate_cost_uses_canonical_model_and_config_override():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}

    cost, source = estimate_cost_usd(
        "modelbridge_anthropic/claude-opus-4-8",
        usage,
        config={"claude-opus-4-8": {"input_usd_per_1m": 1, "output_usd_per_1m": 2}},
    )

    assert cost == 3
    assert source == "config"


def test_model_usage_payload_prefers_provider_cost():
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    payload = model_usage_payload(
        provider="anthropic",
        model="claude-opus-4-8",
        resolved_model="anthropic/claude-opus-4-8",
        endpoint_id="modelbridge_anthropic",
        usage=usage,
        cost_usd=0.123,
        latency_ms=42.5,
        status="ok",
    )

    assert payload["canonical_model"] == "claude-opus-4-8"
    assert payload["usage"]["input_tokens"] == 10
    assert payload["cost_usd"] == 0.123
    assert payload["cost_source"] == "provider"


def test_price_for_unknown_model_missing():
    assert price_for_model("unknown-provider/unknown-model") is None
    cost, source = estimate_cost_usd("unknown-model", {"prompt_tokens": 10})
    assert cost is None
    assert source == "missing_price"
