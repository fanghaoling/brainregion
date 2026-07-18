from __future__ import annotations

import pytest

from brainregion.runtime.pricing import (
    canonical_model_name,
    estimate_cost_usd,
    merge_usage,
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


def test_normalize_usage_accepts_responses_token_details():
    usage = {
        "input_tokens": 8_242,
        "output_tokens": 52,
        "total_tokens": 8_294,
        "input_tokens_details": {"cached_tokens": 5_888},
        "output_tokens_details": {"reasoning_tokens": 11},
    }

    assert normalize_usage(usage) == {
        "input_tokens": 8_242,
        "output_tokens": 52,
        "total_tokens": 8_294,
        "cached_tokens": 5_888,
        "reasoning_tokens": 11,
    }


def test_merge_usage_sums_normalized_categories():
    merged = merge_usage(
        {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 125,
         "prompt_tokens_details": {"cached_tokens": 30}},
        {"input_tokens": 50, "output_tokens": 10, "reasoning_tokens": 4},
    )

    assert merged == {
        "input_tokens": 150,
        "output_tokens": 30,
        "total_tokens": 185,
        "cached_tokens": 30,
        "reasoning_tokens": 4,
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


def test_deepseek_builtin_price_estimates_openai_compatible_usage():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}

    cost, source = estimate_cost_usd("openai/deepseek-v4-flash", usage)

    assert cost == pytest.approx(3.0 / 7.2)
    assert source == "builtin"


def test_sonnet_5_builtin_uses_current_introductory_list_price():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}

    cost, source = estimate_cost_usd("anthropic/claude-sonnet-5", usage)

    assert cost == 12.0
    assert source == "builtin"


def test_model_usage_payload_replaces_unknown_provider_zero_with_known_estimate():
    payload = model_usage_payload(
        provider="openai",
        model="deepseek-v4-flash",
        resolved_model="openai/deepseek-v4-flash",
        endpoint_id="deepseek_openai",
        usage={"prompt_tokens": 1_000, "completion_tokens": 500},
        cost_usd=0.0,
        latency_ms=10,
        status="ok",
    )

    expected = (1_000 * (1.0 / 7.2) + 500 * (2.0 / 7.2)) / 1_000_000
    assert payload["cost_usd"] == expected
    assert payload["cost_source"] == "builtin"


def test_model_usage_payload_does_not_treat_unknown_provider_zero_as_free():
    payload = model_usage_payload(
        provider="anthropic",
        model="future-model",
        resolved_model="anthropic/future-model",
        endpoint_id="relay",
        usage={"prompt_tokens": 1_000, "completion_tokens": 500},
        cost_usd=0.0,
        latency_ms=10,
        status="ok",
    )

    assert payload["cost_usd"] is None
    assert payload["cost_source"] == "missing_price"


def test_price_for_unknown_model_missing():
    assert price_for_model("unknown-provider/unknown-model") is None
    cost, source = estimate_cost_usd("unknown-model", {"prompt_tokens": 10})
    assert cost is None
    assert source == "missing_price"
