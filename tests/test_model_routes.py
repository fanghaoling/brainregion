from __future__ import annotations

import pytest

from brainregion import server
from brainregion.server import _describe_model_routes, _normalize_panel, _suggest_panel


def _defaults_for_suggest_panel():
    return {
        "panel": [],
        "model_profiles": {
            "relay/cheap-mini": {
                "activation_role": "sleep",
                "tier": "economy",
                "tags": ["cheap", "fast"],
                "quality_score": 0.6,
                "cost_score": 0.95,
                "speed_score": 0.9,
                "structured_output_score": 0.7,
            },
            "relay/deep-opus": {
                "activation_role": "awake",
                "tier": "flagship",
                "tags": ["deep_reasoning", "architecture"],
                "quality_score": 0.98,
                "cost_score": 0.2,
                "speed_score": 0.4,
                "structured_output_score": 0.85,
                "context_score": 0.9,
            },
        },
        "endpoints": {
            "relay": {
                "provider": "openai",
                "base_url": "https://relay.example/v1",
                "api_key_env": "RELAY_KEY",
                "models": ["cheap-mini", "deep-opus"],
            },
        },
    }


def test_normalize_panel_accepts_endpoint_model_objects():
    cfg = {
        "relay": {
            "models": [
                {"id": "cheap-model", "tier": "economy", "cost": "low"},
                {"model": "flagship-model", "profile": {"tier": "flagship"}},
            ]
        }
    }

    entries = _normalize_panel(["relay"], {"relay"}, cfg)
    assert entries == [
        {"label": "relay/cheap-model", "model": "cheap-model", "endpoint_id": "relay"},
        {"label": "relay/flagship-model", "model": "flagship-model", "endpoint_id": "relay"},
    ]


def test_describe_model_routes_includes_profiles_and_warnings(monkeypatch):
    monkeypatch.setenv("MODEBRIDGE_API_KEY", "secret")
    defaults = {
        "panel": ["claude-opus-4-8", "modelbridge_anthropic/claude-opus-4-8"],
        "model_profiles": {
            "modelbridge_anthropic/claude-opus-4-8": {
                "activation_role": "awake",
                "tags": ["architecture"],
                "quality_score": 0.98,
                "context_window_tokens": "180000",
            }
        },
        "endpoints": {
            "modelbridge_anthropic": {
                "provider": "anthropic",
                "base_url": "https://www.modelbridge.cloud",
                "api_key_env": "MODEBRIDGE_API_KEY",
                "models": [
                    {
                        "id": "claude-opus-4-8",
                        "tier": "flagship",
                        "cost": "high",
                        "latency": "medium",
                        "context_window_tokens": 200000,
                    }
                ],
            }
        },
    }

    routes = _describe_model_routes(None, defaults, panel_source="test")
    bare, endpoint = routes["resolved_panel"]

    assert bare["route_type"] == "official_litellm"
    assert bare["profile"]["tier"] == "flagship"
    assert "deep_reasoning" in bare["profile"]["tags"]
    assert endpoint["route_type"] == "configured_endpoint"
    assert endpoint["profile"]["activation_role"] == "awake"
    assert endpoint["profile"]["quality_score"] == 0.98
    assert endpoint["profile"]["context_window_tokens"] == 200000
    assert endpoint["profile"]["context_window_source"] == "endpoint_model"
    assert "architecture" in endpoint["profile"]["tags"]
    assert routes["endpoints"][0]["model_profiles"][0]["profile"]["tier"] == "flagship"
    assert routes["warnings"][0]["type"] == "bare_model_has_endpoint_ref"


def test_model_profile_ignores_invalid_context_window_capacity():
    defaults = {
        "panel": ["relay/model-a", "relay/model-b", "relay/model-c"],
        "endpoints": {
            "relay": {
                "models": [
                    {"id": "model-a", "context_window_tokens": 0},
                    {"id": "model-b", "context_window_tokens": True},
                    {"id": "model-c", "context_window_tokens": 12.5},
                ]
            }
        },
    }

    routes = _describe_model_routes(None, defaults, panel_source="test")

    assert all(
        "context_window_tokens" not in route["profile"]
        for route in routes["resolved_panel"]
    )


def test_describe_model_routes_reports_missing_key_and_duplicate_endpoint_models(monkeypatch):
    monkeypatch.delenv("RELAY_KEY", raising=False)
    defaults = {
        "panel": ["endpoints"],
        "endpoints": {
            "relay_a": {
                "provider": "openai",
                "base_url": "https://a.example/v1",
                "api_key_env": "RELAY_KEY",
                "models": ["gpt-5.5"],
            },
            "relay_b": {
                "provider": "openai",
                "base_url": "https://b.example/v1",
                "api_key_env": "RELAY_KEY",
                "models": ["gpt-5.5"],
            },
        },
    }

    routes = _describe_model_routes(None, defaults, panel_source="test")
    assert routes["available_model_refs"] == ["relay_a/gpt-5.5", "relay_b/gpt-5.5"]
    assert {warning["type"] for warning in routes["warnings"]} == {
        "missing_endpoint_key",
        "model_declared_under_multiple_endpoints",
    }


def test_describe_model_routes_warns_for_unconfigured_gateway_prefix():
    defaults = {
        "panel": ["modelbridge_anthropic/claude-opus-4-8", "deepseek/deepseek-v4-pro"],
        "endpoints": {},
    }

    routes = _describe_model_routes(None, defaults, panel_source="test")

    assert [route["route_type"] for route in routes["resolved_panel"]] == [
        "official_litellm",
        "official_litellm",
    ]
    assert [warning["type"] for warning in routes["warnings"]] == ["unknown_endpoint_prefix"]
    assert routes["warnings"][0]["endpoint_id"] == "modelbridge_anthropic"


def test_endpoint_model_object_requires_id_or_model():
    with pytest.raises(ValueError, match="id or model"):
        _normalize_panel(["relay"], {"relay"}, {"relay": {"models": [{"tier": "economy"}]}})


def test_suggest_panel_prefers_cheap_fast_model(monkeypatch):
    monkeypatch.setenv("RELAY_KEY", "secret")

    result = _suggest_panel(
        defaults=_defaults_for_suggest_panel(),
        strategy="cheap_fast",
        task="Need a quick low-cost routing check.",
        max_models=1,
    )

    assert result["selected_panel"] == ["relay/cheap-mini"]
    assert result["selected"][0]["score"] > result["candidates"][1]["score"]
    assert result["trace"]["models_called"] is False
    assert result["trace"]["auto_execute"] is False


def test_suggest_panel_prefers_flagship_for_deep_reasoning(monkeypatch):
    monkeypatch.setenv("RELAY_KEY", "secret")

    result = _suggest_panel(
        defaults=_defaults_for_suggest_panel(),
        strategy="best_reasoning",
        task="Architecture planning needs deep reasoning.",
        max_models=1,
    )

    assert result["selected_panel"] == ["relay/deep-opus"]
    assert result["selected"][0]["bonuses"]["reasoning_tag_or_tier"] == 0.08


def test_suggest_panel_respects_missing_key_by_default(monkeypatch):
    monkeypatch.delenv("RELAY_KEY", raising=False)

    result = _suggest_panel(
        defaults=_defaults_for_suggest_panel(),
        strategy="sleep",
        max_models=1,
    )

    assert result["selected_panel"] == []
    assert {candidate["selectable"] for candidate in result["candidates"]} == {False}
    assert result["candidates"][0]["excluded_reason"] == "credential_missing"
    assert result["trace"]["no_selection_reason"] == "no_selectable_models"

    relaxed = _suggest_panel(
        defaults=_defaults_for_suggest_panel(),
        strategy="sleep",
        max_models=1,
        require_available_key=False,
    )
    assert relaxed["selected_panel"] == ["relay/cheap-mini"]


def test_suggest_panel_tool_uses_defaults(monkeypatch):
    monkeypatch.setenv("RELAY_KEY", "secret")
    defaults = _defaults_for_suggest_panel()
    monkeypatch.setattr(
        server._defaults_mod,
        "get_all",
        lambda: {key: {"value": value, "source": "test"} for key, value in defaults.items()},
    )

    result = server.suggest_panel(strategy="awake", max_models=1)

    assert result["selected_panel"] == ["relay/deep-opus"]
    assert result["trace"]["candidate_source"] == "configured_endpoints"
