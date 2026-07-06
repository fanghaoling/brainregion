"""v1.6 中转站/自定义 endpoint：endpoint 解析、PanelEntry 规范化、LiteLLMBackend per-call 透传。

不调网（monkeypatch litellm.acompletion 捕获参数）。覆盖：
- _resolve_endpoints（api_key_env 解析/明文 fallback/缺 key 报错/bad provider/headers/timeout）
- _normalize_panel（str 官方/dict 中转/label 默认/撞名报错/未知 endpoint/PanelEntry 不含 api_key）
- LiteLLMBackend（endpoint_id→provider 前缀/api_base/api_key/extra_headers/timeout；前缀守卫；官方无 endpoint；多 endpoint 不串）
- compute_hash 输入不含 api_key
"""
from __future__ import annotations

import asyncio
import json

import pytest

from brainregion import reviews_db
from brainregion.providers.litellm import LiteLLMBackend
from brainregion.server import _describe_model_routes, _normalize_one, _normalize_panel, _resolve_endpoints


# ===== _resolve_endpoints =====

def test_resolve_endpoints_api_key_env(monkeypatch):
    monkeypatch.setenv("MY_RELAY_KEY", "sk-relay-123")
    cfg = {"relay": {"provider": "openai", "base_url": "https://x/v1", "api_key_env": "MY_RELAY_KEY"}}
    reg = _resolve_endpoints(cfg)
    assert reg["relay"]["api_key"] == "sk-relay-123"
    assert reg["relay"]["provider"] == "openai"
    assert reg["relay"]["base_url"] == "https://x/v1"
    assert reg["relay"]["headers"] == {}
    assert reg["relay"]["timeout"] is None


def test_resolve_endpoints_plaintext_fallback():
    cfg = {"relay": {"provider": "anthropic", "base_url": "https://x", "api_key": "plain"}}
    reg = _resolve_endpoints(cfg)
    assert reg["relay"]["api_key"] == "plain"


def test_resolve_endpoints_missing_key(monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    cfg = {"relay": {"provider": "openai", "base_url": "https://x", "api_key_env": "NOPE_KEY"}}
    with pytest.raises(ValueError, match="NOPE_KEY"):
        _resolve_endpoints(cfg)


def test_resolve_endpoints_no_key_at_all():
    cfg = {"relay": {"provider": "openai", "base_url": "https://x"}}
    with pytest.raises(ValueError, match="缺 api_key"):
        _resolve_endpoints(cfg)


def test_resolve_endpoints_bad_provider():
    cfg = {"relay": {"provider": "gemini", "base_url": "https://x", "api_key": "k"}}
    with pytest.raises(ValueError, match="provider"):
        _resolve_endpoints(cfg)


def test_resolve_endpoints_headers_timeout():
    cfg = {"r": {"provider": "openai", "base_url": "https://x", "api_key": "k",
                 "headers": {"Authorization": "Bearer k"}, "timeout": 120}}
    reg = _resolve_endpoints(cfg)
    assert reg["r"]["headers"] == {"Authorization": "Bearer k"}
    assert reg["r"]["timeout"] == 120


# ===== _normalize_panel / _normalize_one =====

def test_normalize_panel_str_official():
    entries = _normalize_panel(["gpt-4o", "zai/glm-5.2"], set())
    assert entries == [
        {"label": "gpt-4o", "model": "gpt-4o", "endpoint_id": None},
        {"label": "zai/glm-5.2", "model": "zai/glm-5.2", "endpoint_id": None},
    ]


def test_normalize_panel_dict_relay_no_credential():
    entries = _normalize_panel(
        [{"endpoint": "zhipu", "model": "glm-5.2", "label": "智谱位"}], {"zhipu"}
    )
    assert entries == [{"label": "智谱位", "model": "glm-5.2", "endpoint_id": "zhipu"}]
    # 关键：PanelEntry 不含 credential（key 只在 registry）
    assert "api_key" not in entries[0]
    assert "api_key_env" not in entries[0]


def test_normalize_panel_label_default():
    entries = _normalize_panel([{"endpoint": "zhipu", "model": "glm-5.2"}], {"zhipu"})
    assert entries[0]["label"] == "zhipu/glm-5.2"


def test_normalize_panel_label_collision():
    with pytest.raises(ValueError, match="撞名"):
        _normalize_panel(
            [{"endpoint": "z", "model": "m", "label": "dup"},
             {"endpoint": "z", "model": "m2", "label": "dup"}],
            {"z"},
        )


def test_normalize_panel_unknown_endpoint():
    with pytest.raises(ValueError, match="未定义"):
        _normalize_panel([{"endpoint": "nope", "model": "m"}], {"zhipu"})


def test_normalize_one_str_and_dict():
    assert _normalize_one("gpt-4o", set()) == {"label": "gpt-4o", "model": "gpt-4o", "endpoint_id": None}
    assert _normalize_one({"endpoint": "z", "model": "m"}, {"z"}) == {
        "label": "z/m", "model": "m", "endpoint_id": "z"
    }


# ===== v2.3 短引用 endpoint_id/model + 全展开 endpoint_id =====

def test_normalize_panel_short_ref():
    """endpoint_id/model 短引用 → endpoint + model，label=id/model（省去 dict 一长串）。"""
    entries = _normalize_panel(["zhipu/glm-5.2"], {"zhipu"})
    assert entries == [{"label": "zhipu/glm-5.2", "model": "glm-5.2", "endpoint_id": "zhipu"}]


def test_normalize_panel_expand_all_models():
    """str == endpoint_id → 全展开 endpoints.<id>.models（一行引一家中转站全部模型）。"""
    cfg = {"zhipu": {"models": ["glm-5.2", "glm-4.7"]}}
    entries = _normalize_panel(["zhipu"], {"zhipu"}, cfg)
    assert entries == [
        {"label": "zhipu/glm-5.2", "model": "glm-5.2", "endpoint_id": "zhipu"},
        {"label": "zhipu/glm-4.7", "model": "glm-4.7", "endpoint_id": "zhipu"},
    ]


def test_normalize_panel_expand_requires_models_declared():
    """全展开但 endpoints.<id>.models 未声明 → 清晰报错。"""
    with pytest.raises(ValueError, match="models 未声明"):
        _normalize_panel(["zhipu"], {"zhipu"}, {})


def test_normalize_panel_short_ref_not_endpoint_falls_through():
    """str 含 / 但前缀非 endpoint_id → litellm 原生（zai/glm-5.2 仍走官方 env，不误判短引用）。"""
    entries = _normalize_panel(["zai/glm-5.2"], {"zhipu"})
    assert entries == [{"label": "zai/glm-5.2", "model": "zai/glm-5.2", "endpoint_id": None}]


def test_normalize_panel_short_ref_and_expand_no_label_collision():
    """短引用 + 全展开同 endpoint 不同 model → label 不撞（id/model 唯一）。"""
    cfg = {"zhipu": {"models": ["glm-4.7"]}}
    entries = _normalize_panel(["zhipu/glm-5.2", "zhipu"], {"zhipu"}, cfg)
    assert [e["label"] for e in entries] == ["zhipu/glm-5.2", "zhipu/glm-4.7"]


def test_normalize_panel_wildcard_endpoints():
    """panel 'endpoints' 通配 → 展开所有 endpoints 的所有 models（一行引全部中转站）。"""
    cfg = {
        "zhipu": {"models": ["glm-5.2"]},
        "openai_relay": {"models": ["gpt-4o", "o3"]},
    }
    entries = _normalize_panel(["endpoints"], {"zhipu", "openai_relay"}, cfg)
    assert [e["label"] for e in entries] == ["zhipu/glm-5.2", "openai_relay/gpt-4o", "openai_relay/o3"]
    assert all(e["endpoint_id"] in ("zhipu", "openai_relay") for e in entries)


def test_normalize_panel_wildcard_no_models_declared():
    """通配但无 endpoints.<id>.models 声明 → 清晰报错。"""
    with pytest.raises(ValueError, match="无任何"):
        _normalize_panel(["endpoints"], {"zhipu"}, {"zhipu": {}})


def test_normalize_panel_wildcard_mix_with_official():
    """通配 + 官方模型混合：'endpoints' 展开 + litellm 原生共存。"""
    cfg = {"zhipu": {"models": ["glm-5.2"]}}
    entries = _normalize_panel(["gpt-4o", "endpoints"], {"zhipu"}, cfg)
    assert [e["label"] for e in entries] == ["gpt-4o", "zhipu/glm-5.2"]
    assert entries[0]["endpoint_id"] is None      # gpt-4o 官方
    assert entries[1]["endpoint_id"] == "zhipu"   # 通配展开


def test_describe_model_routes_distinguishes_bare_and_endpoint_refs(monkeypatch):
    monkeypatch.setenv("MODEBRIDGE_API_KEY", "secret")
    defaults = {
        "panel": ["claude-opus-4-8", "modelbridge_anthropic/claude-opus-4-8"],
        "endpoints": {
            "modelbridge_anthropic": {
                "provider": "anthropic",
                "base_url": "https://www.modelbridge.cloud",
                "api_key_env": "MODEBRIDGE_API_KEY",
                "models": ["claude-opus-4-8"],
            }
        },
    }

    routes = _describe_model_routes(None, defaults, panel_source="test")
    bare, endpoint = routes["resolved_panel"]
    assert bare["route_type"] == "official_litellm"
    assert bare["credential_hint"] == "ANTHROPIC_API_KEY"
    assert endpoint["route_type"] == "configured_endpoint"
    assert endpoint["endpoint_id"] == "modelbridge_anthropic"
    assert endpoint["api_key_env"] == "MODEBRIDGE_API_KEY"
    assert endpoint["api_key_status"] == "set"
    assert routes["ambiguous_models"] == [
        {
            "model": "claude-opus-4-8",
            "endpoint_refs": ["modelbridge_anthropic/claude-opus-4-8"],
            "official_ref": "claude-opus-4-8",
            "reasons": ["bare_model_string_also_used"],
        }
    ]


def test_describe_model_routes_reports_same_model_on_multiple_endpoints(monkeypatch):
    monkeypatch.delenv("MODEBRIDGE_API_KEY", raising=False)
    defaults = {
        "panel": ["endpoints"],
        "endpoints": {
            "relay_a": {
                "provider": "openai",
                "base_url": "https://a.example/v1",
                "api_key_env": "MODEBRIDGE_API_KEY",
                "models": ["gpt-5.5"],
            },
            "relay_b": {
                "provider": "openai",
                "base_url": "https://b.example/v1",
                "api_key_env": "MODEBRIDGE_API_KEY",
                "models": ["gpt-5.5"],
            },
        },
    }

    routes = _describe_model_routes(None, defaults, panel_source="test")
    assert routes["available_model_refs"] == ["relay_a/gpt-5.5", "relay_b/gpt-5.5"]
    assert routes["endpoints"][0]["api_key_status"] == "missing"
    assert routes["ambiguous_models"][0]["reasons"] == ["declared_under_multiple_endpoints"]


def test_server_list_model_routes_tool(monkeypatch):
    from brainregion import server

    monkeypatch.setattr(
        server._defaults_mod,
        "get_all",
        lambda: {
            "panel": {"value": ["relay/m"], "source": "test_config"},
            "endpoints": {
                "value": {
                    "relay": {
                        "provider": "openai",
                        "base_url": "https://relay.example/v1",
                        "api_key_env": "RELAY_KEY",
                        "models": ["m"],
                    }
                },
                "source": "test_config",
            },
        },
    )

    configured = server.list_model_routes()
    assert configured["panel_source"] == "test_config"
    assert configured["resolved_panel"][0]["endpoint_id"] == "relay"

    explicit = server.list_model_routes(panel=["m"])
    assert explicit["panel_source"] == "explicit"
    assert explicit["resolved_panel"][0]["route_type"] == "official_litellm"


# ===== LiteLLMBackend：endpoint_id → litellm.acompletion 参数 =====

class _Capture:
    """捕获 litellm.acompletion 的 kwargs，返回假响应。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, **kw):
        self.calls.append(kw)

        class _Msg:
            content = '{"issues":[]}'

        class _Choice:
            message = _Msg()

        class _Usage:
            def model_dump(self):
                return {"total_tokens": 1}

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()
            _hidden_params = {}

        return _Resp()


def _patch_litellm(monkeypatch):
    cap = _Capture()
    import litellm
    monkeypatch.setattr(litellm, "acompletion", cap)
    return cap


def test_backend_openai_prefix_and_passthrough(monkeypatch):
    cap = _patch_litellm(monkeypatch)
    reg = {"r": {"provider": "openai", "base_url": "https://x/v1", "api_key": "k", "headers": {}, "timeout": None}}
    backend = LiteLLMBackend(endpoint_registry=reg)
    asyncio.run(backend.complete(model="glm-4", system="s", user="u", endpoint_id="r"))
    kw = cap.calls[0]
    assert kw["model"] == "openai/glm-4"  # openai 前缀
    assert kw["api_base"] == "https://x/v1"  # snake_case
    assert kw["api_key"] == "k"


def test_backend_anthropic_prefix(monkeypatch):
    cap = _patch_litellm(monkeypatch)
    reg = {"r": {"provider": "anthropic", "base_url": "https://open.bigmodel.cn/api/anthropic",
                 "api_key": "k", "headers": {}, "timeout": None}}
    backend = LiteLLMBackend(endpoint_registry=reg)
    asyncio.run(backend.complete(model="glm-5.2", system="s", user="u", endpoint_id="r"))
    kw = cap.calls[0]
    assert kw["model"] == "anthropic/glm-5.2"
    assert kw["api_base"] == "https://open.bigmodel.cn/api/anthropic"


def test_backend_anthropic_sampling_omits_top_p(monkeypatch):
    cap = _patch_litellm(monkeypatch)
    reg = {"r": {"provider": "anthropic", "base_url": "https://x", "api_key": "k", "headers": {}, "timeout": None}}
    backend = LiteLLMBackend(endpoint_registry=reg)
    asyncio.run(backend.complete(model="claude-haiku-4-5", system="s", user="u", endpoint_id="r"))
    kw = cap.calls[0]
    assert kw["temperature"] == 0.3
    assert "top_p" not in kw


def test_backend_anthropic_effort_uses_temperature_one(monkeypatch):
    cap = _patch_litellm(monkeypatch)
    reg = {"r": {"provider": "anthropic", "base_url": "https://x", "api_key": "k", "headers": {}, "timeout": None}}
    backend = LiteLLMBackend(endpoint_registry=reg)
    asyncio.run(backend.complete(model="claude-haiku-4-5", system="s", user="u", endpoint_id="r", effort="low"))
    kw = cap.calls[0]
    assert kw["temperature"] == 1
    assert "top_p" not in kw
    assert kw["thinking"] == {"type": "adaptive"}


def test_backend_prefix_guard(monkeypatch):
    """model 已含 / 则不再拼前缀，防 openai/openai/。"""
    cap = _patch_litellm(monkeypatch)
    reg = {"r": {"provider": "openai", "base_url": "https://x", "api_key": "k", "headers": {}, "timeout": None}}
    backend = LiteLLMBackend(endpoint_registry=reg)
    asyncio.run(backend.complete(model="openai/glm-4", system="s", user="u", endpoint_id="r"))
    assert cap.calls[0]["model"] == "openai/glm-4"


def test_backend_prefix_for_slash_model_id(monkeypatch):
    """中转站模型名本身含 /（如 SiliconFlow Qwen/Qwen3-8B）仍正确拼 provider 前缀。

    旧守卫用 "/" not in model 判断，会把 Qwen/Qwen3-8B 误判为"已加前缀"而漏拼 →
    litellm 拿到无前缀的 Qwen/Qwen3-8B，把 Qwen 当 provider，不认 → 调用失败。
    """
    cap = _patch_litellm(monkeypatch)
    reg = {"siliconflow": {"provider": "openai", "base_url": "https://api.siliconflow.cn/v1",
                           "api_key": "k", "headers": {}, "timeout": None}}
    backend = LiteLLMBackend(endpoint_registry=reg)
    asyncio.run(backend.complete(model="Qwen/Qwen3-8B", system="s", user="u", endpoint_id="siliconflow"))
    kw = cap.calls[0]
    assert kw["model"] == "openai/Qwen/Qwen3-8B"  # 拼 openai/ 前缀，model 名含 / 不影响
    assert kw["api_base"] == "https://api.siliconflow.cn/v1"
    assert kw["api_key"] == "k"


def test_backend_headers_and_timeout(monkeypatch):
    cap = _patch_litellm(monkeypatch)
    reg = {"r": {"provider": "openai", "base_url": "https://x", "api_key": "k",
                 "headers": {"Authorization": "Bearer k"}, "timeout": 120}}
    backend = LiteLLMBackend(endpoint_registry=reg)
    asyncio.run(backend.complete(model="m", system="s", user="u", endpoint_id="r"))
    kw = cap.calls[0]
    assert kw["extra_headers"] == {"Authorization": "Bearer k"}
    assert kw["timeout"] == 120  # endpoint timeout 覆盖全局


def test_backend_no_endpoint_official(monkeypatch):
    """endpoint_id=None 走官方 env，不加前缀、不传 api_base/api_key。"""
    cap = _patch_litellm(monkeypatch)
    backend = LiteLLMBackend(endpoint_registry={})
    asyncio.run(backend.complete(model="gpt-4o", system="s", user="u"))
    kw = cap.calls[0]
    assert kw["model"] == "gpt-4o"
    assert "api_base" not in kw
    assert "api_key" not in kw


def test_backend_emits_model_usage_events(monkeypatch):
    emitted = []

    class _Msg:
        content = '{"issues":[]}'

    class _Choice:
        message = _Msg()

    class _Usage:
        def model_dump(self):
            return {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()
        _hidden_params = {"response_cost": 0.0042}

    async def fake_acompletion(**kw):
        return _Resp()

    import litellm
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        "brainregion.providers.litellm.emit_event",
        lambda event_type, **fields: emitted.append((event_type, fields)),
    )
    reg = {"r": {"provider": "anthropic", "base_url": "https://x", "api_key": "k", "headers": {}, "timeout": None}}
    backend = LiteLLMBackend(endpoint_registry=reg)

    resp = asyncio.run(backend.complete(model="claude-opus-4-8", system="s", user="u", endpoint_id="r"))

    assert resp.cost_usd == 0.0042
    assert [event[0] for event in emitted] == ["model.call_started", "model.call_finished"]
    payload = emitted[1][1]["payload"]
    assert payload["provider"] == "anthropic"
    assert payload["model"] == "claude-opus-4-8"
    assert payload["resolved_model"] == "anthropic/claude-opus-4-8"
    assert payload["canonical_model"] == "claude-opus-4-8"
    assert payload["usage"]["input_tokens"] == 100
    assert payload["usage"]["output_tokens"] == 25
    assert payload["cost_usd"] == 0.0042
    assert payload["cost_source"] == "provider"


def test_backend_multi_endpoint_no_crosstalk(monkeypatch):
    """多 endpoint 并存：各自收到自己的 base/key，不串。"""
    cap = _patch_litellm(monkeypatch)
    reg = {
        "a": {"provider": "openai", "base_url": "https://a/v1", "api_key": "ka", "headers": {}, "timeout": None},
        "b": {"provider": "anthropic", "base_url": "https://b", "api_key": "kb", "headers": {}, "timeout": None},
    }
    backend = LiteLLMBackend(endpoint_registry=reg)
    asyncio.run(backend.complete(model="m1", system="s", user="u", endpoint_id="a"))
    asyncio.run(backend.complete(model="m2", system="s", user="u", endpoint_id="b"))
    assert cap.calls[0]["api_base"] == "https://a/v1" and cap.calls[0]["api_key"] == "ka"
    assert cap.calls[1]["api_base"] == "https://b" and cap.calls[1]["api_key"] == "kb"


# ===== compute_hash 输入不含 api_key =====

def test_hash_panel_has_no_api_key():
    entries = [{"label": "x", "model": "m", "endpoint_id": "r"}]
    serialized = json.dumps(entries, sort_keys=True)
    assert "api_key" not in serialized  # PanelEntry 天然无 key
    h1 = reviews_db.compute_hash(
        document_content="doc", document_files=None, panel=entries,
        dimensions=["d"], adapter="a", project_version={},
        retrieved_cases_ids=[], extra_context="",
    )
    h2 = reviews_db.compute_hash(
        document_content="doc", document_files=None, panel=entries,
        dimensions=["d"], adapter="a", project_version={},
        retrieved_cases_ids=[], extra_context="",
    )
    assert h1 == h2  # 同 entries hash 稳定（dict 可序列化）
