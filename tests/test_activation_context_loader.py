"""Activation plan -> bounded ContextProvider materialization."""

from __future__ import annotations

from brainregion.core.activation import ActivationSignal
from brainregion.core.context import ContextBlock, ContextQuery, ProviderRegistry, RetrieveResult
from brainregion.core.context_loader import load_activation_context
from brainregion.core.skills import SKILLS_DIR, SkillRegistry, load_skills, setup_resolvers
from brainregion.memory import ExperienceEvent, MemoryProvider


def _skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    for manifest in load_skills(
        SKILLS_DIR,
        region_exists=lambda _region: True,
        provider_exists=lambda _provider: True,
    ):
        registry.register(manifest)
    return registry


def _plan(registry: SkillRegistry, *, max_context_tokens: int = 4000):
    return registry.plan_activation(
        ActivationSignal.from_dict({"events": ["repeated_attempt_failed"]}),
        max_regions=3,
        max_context_tokens=max_context_tokens,
    )


def test_repeated_failure_loads_only_scoped_project_memory():
    records = [
        ExperienceEvent(
            id="debug-memory",
            region="debugging",
            summary="修复重复解析失败",
            details="保留原始响应作为证据锚点，再做宽容 JSON 解析。",
            triggers=["parse_error", "重复失败"],
        ),
        ExperienceEvent(
            id="security-memory",
            region="security",
            summary="无关安全记录",
            details="不应跨脑区装载。",
            triggers=["parse_error"],
        ),
    ]
    providers = ProviderRegistry()
    providers.register("memory", MemoryProvider.from_records(records))
    registry = _skill_registry()

    result = load_activation_context(
        _plan(registry),
        query_text="parse_error 重复失败",
        skill_registry=registry,
        resolvers=setup_resolvers(provider_registry=providers),
    ).to_dict()

    assert set(result["activation"]["woken_regions"]) == {"debugging", "memory"}
    assert [block["metadata"]["id"] for block in result["context_blocks"]] == ["debug-memory"]
    by_skill = {load["skill_id"]: load for load in result["loads"]}
    assert by_skill["memory-recall"]["status"] == "loaded"
    assert by_skill["debugger"]["reason"] == "activation_mode_not_context"
    assert result["trace"]["scope_regions"] == ["debugging", "memory"]
    assert result["trace"]["models_called"] is False
    assert result["trace"]["retained_by_runtime"] is False


def test_context_loader_enforces_request_token_budget_and_marks_truncation():
    records = [
        ExperienceEvent(
            id="long",
            region="debugging",
            summary="短标题",
            details="细" * 200,
            triggers=["overflow"],
        )
    ]
    providers = ProviderRegistry()
    providers.register("memory", MemoryProvider.from_records(records))
    registry = _skill_registry()

    result = load_activation_context(
        _plan(registry, max_context_tokens=24),
        query_text="overflow",
        skill_registry=registry,
        resolvers=setup_resolvers(provider_registry=providers),
        scope_regions=frozenset({"debugging"}),
    ).to_dict()

    load = next(item for item in result["loads"] if item["skill_id"] == "memory-recall")
    assert load["estimated_tokens"] <= load["requested_tokens"] == 24
    assert load["truncated"] is True
    assert result["context_blocks"][0]["metadata"]["activation_truncated"] is True


class _BrokenProvider:
    def retrieve(self, _query: ContextQuery) -> RetrieveResult:
        raise RuntimeError("provider offline")


def test_context_loader_isolates_provider_failure():
    providers = ProviderRegistry()
    providers.register("memory", _BrokenProvider())
    registry = _skill_registry()

    result = load_activation_context(
        _plan(registry),
        query_text="parse_error",
        skill_registry=registry,
        resolvers=setup_resolvers(provider_registry=providers),
    ).to_dict()

    memory_load = next(item for item in result["loads"] if item["skill_id"] == "memory-recall")
    assert memory_load["status"] == "failed"
    assert "provider offline" in memory_load["reason"]
    assert result["context_blocks"] == []


def test_mcp_load_region_context_uses_activation_and_provider_registry(monkeypatch):
    from brainregion import server

    providers = ProviderRegistry()
    providers.register(
        "memory",
        MemoryProvider.from_records(
            [
                ExperienceEvent(
                    id="mcp-memory",
                    region="debugging",
                    summary="MCP 理解卡片",
                    details="反复失败时先查历史尝试。",
                    triggers=["repeated parse"],
                )
            ]
        ),
    )
    monkeypatch.setattr(server, "_default_provider_registry", providers)
    monkeypatch.setattr(server, "_skill_registry_singleton", None)

    result = server.load_region_context(
        query="repeated parse",
        events=["repeated_attempt_failed"],
        max_context_tokens=2000,
    )

    assert result["context_blocks"][0]["metadata"]["id"] == "mcp-memory"
    assert result["trace"]["providers_called"] == ["memory"]
    assert result["trace"]["models_called"] is False
    assert result["activation"]["trace"]["models_called"] is False


def test_context_query_carries_selectors_across_provider_boundary():
    seen: list[tuple[str, ...]] = []

    class _SelectorProvider:
        def retrieve(self, query: ContextQuery) -> RetrieveResult:
            seen.append(query.selectors)
            return RetrieveResult(
                provider="memory",
                blocks=[ContextBlock(source="memory", title="t", content="c")],
            )

    providers = ProviderRegistry()
    providers.register("memory", _SelectorProvider())
    registry = _skill_registry()
    load_activation_context(
        _plan(registry),
        query_text="context_missing",
        skill_registry=registry,
        resolvers=setup_resolvers(provider_registry=providers),
    )

    assert seen == [
        (
            "project_understanding",
            "decisions",
            "constraints",
            "failure_lessons",
            "evidence_anchors",
        )
    ]
