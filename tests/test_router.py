"""Phase 5:Router API + wake_gate 接入 测试。

覆盖:KeywordRouter 等价 route_regions(回归,wake_gate flag 开=安全前提)、RouterResult shape lock、
ManifestRouter(reference baseline;有/空 corpus)、compare_routers(Comparator;空集安全)、
manifest-only seed 隔离(resolve raise)、wake_gate USE_ROUTER_API flag(off=现状/on=KeywordRouter 等价)。
"""
from __future__ import annotations

import pytest

from brainregion.core.regions import load_regions, route_regions
from brainregion.core.regions.router import (
    KeywordRouter,
    ManifestRouter,
    RouterResult,
    compare_routers,
    route,
    use_router_api,
)
from brainregion.core.skills import SkillRegistry, UnsupportedSkillKind, resolve_skill_body, setup_resolvers
from brainregion.core.context import ContextQuery, ProviderRegistry

QUERY = dict(goal="debug a null reference crash", problem="", context="", files=None, top_k=3, min_score=2)
REGIONS = load_regions()


# ── KeywordRouter 等价回归(review consensus:wake_gate drop-in 前提)─────────────

def test_keyword_router_equivalent_to_route_regions():
    """KeywordRouter.route() == route_regions()(除 +router 字段)。"""
    direct = route_regions(regions=REGIONS, **QUERY)
    kwr = KeywordRouter(REGIONS).route(**QUERY)
    assert kwr["router"] == "keyword"
    assert kwr["selected"] == direct["selected"]            # 选区一致(含 score/排序)
    assert kwr["candidates"] == direct["candidates"]
    assert kwr["trace"] == direct["trace"]
    assert "router" not in direct and kwr["router"] == "keyword"  # 仅 KeywordRouter 多 router 字段


def test_router_result_shape_lock():
    """RouterResult 字段集合稳定(selected/candidates/trace/router)。"""
    kwr = KeywordRouter(REGIONS).route(**QUERY)
    assert set(kwr.keys()) >= {"selected", "candidates", "trace", "router"}
    assert isinstance(kwr, RouterResult)


# ── ManifestRouter(reference deterministic baseline)──────────────────────────────

def _seeded_corpus() -> list[tuple[str, str]]:
    """从 bootstrap SkillRegistry 建 ManifestRouter corpus(region → 聚合 skill 描述+tags)。"""
    from brainregion.server import _skill_registry
    reg = _skill_registry()
    by_region: dict[str, list[str]] = {}
    for m in reg.manifests_for_router():
        by_region.setdefault(m["region"], []).append(m["description"] + " " + " ".join(m["tags"]))
    return [(rid, " ".join(parts)) for rid, parts in by_region.items()]


def test_manifest_router_routes_by_skill_corpus():
    mr = ManifestRouter(_seeded_corpus())
    # debug 查询 → debugging region(debugger skill 的 bug/crash/null 术语命中)
    out = mr.route(goal="debug a null reference crash and stacktrace", top_k=3, min_score=2)
    assert out["router"] == "manifest"
    sel_ids = [s["id"] for s in out["selected"]]
    assert "debugging" in sel_ids


def test_manifest_router_empty_corpus_marker():
    mr = ManifestRouter([])                                  # 空 corpus → 空选 + no_corpus marker
    out = mr.route(goal="anything", top_k=3)
    assert out["selected"] == [] and out["trace"]["no_corpus"] is True


# ── compare_routers(Comparator;空集安全)─────────────────────────────────────────

def test_compare_routers_both_empty_is_safe():
    a = KeywordRouter(REGIONS)                               # 不命中查询 → 可能空选
    b = ManifestRouter([])
    out = compare_routers(a, b, goal="zzz_no_match_zzz", top_k=3, min_score=2)
    assert out["agreement"]["jaccard_selected"] is None     # 双空(或一方空)→ 不除零
    assert out["agreement"]["same_top"] is None             # 空 top → None(不索引)


def test_compare_routers_agreement_values():
    a = KeywordRouter(REGIONS)
    b = ManifestRouter(_seeded_corpus())
    out = compare_routers(a, b, goal="debug a null reference crash and stacktrace", top_k=3, min_score=2)
    sa, sb = set(out["a"]["selected"]), set(out["b"]["selected"])
    union = sa | sb
    expected_j = None if not union else round(len(sa & sb) / len(union), 4)
    assert out["agreement"]["jaccard_selected"] == expected_j
    assert set(out["agreement"]["per_region_disagree"]) == (union - (sa & sb))


def test_route_entry_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        route(strategy="bogus", regions=REGIONS, **QUERY)


# ── manifest-only seed 隔离(review:不暴露为可调用 skill)─────────────────────────

def test_consultant_seeds_unresolvable_but_listable():
    from brainregion.server import list_skills as list_skills_mcp
    out = list_skills_mcp()
    ids = {s["id"] for s in out["skills"]}
    assert {"debugger", "security-audit", "perf-analysis", "review-quality"} <= ids  # 进 registry(drive ManifestRouter)
    assert all(s["status"] == "experimental" for s in out["skills"])                 # 标 experimental
    # 但 resolve 它们 → UnsupportedSkillKind(body 未接通)
    reg = _seeded_registry()
    resolvers = setup_resolvers(provider_registry=ProviderRegistry())
    for sid in ("debugger", "security-audit"):
        m = reg.get(sid)
        with pytest.raises(UnsupportedSkillKind):
            resolve_skill_body(m, ContextQuery(text="x"), resolvers=resolvers)


def _seeded_registry() -> SkillRegistry:
    from brainregion.server import _skill_registry
    return _skill_registry()


# ── wake_gate USE_ROUTER_API flag(真 seam)──────────────────────────────────────

def test_wake_gate_flag_off_is_current_behavior(monkeypatch):
    monkeypatch.delenv("BRAIN_REGION_USE_ROUTER_API", raising=False)
    assert use_router_api() is False
    from brainregion.core.wake import wake_gate
    out = wake_gate(goal="debug a null reference crash", gold_regions=["debugging"])
    assert out["trace"]["use_router_api"] is False
    assert "debugging" in out["activated_regions"]["woken"]   # 现有行为不破


def test_wake_gate_flag_on_keyword_router_path(monkeypatch):
    """flag 开 → KeywordRouter seam;selected 等价 off 路径(等价回归保证)。"""
    from brainregion.core.wake import wake_gate
    monkeypatch.setenv("BRAIN_REGION_USE_ROUTER_API", "1")
    assert use_router_api() is True
    out_on = wake_gate(goal="debug a null reference crash", gold_regions=["debugging"])
    assert out_on["trace"]["use_router_api"] is True
    monkeypatch.delenv("BRAIN_REGION_USE_ROUTER_API", raising=False)
    out_off = wake_gate(goal="debug a null reference crash", gold_regions=["debugging"])
    # KeywordRouter == route_regions → woken 一致
    assert set(out_on["activated_regions"]["woken"]) == set(out_off["activated_regions"]["woken"])
