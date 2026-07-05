"""Phase 4:Skill/Region Manifest + Registry 测试(三级结构 Skill 层地基)。

覆盖 review 矩阵:SkillManifest sanitized、loader fail-fast、SkillRegistry(id 冲突 raise/manifests_for_router
屏蔽 ref)、ProviderRegistry(register/get/warn-dup)、Resolver(provider live + UnknownProvider +
UnsupportedSkillKind)、list_skills MCP sanitized、drift(bootstrap 注册 MemoryProvider 同源)。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from brainregion.core.context import (
    ContextBlock,
    ContextQuery,
    ProviderRegistry,
    RetrieveResult,
    default_provider_registry,
)
from brainregion.core.skills import (
    SKILLS_DIR,
    SkillManifest,
    SkillRegistry,
    UnsupportedSkillKind,
    UnknownProvider,
    load_skill,
    load_skills,
    resolve_skill_body,
    setup_resolvers,
)
from brainregion.server import list_skills as list_skills_mcp


# ── helpers ───────────────────────────────────────────────────────────────────

class _FakeProvider:
    """实现 ContextProvider retrieve(返带 marker 的 RetrieveResult;用于验 Resolver 真调 retrieve)。"""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def retrieve(self, query: ContextQuery) -> RetrieveResult:
        return RetrieveResult(
            provider=self.marker,
            blocks=[ContextBlock(source=self.marker, title="t", content="c")],
            meta={"query_text_len": len(query.text)},
        )


def _mf(i: int, **kw) -> SkillManifest:
    base: dict = dict(id=f"s{i}", name=f"S{i}", region="memory", kind="provider", ref="memory")
    base.update(kw)
    return SkillManifest(**base)


def _write_yaml(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / f"{name}.yaml"
    p.write_text(body, encoding="utf-8")
    return p


# ── SkillManifest ──────────────────────────────────────────────────────────────

def test_manifest_to_public_dict_hides_ref_includes_metadata():
    m = SkillManifest(id="x", name="X", region="memory", kind="provider", description="d",
                      tags=("a", "b"), ref="memory", status="experimental", metadata={"k": "v"})
    d = m.to_public_dict()
    assert "ref" not in d                                      # 屏蔽 body 引用
    assert d["id"] == "x" and d["kind"] == "provider" and d["tags"] == ["a", "b"]
    assert d["metadata"] == {"k": "v"} and d["status"] == "experimental"


# ── SkillRegistry ──────────────────────────────────────────────────────────────

def test_registry_basic_and_id_collision_raise():
    r = SkillRegistry()
    r.register(_mf(1))
    assert r.get("s1") is not None and r.has("s1") and not r.has("nope")
    assert [m.id for m in r.by_region("memory")] == ["s1"]
    with pytest.raises(ValueError, match="duplicate skill id"):
        r.register(_mf(1))                                     # 撞 id raise(不静默覆盖)


def test_registry_manifests_for_router_hides_ref_and_filters_region():
    r = SkillRegistry()
    r.register(_mf(1, region="memory"))
    r.register(_mf(2, region="debugging"))
    out = r.manifests_for_router()
    assert len(out) == 2 and all("ref" not in d for d in out)
    assert [d["id"] for d in r.manifests_for_router(regions=["memory"])] == ["s1"]


# ── ProviderRegistry ───────────────────────────────────────────────────────────

def test_provider_registry_register_get_unknown_returns_none():
    r = ProviderRegistry()
    p = _FakeProvider("memory")
    r.register("memory", p)
    assert r.get("memory") is p and r.has("memory") and "memory" in r.list_names()
    assert r.get("unknown") is None                            # 未知名返 None(warn-only 不 raise)


def test_provider_registry_dup_register_warns_and_overwrites(caplog):
    r = ProviderRegistry()
    r.register("memory", _FakeProvider("a"))
    with caplog.at_level("WARNING", logger="brainregion.core.context"):
        r.register("memory", _FakeProvider("b"))
    assert r.get("memory").marker == "b"                       # 覆盖
    assert any("覆盖" in rec.message for rec in caplog.records)  # warn


# ── loader fail-fast(review #3)─────────────────────────────────────────────────

def test_loader_loads_seeded_memory_recall():
    m = load_skill("memory-recall", SKILLS_DIR,
                   region_exists=lambda r: r == "memory", provider_exists=lambda n: n == "memory")
    assert m.id == "memory-recall" and m.kind == "provider" and m.ref == "memory"
    assert m.region == "memory" and m.status == "experimental"


def test_loader_filters_yaml_null_tags(tmp_path):
    _write_yaml(
        tmp_path,
        "debugger",
        "id: debugger\nname: Debugger\nregion: debugging\nkind: consultant\ntags: [bug, null, fix]\n",
    )

    m = load_skill("debugger", tmp_path, region_exists=lambda r: r == "debugging")

    assert m.tags == ("bug", "fix")


def test_loader_missing_required_field(tmp_path):
    _write_yaml(tmp_path, "bad", "id: bad\nname: Bad\nregion: memory\n")  # 缺 kind
    with pytest.raises(ValueError, match="required field 'kind'"):
        load_skill("bad", tmp_path)


def test_loader_unknown_kind(tmp_path):
    _write_yaml(tmp_path, "bad", "id: bad\nname: Bad\nregion: memory\nkind: bogus\nref: memory\n")
    with pytest.raises(ValueError, match="kind 'bogus'"):
        load_skill("bad", tmp_path, provider_exists=lambda n: True)


def test_loader_unknown_region(tmp_path):
    _write_yaml(tmp_path, "bad", "id: bad\nname: Bad\nregion: ghost\nkind: provider\nref: memory\n")
    with pytest.raises(ValueError, match="region 'ghost' not in regions"):
        load_skill("bad", tmp_path, region_exists=lambda r: r == "memory",
                   provider_exists=lambda n: True)


def test_loader_unknown_provider_ref(tmp_path):
    _write_yaml(tmp_path, "bad", "id: bad\nname: Bad\nregion: memory\nkind: provider\nref: ghost\n")
    with pytest.raises(ValueError, match="provider 'ghost' not registered"):
        load_skill("bad", tmp_path, region_exists=lambda r: r == "memory",
                   provider_exists=lambda n: n == "memory")


def test_loader_duplicate_id(tmp_path):
    _write_yaml(tmp_path, "a", "id: dup\nname: A\nregion: memory\nkind: provider\nref: memory\n")
    with pytest.raises(ValueError, match="duplicate skill id 'dup'"):
        load_skill("a", tmp_path, region_exists=lambda r: r == "memory",
                   provider_exists=lambda n: True, known_ids={"dup"})


def test_loader_unknown_field_strict(tmp_path):
    _write_yaml(tmp_path, "bad",
                "id: bad\nname: Bad\nregion: memory\nkind: provider\nref: memory\nbogus_field: x\n")
    with pytest.raises(ValueError, match="unknown field"):
        load_skill("bad", tmp_path, region_exists=lambda r: r == "memory",
                   provider_exists=lambda n: True)


def test_loader_load_skills_dir_unique_ids(tmp_path):
    _write_yaml(tmp_path, "a", "id: x\nname: A\nregion: memory\nkind: provider\nref: memory\n")
    _write_yaml(tmp_path, "b", "id: x\nname: B\nregion: memory\nkind: provider\nref: memory\n")
    with pytest.raises(ValueError, match="duplicate skill id 'x'"):
        load_skills(tmp_path, region_exists=lambda r: r == "memory", provider_exists=lambda n: True)


# ── Resolver(真 resolve,非自证循环)──────────────────────────────────────────

def test_resolve_skill_body_provider_kind_live():
    pr = ProviderRegistry()
    pr.register("memory", _FakeProvider("memory-marker"))
    resolvers = setup_resolvers(provider_registry=pr)
    m = SkillManifest(id="m", name="M", region="memory", kind="provider", ref="memory")
    res = resolve_skill_body(m, ContextQuery(text="hello"), resolvers=resolvers)
    assert res.provider == "memory-marker" and res.blocks            # 真调 retrieve(非 mock-only)


def test_resolve_skill_body_unknown_provider():
    pr = ProviderRegistry()                                    # 空 → ref 未注册
    resolvers = setup_resolvers(provider_registry=pr)
    m = SkillManifest(id="m", name="M", region="memory", kind="provider", ref="ghost")
    with pytest.raises(UnknownProvider) as ei:
        resolve_skill_body(m, ContextQuery(text="x"), resolvers=resolvers)
    assert ei.value.ref == "ghost" and ei.value.available == []   # 空 registry → available=[]


def test_resolve_skill_body_unsupported_kind():
    pr = ProviderRegistry()
    resolvers = setup_resolvers(provider_registry=pr)          # 仅 provider resolver
    m = SkillManifest(id="c", name="C", region="debugging", kind="consultant", ref="debugger")
    with pytest.raises(UnsupportedSkillKind) as ei:
        resolve_skill_body(m, ContextQuery(text="x"), resolvers=resolvers)
    assert ei.value.kind == "consultant" and ei.value.skill_id == "c"


# ── list_skills MCP + drift ────────────────────────────────────────────────────

def test_list_skills_mcp_sanitized_no_ref():
    out = list_skills_mcp()
    assert out["count"] >= 1
    assert any(s["id"] == "memory-recall" for s in out["skills"])
    assert all("ref" not in s for s in out["skills"])          # MCP 输出不泄 ref


def test_list_skills_mcp_has_no_none_tags():
    out = list_skills_mcp()

    assert all("None" not in s["tags"] for s in out["skills"])


def test_debugger_skill_tags_include_nullref_not_none():
    out = list_skills_mcp(region="debugging")
    debugger = next(s for s in out["skills"] if s["id"] == "debugger")

    assert "nullref" in debugger["tags"]
    assert "None" not in debugger["tags"]


def test_list_skills_mcp_by_region():
    out = list_skills_mcp(region="memory")
    assert all(s["region"] == "memory" for s in out["skills"])
    assert any(s["id"] == "memory-recall" for s in out["skills"])


def test_drift_bootstrap_registers_memory_provider_same_source():
    """drift 检测:registry 的 'memory' provider = production wiring 用的 MemoryProvider(同源,不发散)。"""
    from brainregion.memory import MemoryProvider
    from brainregion.server import _skill_registry
    _skill_registry()                                          # 触发 bootstrap(register MemoryProvider)
    provider = default_provider_registry.get("memory")
    assert isinstance(provider, MemoryProvider)                # 与 server.py:1366 内联 from_store 同类
