"""Phase 7:git production —— consult provider-loop 测试。

覆盖:MemoryProvider 自 scope(query.regions)/ woken 空集语义 / 构造 scope 优先(eval 回归)/
consult provider-loop 注入 memory+git / per-provider 异常隔离 / 顺序确定 / 门控默认关 /
兼容别名 / _ensure_default_providers idempotent + 部分初始化。
"""
from __future__ import annotations

import pytest

from brainregion.core.context import ContextBlock, ContextQuery, ProviderRegistry, RetrieveResult
from brainregion.memory import MemoryProvider, MemoryScope, store as memory_store
from brainregion.memory.base import ExperienceEvent


@pytest.fixture
def mem_root(monkeypatch, tmp_path):
    """memory_store DB 路径隔离(UNITY_PROJECT_ROOT → tmp_path;同 test_memory_scoping)。"""
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    return tmp_path


# ── MemoryProvider self-scope(Phase 7;unit)─────────────────────────────────────

def test_memory_provider_self_scopes_from_query_regions(mem_root):
    """from_store()(unscoped,registry 式)+ query.regions → memory 自 scope(= 旧 MemoryScope(woken))。"""
    memory_store.record_experience(summary="DEBUG-MARKER", triggers=["x"], region="debugging")
    memory_store.record_experience(summary="SECURITY-MARKER", triggers=["x"], region="security")
    p = MemoryProvider.from_store()                       # unscoped(bootstrap 注册式)
    rr = p.retrieve(ContextQuery(text="x", regions=frozenset({"debugging"}), top_k=5))
    titles = [b.title for b in rr.blocks]
    assert any("DEBUG-MARKER" in t for t in titles)
    assert not any("SECURITY-MARKER" in t for t in titles)   # scoped 到 debugging
    assert rr.meta["scope"] == ["debugging"]


def test_memory_provider_empty_regions_only_global(mem_root):
    """woken 空集 → frozenset() → 只召回全局(行为同旧 MemoryScope(frozenset()));review⑤。"""
    memory_store.record_experience(summary="GLOBAL-MARKER", triggers=["x"], region="")
    memory_store.record_experience(summary="DEBUG-MARKER", triggers=["x"], region="debugging")
    p = MemoryProvider.from_store()
    rr_empty = p.retrieve(ContextQuery(text="x", regions=frozenset(), top_k=5))
    titles = [b.title for b in rr_empty.blocks]
    assert any("GLOBAL-MARKER" in t for t in titles)         # include_global → 全局通过
    assert not any("DEBUG-MARKER" in t for t in titles)      # 无 region 命中 → 只全局
    rr_none = p.retrieve(ContextQuery(text="x", regions=None, top_k=5))  # None=unscoped 全量
    assert any("DEBUG-MARKER" in b.title for b in rr_none.blocks)


def test_construction_scope_wins_over_query_regions():
    """构造 scope(_scope)优先于 query.regions → eval from_records(scope=) 不受影响;review⑥。"""
    evs = [
        ExperienceEvent(id="1", region="debugging", summary="D", triggers=["x"]),
        ExperienceEvent(id="2", region="security", summary="S", triggers=["x"]),
    ]
    p = MemoryProvider.from_records(evs, scope=MemoryScope(frozenset({"security"})))
    rr = p.retrieve(ContextQuery(text="x", regions=frozenset({"debugging"}), top_k=5))
    titles = [b.title for b in rr.blocks]
    assert "S" in titles and "D" not in titles               # 构造 scope(security)赢,query.regions 被忽略


# ── _ensure_default_providers(idempotent + 部分初始化;review④)──────────────────

def test_ensure_default_providers_idempotent_and_partial(monkeypatch):
    """分别 has() 判断:部分初始化(memory 已注 git 未注)只补 git;idempotent 不覆盖 memory 实例。"""
    from brainregion import server
    reg = ProviderRegistry()
    reg.register("memory", MemoryProvider.from_records([]))   # 部分初始化:仅 memory
    monkeypatch.setattr(server, "_default_provider_registry", reg)
    server._ensure_default_providers()
    assert reg.has("memory") and reg.has("git")              # git 补上
    mem_before = reg.get("memory")
    server._ensure_default_providers()                       # 再调:idempotent
    assert reg.get("memory") is mem_before                   # memory 实例不变(未重复注册)


# ── consult provider-loop(integration)──────────────────────────────────────────


class _FakeEngine:
    """捕获 context_blocks;返最小 ConsultReport(不调模型)。"""

    def __init__(self, captured: dict) -> None:
        self._captured = captured

    async def consult(self, *a, **kw):
        self._captured["context_blocks"] = list(kw.get("context_blocks") or [])
        from brainregion.core.consult.report import ConsultAdvice, ConsultReport
        return ConsultReport(
            consultation_id="c", summary="ok",
            individual=[ConsultAdvice(id="c0", model="m", consultant="debugger", summary="ok")],
            usage={"cost_usd": 0.0},
        )


def _fake_git(*, fail: bool = False):
    """确定性 git provider(避免依赖真 repo);fail=True 则 retrieve 抛异常。"""

    class _P:
        def retrieve(self, query: ContextQuery) -> RetrieveResult:
            if fail:
                raise RuntimeError("git boom")
            return RetrieveResult(
                provider="git",
                blocks=[ContextBlock(source="git", title="abc123 GIT-MARKER", content="GIT-MARKER",
                                     framing="data", metadata={"sha": "abc123"})],
                meta={"git_available": True, "commits_found": 1, "returned": 1},
            )

    return _P()


def _wire(monkeypatch, reg, captured):
    """共用:patch registry + FakeEngine + route_regions(debugging)。"""
    from brainregion import server
    monkeypatch.setattr(server, "_default_provider_registry", reg)
    monkeypatch.setattr(server, "_build_consult_engine", lambda dd: _FakeEngine(captured))
    monkeypatch.setattr(server, "_route_regions", lambda **kw: {"selected": [{"id": "debugging"}]})


@pytest.mark.asyncio
async def test_consult_loop_injects_memory_and_git(mem_root, monkeypatch):
    """memory(自 scope)+ git(scopeless,always-inject)都注入;git framing=data;兼容别名。"""
    from brainregion import server
    memory_store.record_experience(summary="MEM-MARKER", triggers=["x"], region="debugging")
    monkeypatch.setenv("BRAIN_REGION_DEFAULT_MEMORY_INJECT", "true")
    reg = ProviderRegistry()
    reg.register("memory", MemoryProvider.from_store())
    reg.register("git", _fake_git())
    captured: dict = {}
    _wire(monkeypatch, reg, captured)
    result = await server.consult_problem(problem="x", mode="debugging")
    assert set(result["context_providers"].keys()) == {"git", "memory"}
    sources = {b.source for b in captured["context_blocks"]}
    assert sources == {"git", "memory"}                       # 两 source 都注入
    assert all(b.framing == "data" for b in captured["context_blocks"] if b.source == "git")  # 防注入
    assert result["memory"]["provider"] == "memory"           # 兼容别名
    assert result["context_providers"]["git"]["git_available"] is True
    # git always-inject:woken={debugging}(非 review)git 仍注入(scopeless)


@pytest.mark.asyncio
async def test_consult_loop_exception_isolation(mem_root, monkeypatch):
    """review①:git 抛异常 → 该 meta available:False + error,memory 照注入,consult 不崩。"""
    from brainregion import server
    memory_store.record_experience(summary="MEM-MARKER", triggers=["x"], region="debugging")
    monkeypatch.setenv("BRAIN_REGION_DEFAULT_MEMORY_INJECT", "true")
    reg = ProviderRegistry()
    reg.register("memory", MemoryProvider.from_store())
    reg.register("git", _fake_git(fail=True))                 # 抛异常
    captured: dict = {}
    _wire(monkeypatch, reg, captured)
    result = await server.consult_problem(problem="x", mode="debugging")
    assert result["context_providers"]["git"]["available"] is False
    assert "error" in result["context_providers"]["git"]
    assert result["context_providers"]["memory"]["provider"] == "memory"  # memory 未被拖垮
    assert any(b.source == "memory" for b in captured["context_blocks"])


@pytest.mark.asyncio
async def test_consult_loop_gate_off_no_injection(mem_root, monkeypatch):
    """review④:memory_inject 默认关 → context_blocks=[] + providers_meta={}(零行为变化,防 NameError)。"""
    from brainregion import server
    monkeypatch.delenv("BRAIN_REGION_DEFAULT_MEMORY_INJECT", raising=False)
    reg = ProviderRegistry()
    reg.register("memory", MemoryProvider.from_store())
    reg.register("git", _fake_git())
    captured: dict = {}
    _wire(monkeypatch, reg, captured)
    result = await server.consult_problem(problem="x", mode="debugging")
    assert captured["context_blocks"] == []
    assert result["context_providers"] == {}                  # 门控关 → 空(init 在 if 外)
    assert result["memory"] == {}                             # 兼容别名也空


@pytest.mark.asyncio
async def test_consult_loop_ordering_sorted(mem_root, monkeypatch):
    """list_names() sorted()(context.py)→ context_providers key 顺序稳定 [git, memory]。"""
    from brainregion import server
    monkeypatch.setenv("BRAIN_REGION_DEFAULT_MEMORY_INJECT", "true")
    reg = ProviderRegistry()
    reg.register("memory", MemoryProvider.from_store())
    reg.register("git", _fake_git())
    captured: dict = {}
    _wire(monkeypatch, reg, captured)
    result = await server.consult_problem(problem="x", mode="debugging")
    assert list(result["context_providers"].keys()) == ["git", "memory"]   # sorted 注入序
