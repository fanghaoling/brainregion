"""MemoryProvider：第一个 ContextProvider 实现（experience 召回 → ContextBlock）。

- ``from_store()`` = 生产（读 SQLite brain_region_reviews.db）。
- ``from_records()`` = eval（纯内存，防伪记忆，不读 DB；roadmap §15.3 🔍）。

retrieve 不调 LLM（§6）；ContextBlock.framing 恒为 "data"（存储型 prompt-injection 防御）。

**region scoping（Phase A）**：``scope=MemoryScope(...)`` 把召回限定到 wake 激活的 region ∪ 全局
（selective context：防跨项目记忆 bleed）。``scope=None``=unscoped 全部（向后兼容）。records/DB 两路
统一用 ``scope.matches()`` 过滤（单一真相源）+ 漏斗 meta（candidates_before_top_k/after_scope/returned）。
旧 ``region`` 单值参数保留（→ MemoryScope({region}, include_global=False)，仅 region 不含全局）。
"""
from __future__ import annotations

from ..core.context import ContextBlock, ContextQuery, RetrieveResult
from . import governance, store
from .base import ExperienceEvent
from .scope import MemoryScope


def _region_to_scope(region: str | None) -> MemoryScope | None:
    """旧单 region 参数 → scope（仅该 region，不含全局，保旧语义 WHERE region=?）。"""
    return MemoryScope(frozenset({region}), include_global=False) if region else None


class MemoryProvider:
    """Experience memory ContextProvider（结构化实现 ContextProvider 协议）。"""

    def __init__(
        self,
        *,
        records: list[ExperienceEvent] | None = None,
        region: str | None = None,
        scope: MemoryScope | None = None,
    ) -> None:
        # records=None → 生产读 DB；records 非空 → eval 纯内存（防伪记忆）。
        self._records = records
        # scope 优先；旧 region 单值兜底（→ 不含全局的单 region scope）。
        self._scope = scope if scope is not None else _region_to_scope(region)

    @classmethod
    def from_store(
        cls,
        region: str | None = None,
        scope: MemoryScope | None = None,
    ) -> "MemoryProvider":
        return cls(region=region, scope=scope)

    @classmethod
    def from_records(
        cls,
        records: list[ExperienceEvent] | None,
        region: str | None = None,
        scope: MemoryScope | None = None,
    ) -> "MemoryProvider":
        return cls(records=list(records or []), region=region, scope=scope)

    def retrieve(self, query: ContextQuery) -> RetrieveResult:
        top_k = max(0, int(query.top_k or 5))
        # scope 优先级:构造 _scope > query.regions(多 region,consult woken) > query.region(单 region,旧) > None。
        # 构造 scope 永远赢 → eval 的 from_records(scope=) 不受 query.regions 影响(review ⑥)。
        scope = self._scope
        if scope is None and getattr(query, "regions", None) is not None:
            scope = MemoryScope(query.regions)           # Phase 7:consult 传 woken 集,memory 自 scope
        elif scope is None and query.region:
            scope = _region_to_scope(query.region)       # 旧:单 region ad-hoc

        if self._records is not None:
            pool_all = self._records
        else:
            pool_all = store.list_experiences()  # 全部（scope=None）；DB 错 → []
        before = len(pool_all)
        pool = pool_all if scope is None else [e for e in pool_all if scope.matches(e.region)]
        after_scope = len(pool)
        # governance 过滤(v6 stage 1):剔 superseded/wrong/expired。调 governance 不内联。
        pool, gov_stats = governance.filter_events(pool)
        hits = store.search_from_records(pool, query.text, top_k)

        blocks = [
            ContextBlock(
                source="memory",
                title=e.summary or e.id,
                content=(e.details or e.summary),
                framing="data",
                metadata={"id": e.id, "region": e.region},
            )
            for e in hits
        ]
        return RetrieveResult(
            provider="memory",
            blocks=blocks,
            meta={
                "candidates_before_top_k": before,
                "candidates_after_scope": after_scope,
                **gov_stats,  # candidates_after_governance + removed_superseded/wrong/expired
                "returned": len(blocks),
                "scope": sorted(scope.regions) if scope is not None else None,
            },
        )
