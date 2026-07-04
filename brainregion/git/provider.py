"""GitProvider:第二个 ContextProvider(git 历史 → ContextBlock;Phase 6;镜像 MemoryProvider)。

- ``from_repo()`` = 生产(惰性跑 git log;注册实例零成本,git 在 retrieve 时才跑)。
- ``from_events()`` = eval/test(纯内存,防伪,不跑子进程;镜像 MemoryProvider.from_records)。

retrieve 不调 LLM(§6);ContextBlock.framing 恒 ``"data"``(git 输出=外部不可信数据;commit message
可含注入性内容)。region 忽略:git 历史=repo 全局,非 region 分区;skill 归 review 仅管路由唤醒。
GitEvent 是 git-package 内部,跨 ContextProvider 边界的只有 ContextBlock。
"""

from __future__ import annotations

from ..core.context import ContextBlock, ContextQuery, RetrieveResult
from .base import GitEvent
from .store import GitStore, search_commits


class GitProvider:
    """Git 历史 ContextProvider(结构化实现 ContextProvider 协议)。"""

    def __init__(self, *, events: list[GitEvent] | None = None, store: GitStore | None = None) -> None:
        # events 非 None → eval/test 纯内存;None → 生产走 store(惰性 git log)。
        self._events = events
        self._store = store

    @classmethod
    def from_repo(cls, **store_kw) -> "GitProvider":
        """生产:cwd 默认 '.'(= MCP 调用方的当前 repo)。注册实例零成本(git 在 retrieve 时才跑)。"""
        return cls(store=GitStore(**store_kw))

    @classmethod
    def from_events(cls, events: list[GitEvent] | None) -> "GitProvider":
        """eval/test:纯内存事件,不跑子进程(防伪,镜像 MemoryProvider.from_records)。"""
        return cls(events=list(events or []))

    def retrieve(self, query: ContextQuery) -> RetrieveResult:
        top_k = max(0, int(query.top_k or 5))
        if self._events is not None:
            events: list[GitEvent] = self._events
            store_meta: dict = {"source": "injected"}
        else:
            events, store_meta = self._store.list_commits()
        hits = search_commits(events, query.text, top_k)
        blocks = [
            ContextBlock(
                source="git",
                title=f"{e.sha[:7]} {e.subject}".strip(),
                content=(f"{e.subject}\nfiles: {', '.join(e.files)}" if e.files else e.subject),
                framing="data",
                metadata={"sha": e.sha, "author": e.author, "date": e.date, "files": list(e.files)},
            )
            for e in hits
        ]
        return RetrieveResult(
            provider="git",
            blocks=blocks,
            meta={
                "candidates_before_top_k": len(events),
                "returned": len(blocks),
                **store_meta,
            },
        )
