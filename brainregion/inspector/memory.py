"""inspect_memory：Experience Memory 盘点（read-only）+ Health（v6 stage 1 治理状态）。

按 region 计数 + 总量 + 最近 N 条（带年龄 days）+ 预览;Health: by_status(active/pending/
superseded/wrong) + expired_count + recallable/non_recallable(一眼看 memory 是否腐烂)。
by_region_recallable:每 region 可召回数(viz RegionSnapshot.recallable 用;同 is_recallable 循环,零额外 pass)。
复用 memory_store.list_experiences + governance 谓词。
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..memory import governance, store as memory_store


def inspect_memory(*, region: str | None = None, preview_k: int = 3, manifest: bool = False) -> dict:
    events = memory_store.list_experiences(region=region)  # 新→旧；DB 错 → []
    by_region: dict[str, int] = {}
    by_region_recallable: dict[str, int] = {}  # 每 region 可召回数(viz RegionSnapshot 用)
    by_status: dict[str, int] = {s: 0 for s in (governance.ACTIVE, governance.PENDING,
                                                governance.SUPERSEDED, governance.WRONG)}
    expired_count = 0
    recallable = 0
    for e in events:
        r = e.region or "(global)"
        by_region[r] = by_region.get(r, 0) + 1
        status = getattr(e, "status", governance.ACTIVE) or governance.ACTIVE
        by_status[status] = by_status.get(status, 0) + 1
        if governance.is_expired(getattr(e, "valid_until_ts", 0) or 0):
            expired_count += 1
        if governance.is_recallable(e):
            recallable += 1
            by_region_recallable[r] = by_region_recallable.get(r, 0) + 1
    preview = [_event_summary(e) for e in events[: max(0, int(preview_k))]]
    result = {
        "total": len(events),
        "region_filter": region,
        "by_region": by_region,
        "by_region_recallable": by_region_recallable,
        "health": {
            "by_status": by_status,
            "expired_count": expired_count,
            "recallable": recallable,
            "non_recallable": len(events) - recallable,
        },
        "preview": preview,
    }
    if manifest:
        # 全量记忆清单(Phase 2 Brain Diff 用;复用 events,无二次查询)。默认不开 → live inspect 精简。
        result["manifest"] = [
            {"id": e.id, "region": e.region or "(global)",
             "status": getattr(e, "status", governance.ACTIVE) or governance.ACTIVE,
             "summary": e.summary or "", "triggers": list(e.triggers or []),
             "created_at": e.created_at or ""}
            for e in events
        ]
    return result


def _event_summary(e) -> dict:
    return {
        "id": e.id,
        "region": e.region,
        "summary": e.summary,
        "triggers": list(e.triggers or []),
        "created_at": e.created_at,
        "age_days": _age_days(e.created_at),
        "source": e.source,
        "status": getattr(e, "status", governance.ACTIVE),
        "valid_until_ts": getattr(e, "valid_until_ts", 0) or 0,
        "superseded_by": getattr(e, "superseded_by", ""),
        "last_reviewed": getattr(e, "last_reviewed", ""),
    }


def _age_days(created_at: str | None) -> float | None:
    if not created_at:
        return None
    try:
        dt = datetime.fromisoformat(created_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - dt).total_seconds() / 86400.0, 1)
    except Exception:  # noqa: BLE001
        return None
