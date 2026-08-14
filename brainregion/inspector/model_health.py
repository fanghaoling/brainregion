"""model_health 视图:指纹/能力探针的基线与运行历史(纯只读,读 probe.db)。

给 inspect(view="model_health") 用:一眼看到各 model_key 的现存基线、最近探针判定、
按判定聚合的Counts——漂移趋势的可观测面。
"""
from __future__ import annotations

from collections import Counter

from ..probe import storage


def inspect_model_health(model_key: str | None = None, limit: int = 20) -> dict:
    runs = storage.recent_runs(model_key=model_key, limit=limit)
    baselines = storage.list_baselines(active_only=True)
    if model_key:
        baselines = [b for b in baselines if b["model_key"] == model_key]
    latest: dict[str, dict] = {}
    for r in reversed(runs):  # 倒序列表倒着覆盖 → 留每个 model_key 最新一次
        latest[r["model_key"]] = {
            "kind": r["kind"],
            "verdict": r["verdict"],
            "score": r["score"],
            "created_at": r["created_at"],
        }
    return {
        "active_baselines": baselines,
        "recent_runs": runs,
        "latest_verdict_by_model": latest,
        "verdict_counts": dict(Counter(r["verdict"] or "unknown" for r in runs)),
    }
