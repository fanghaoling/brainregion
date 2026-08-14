"""Inspector facade：``inspect(view, ...)`` → 按 view 过滤返回 section。

统一入口（吸收 GPT：1 MCP 工具 + 1 CLI 子命令 都走这里）。

- ``view`` 白名单（Literal 校验）：all / activation / memory / run / calibration / model_health。
  未知 → ValueError（review #5，防 **kw 透传未过滤参数）。
- ``view∉{all,activation}`` 时**绝不调 wake_gate**（review #1/#13）—— activation 是唯一参数化、唯一触 wake 的 view。
- run 无 run_id → 最近 N run 历史表（bounded by history_limit，非全表扫——review #13 + GPT 二-三）。
- 只含请求的 section（``all`` 含全部 5 个）。
"""
from __future__ import annotations

from . import activation, calibration, memory, model_health, run

VIEWS = ("all", "activation", "memory", "run", "calibration", "model_health")


def inspect(
    *,
    view: str = "all",
    goal: str = "",
    problem: str = "",
    context: str = "",
    files: dict | None = None,
    gold_regions: list[str] | None = None,
    run_id: str | None = None,
    region: str | None = None,
    judge_id: str | None = None,
    model_key: str | None = None,
    escalate_confidence: float = 0.5,
    shadow_wake_threshold: float | None = None,
    top_k: int = 3,
    memory_preview_k: int = 3,
    memory_manifest: bool = False,
    history_limit: int = 20,
) -> dict:
    if view not in VIEWS:
        raise ValueError(f"unknown view: {view!r}; expected one of {VIEWS}")
    out: dict = {}
    if view in ("all", "activation"):
        out["activation"] = activation.inspect_activation(
            goal=goal, problem=problem, context=context, files=files,
            gold_regions=gold_regions, escalate_confidence=escalate_confidence,
            shadow_wake_threshold=shadow_wake_threshold, top_k=top_k,
        )
    if view in ("all", "memory"):
        out["memory"] = memory.inspect_memory(region=region, preview_k=memory_preview_k,
                                              manifest=memory_manifest)
    if view in ("all", "run"):
        out["run"] = run.inspect_run(run_id=run_id, history_limit=history_limit)
    if view in ("all", "calibration"):
        out["calibration"] = calibration.inspect_calibration(judge_id=judge_id)
    if view in ("all", "model_health"):
        out["model_health"] = model_health.inspect_model_health(
            model_key=model_key, limit=history_limit
        )
    return out
