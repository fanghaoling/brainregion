"""inspect_activation：调 wake_gate（无模型，read-only）+ 扁平化 debug 视图 + plain-language explain。

wake_gate 已验源码为纯读（models_called=False、无 record_*/memory_store.append、_reverse_wake_hook 是 stub）。
本 view 只重排它的返回 + 写一句「发生了什么/漏了谁」。绝不调模型、绝不写。
"""
from __future__ import annotations

from ..core.regions import REGIONS_DIR, load_regions
from ..core.wake import wake_gate


def inspect_activation(
    *,
    goal: str = "",
    problem: str = "",
    context: str = "",
    files: dict | None = None,
    gold_regions: list[str] | None = None,
    escalate_confidence: float = 0.5,
    shadow_wake_threshold: float | None = None,
    top_k: int = 3,
    sentinel: bool = True,
    shadow_top_n: int = 3,
    regions_dir: str | None = None,
) -> dict:
    """重跑 wake_gate（cheap，无模型）并扁平化。regions_dir=None → 用内置默认。"""
    kw = dict(
        goal=goal, problem=problem, context=context, files=files or {},
        gold_regions=gold_regions, escalate_confidence=escalate_confidence,
        shadow_wake_threshold=shadow_wake_threshold, top_k=top_k, sentinel=sentinel,
        shadow_top_n=shadow_top_n,
    )
    if regions_dir is not None:
        kw["regions_dir"] = regions_dir  # None 会覆盖默认 → 只在非 None 时传
    result = wake_gate(**kw)
    known_region_ids = _known_region_ids(regions_dir)
    return _summarize_activation(result, known_region_ids=known_region_ids)


def _known_region_ids(regions_dir: str | None = None) -> list[str]:
    try:
        return [r.id for r in load_regions(regions_dir or REGIONS_DIR)]
    except Exception:  # noqa: BLE001
        return []


def _summarize_activation(result: dict, *, known_region_ids: list[str] | None = None) -> dict:
    act = result.get("activated_regions") or {}
    metrics = result.get("wake_metrics") or {}
    trace = result.get("trace") or {}

    woken = act.get("woken") or []
    retrieved = [
        {"id": r.get("id"), "score": r.get("score"), "source": r.get("source")}
        for r in (act.get("retrieved") or [])
    ]
    shadow = [
        {"id": s.get("id"), "confidence": s.get("confidence"),
         "promoted": s.get("promoted"), "reason": s.get("reason")}
        for s in (act.get("shadow") or [])
    ]

    missed = metrics.get("missed") or []
    false_wake = metrics.get("false_wake") or []
    scored = metrics.get("metrics_status") == "scored"

    bits: list[str] = []
    if woken:
        bits.append(f"唤醒 {len(woken)} 个 region：{', '.join(woken)}")
    else:
        bits.append("没有唤醒任何 region（输入太短或无 trigger 命中）")
    if scored:
        if missed:
            bits.append(f"⚠️ 漏唤醒 {len(missed)} 个 gold region：{', '.join(missed)}（该醒没醒）")
        else:
            bits.append("✅ gold region 全部命中（无漏唤醒）")
        if false_wake:
            bits.append(f"误唤醒 {len(false_wake)} 个非 gold：{', '.join(false_wake)}")
    else:
        bits.append("未给 gold_regions → unscored（无法判漏唤醒，绝不伪装 0-漏）")
    sp = trace.get("shadow_promoted") or 0
    if sp:
        bits.append(f"shadow fallback 提升 {sp} 个 near-threshold region")
    sh = trace.get("sentinel_hits") or []
    if sh:
        bits.append(f"sentinel 兜底唤醒 {len(sh)} 个：{', '.join(s.get('region', '') for s in sh)}")

    actions = result.get("suggested_actions") or []
    region_matrix = _region_matrix(
        known_region_ids or [],
        activation=act,
        metrics=metrics,
        suggested_actions=actions,
    )
    call_status = {
        "models_called": bool(trace.get("models_called")),
        "retrieved_count": len(retrieved),
        "escalated_count": len(act.get("escalated") or []),
        "woken_count": len(woken),
        "suggested_actions_count": len(actions),
        "requires_user_approval_count": sum(1 for a in actions if a.get("requires_user_approval")),
        "action_tools": [a.get("tool") for a in actions if a.get("tool")],
        "metrics_status": metrics.get("metrics_status"),
        "shadow_promoted": sp,
        "sentinel_hits_count": len(sh),
    }

    return {
        "woken": woken,
        "retrieved": retrieved,
        "escalated": act.get("escalated") or [],
        "shadow": shadow,
        "reasons": act.get("reasons") or {},
        "confidence": act.get("confidence") or {},
        "region_matrix": region_matrix,
        "call_status": call_status,
        "wake_metrics": metrics,
        "trace": {
            "strategy": trace.get("strategy"),
            "escalate_confidence": trace.get("escalate_confidence"),
            "shadow_wake_threshold": trace.get("shadow_wake_threshold"),
            "shadow_promoted": sp,
            "sentinel_hits": sh,
            "models_called": trace.get("models_called"),
            "use_router_api": trace.get("use_router_api"),
            "routing_trace": trace.get("routing_trace") or {},
        },
        "suggested_actions": actions,
        "explain": " | ".join(bits),
    }


def _region_matrix(
    known_region_ids: list[str],
    *,
    activation: dict,
    metrics: dict,
    suggested_actions: list[dict],
) -> list[dict]:
    retrieved = {r.get("id"): r for r in (activation.get("retrieved") or []) if r.get("id")}
    shadow = {s.get("id"): s for s in (activation.get("shadow") or []) if s.get("id")}
    confidence = activation.get("confidence") or {}
    reasons = activation.get("reasons") or {}
    woken = set(activation.get("woken") or [])
    escalated = set(activation.get("escalated") or [])
    missed = set((metrics or {}).get("missed") or [])
    false_wake = set((metrics or {}).get("false_wake") or [])
    action_tools_by_region: dict[str, list[str]] = {}
    for action in suggested_actions:
        tool = action.get("tool")
        if not tool:
            continue
        for region_id in action.get("source_regions") or []:
            action_tools_by_region.setdefault(str(region_id), []).append(str(tool))

    ids = list(dict.fromkeys([*known_region_ids, *retrieved.keys(), *shadow.keys(), *woken, *missed, *false_wake]))
    rows: list[dict] = []
    for region_id in ids:
        score = int(retrieved.get(region_id, {}).get("score") or 0)
        conf = float(confidence.get(region_id) or shadow.get(region_id, {}).get("confidence") or 0.0)
        phase = _region_phase(
            region_id,
            retrieved=region_id in retrieved,
            shadow=shadow.get(region_id),
            woken=woken,
            escalated=escalated,
            missed=missed,
            false_wake=false_wake,
        )
        action_tools = action_tools_by_region.get(region_id, [])
        rows.append(
            {
                "id": region_id,
                "score": score,
                "confidence": round(max(0.0, min(1.0, conf)), 3),
                "phase": phase,
                "source": retrieved.get(region_id, {}).get("source", ""),
                "reason": reasons.get(region_id, ""),
                "suggested_actions": len(action_tools),
                "action_tools": action_tools,
            }
        )
    rows.sort(key=lambda r: (-float(r["confidence"]), -int(r["score"]), str(r["id"])))
    return rows


def _region_phase(
    region_id: str,
    *,
    retrieved: bool,
    shadow: dict | None,
    woken: set[str],
    escalated: set[str],
    missed: set[str],
    false_wake: set[str],
) -> str:
    if region_id in missed:
        return "missed"
    if region_id in false_wake:
        return "false_wake"
    if region_id in woken:
        return "woken"
    if region_id in escalated:
        return "escalated"
    if shadow:
        return "shadow_promoted" if shadow.get("promoted") else "shadow"
    if retrieved:
        return "retrieved"
    return "quiet"
