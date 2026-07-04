"""Wake gate — region-routing 的 escalate + 假阴性兜底层（sidecar，只读，不调模型）。

建立在 route_regions（retrieve 层）之上：
  retrieve（route_regions，确定性，不调模型）
    → escalate（confidence >= escalate_confidence）
    → wake（+ shadow 提升 + sentinel 兜底）
    → activation trace（对齐 docs/eval_harness.zh-CN.md §3.3）+ wake_metrics（对 gold_regions）。

三层用不同判据：retrieve=score/min_score，escalate=confidence——故可严格分离（吸收 review_plan 审核②：
escalate 不能等于 retrieve 的 min_score 否则塌缩成恒等变换）。

假阴性兜底双路径（吸收审核③⑥：不能只剩 sentinel 单点）：
- sentinel：跨域风险词，region 化（每 region 自带 sentinel_keywords，无则用内置 fallback）+ registry
  校验（非法 id 只记 sentinel_hits 不进 woken，吸收审核①：concurrency/data 无对应 region 时不当 woken）。
- shadow 提升：retrieved-but-not-escalated 且 confidence >= shadow_wake_threshold 的候选真唤醒。

missed-wake 是硬门槛：无 gold 时 metrics_status="unscored"，绝不让 missed:[] 伪装成"0 漏唤醒"（吸收审核④）。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..regions import REGIONS_DIR, RegionDefinition, load_regions, route_regions
from ..regions.loader import _normalize
from ..regions.router import KeywordRouter, use_router_api
from ..workflow import _build_actions

# 内置 sentinel fallback（region yaml 无 sentinel_keywords 时用）。key 必须是已加载的 region id，
# 否则不生效——避免唤醒不存在的 region（吸收审核①）。含中文同义词（吸收审核⑤）。
_DEFAULT_SENTINEL_KEYWORDS: dict[str, list[str]] = {
    "security": [
        "注入", "injection", "sqli", "xss", "csrf", "auth", "secret",
        "密钥", "password", "passwd", "token", "越权", "privilege",
    ],
}

# shadow_wake_threshold 默认 = escalate_confidence - gap（保证 < escalate，留出 promotion 带）。
_DEFAULT_SHADOW_GAP = 0.15
_SENTINEL_CONFIDENCE = 0.3  # sentinel 兜底唤醒的置信度（低，标记为兜底）


def _check_confidence(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return v


def _check_int(value: Any, name: str, *, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}]")
    return value


def _sentinel_keywords_by_region(regions, *, enabled: bool) -> dict[str, list[str]]:
    """region 自带 sentinel_keywords 优先；无则用内置 fallback（仅已知 region）。"""
    if not enabled:
        return {}
    out: dict[str, list[str]] = {}
    for region in regions:
        kws = list(region.sentinel_keywords) or list(_DEFAULT_SENTINEL_KEYWORDS.get(region.id, []))
        if kws:
            out[region.id] = kws
    return out


def _source_of(candidate: dict) -> str:
    sources = [m.get("source") for m in candidate.get("matched_triggers", [])]
    if "text" in sources:
        return "text"
    if "files" in sources:
        return "files"
    return "keyword"


def _reverse_wake_hook(reason: str) -> dict:
    """Stub：合成阶段发现高风险/低置信时反向补审的 hook 点。MVP 不 re-trigger（gate 后实现）。"""
    return {"triggered": False, "reason": reason}


def wake_gate(
    *,
    goal: str = "",
    problem: str = "",
    context: str = "",
    files: dict[str, str] | None = None,
    escalate_confidence: float = 0.5,
    shadow_wake_threshold: float | None = None,
    top_k: int = 3,
    sentinel: bool = True,
    shadow_top_n: int = 3,
    gold_regions: list[str] | None = None,
    regions: list[RegionDefinition] | None = None,
    regions_dir: str | Path = REGIONS_DIR,
) -> dict:
    """Region-routing wake gate (read-only sidecar; never calls models).

    Returns:
      activated_regions: {retrieved, escalated, woken, shadow, reasons, confidence}
      wake_metrics: {hit, false_wake, missed} + metrics_status ("scored" iff gold given)
      suggested_actions: workflow actions built from the woken set
      trace: strategy + thresholds + sentinel_hits + shadow_promoted + flags
    """
    escalate_confidence = _check_confidence(escalate_confidence, "escalate_confidence")
    if shadow_wake_threshold is not None:
        shadow_wake_threshold = _check_confidence(shadow_wake_threshold, "shadow_wake_threshold")
    else:
        shadow_wake_threshold = max(0.0, escalate_confidence - _DEFAULT_SHADOW_GAP)
    _check_int(top_k, "top_k", lo=1, hi=20)
    _check_int(shadow_top_n, "shadow_top_n", lo=0, hi=20)

    files = files or {}
    gold = [str(g) for g in (gold_regions or [])]

    regions = regions if regions is not None else load_regions(regions_dir)
    region_ids = {r.id for r in regions}
    router_api = use_router_api()
    if router_api:
        # USE_ROUTER_API seam:关=route_regions 现状;开=KeywordRouter(==route_regions,等价回归保证)。
        # 以后换 manifest/LLM router 改这一处(全 runtime 换)。GPT round-2:无消费者的 Router API 无意义。
        routing = KeywordRouter(regions).route(
            goal=goal, problem=problem, context=context, files=files, top_k=top_k)
    else:
        routing = route_regions(
            goal=goal,
            problem=problem,
            context=context,
            files=files,
            top_k=top_k,
            regions=regions,
            regions_dir=regions_dir,
        )
    candidates = list(routing.get("candidates", []))

    # --- retrieve（不调模型；route_regions 已做 negative_triggers 降噪）---
    retrieved: list[dict] = []
    confidence_by_id: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for cand in candidates:
        rid = cand["id"]
        retrieved.append({"id": rid, "score": cand["score"], "source": _source_of(cand)})
        confidence_by_id[rid] = float(cand.get("confidence", 0.0))
        joined = "; ".join(cand.get("reasons", []))
        reasons[rid] = joined or "matched"

    # --- escalate（confidence 判据，≠ retrieve 的 score/min_score）---
    escalated: list[str] = [c["id"] for c in candidates if confidence_by_id[c["id"]] >= escalate_confidence]
    escalated_set = set(escalated)

    woken: list[str] = list(escalated)
    woken_set = set(escalated_set)
    shadow_records: list[dict] = []
    shadow_promoted = 0

    # --- shadow → fallback-wake（真唤醒）：retrieved-but-not-escalated 且 conf >= shadow_wake_threshold ---
    for cand in candidates:
        rid = cand["id"]
        if rid in escalated_set:
            continue
        conf = confidence_by_id[rid]
        promoted = conf >= shadow_wake_threshold
        shadow_records.append(
            {
                "id": rid,
                "score": cand["score"],
                "confidence": round(conf, 3),
                "reason": "shadow fallback" if promoted else "below shadow threshold",
                "promoted": promoted,
            }
        )
        if promoted and rid not in woken_set:
            woken.append(rid)
            woken_set.add(rid)
            shadow_promoted += 1
    promoted_records = [s for s in shadow_records if s["promoted"]]
    observed = [s for s in shadow_records if not s["promoted"]][:shadow_top_n]
    shadow_out = promoted_records + observed

    # --- sentinel（region 化 + registry 校验；含中文）---
    sentinel_hits: list[dict] = []
    text = _normalize("\n".join(p for p in (goal, problem, context) if p))
    sent_map = _sentinel_keywords_by_region(regions, enabled=sentinel)
    for rid, kws in sent_map.items():
        if rid in woken_set or rid not in region_ids:
            continue
        hit_kw = next((kw for kw in kws if _normalize(kw) in text), None)
        if hit_kw is not None:
            sentinel_hits.append({"region": rid, "keyword": hit_kw})
            woken.append(rid)
            woken_set.add(rid)
            confidence_by_id[rid] = _SENTINEL_CONFIDENCE
            reasons[rid] = f"sentinel fallback: {hit_kw}"

    # --- wake_metrics（对 gold_regions；无 gold → unscored，绝不伪装 0-漏）---
    gold_set = set(gold)
    if gold:
        wake_metrics = {
            "hit": sorted(gold_set & woken_set),
            "false_wake": sorted(woken_set - gold_set),
            "missed": sorted(gold_set - woken_set),
            "metrics_status": "scored",
        }
    else:
        wake_metrics = {
            "hit": [],
            "false_wake": [],
            "missed": [],
            "metrics_status": "unscored",
        }

    # --- suggested_actions（复用 workflow 映射，基于 woken 集；sentinel/shadow 唤醒也能出 action）---
    suggested_actions = _build_actions(
        woken,
        confidence_by_id,
        goal=goal,
        problem=problem,
        context=context,
        files=files,
    )

    reverse = _reverse_wake_hook("synthesis-stage high-risk/low-confidence re-wake (stub)")

    return {
        "activated_regions": {
            "retrieved": retrieved,
            "escalated": escalated,
            "woken": woken,
            "shadow": shadow_out,
            "reasons": reasons,
            "confidence": {rid: round(c, 3) for rid, c in confidence_by_id.items()},
        },
        "wake_metrics": wake_metrics,
        "suggested_actions": suggested_actions,
        "trace": {
            "strategy": "wake_gate_rule_v1",
            "escalate_confidence": escalate_confidence,
            "shadow_wake_threshold": shadow_wake_threshold,
            "sentinel_hits": sentinel_hits,
            "shadow_promoted": shadow_promoted,
            "models_called": False,
            "reverse_wake_triggered": reverse["triggered"],
            "use_router_api": router_api,
            "routing_trace": routing.get("trace", {}),
        },
    }
