"""Outcome eval：wake_gate 的 woken 真正驱动 consult 选 consultants，盲评 judge 量建议质量，
A(default) vs B(routed) 对照 cost_per_useful_advice（roadmap §8 v5.5 闸门主指标）。

level-1（LM-judge）。level-2 沙盒不做。eval-only：harness 内部建 consult 引擎 + 应用 woken→consultants，
**不改生产 server.consult_problem**。

CI-aware gate（吸收 3 轮对抗评审）：advice 校准前提（CALIBRATION_REQUIRED）+ 估计量层 bootstrap CI
（重采 task、重算聚合 ratio/delta，非 per-task ratio）+ OR 语义（任一 primary 整段 CI 确定失败即 NO_GO）+
pilot 标记（n<formal_min_n）+ 丰富 diagnostics（CI/quantiles/effective_rate/per-judge/consultant trace）。

复用：wake_gate（免费）/ ConsultEngine.consult / aggregate_variant_stats（点估计）/ judge_task_advice /
store ledger（含 eval_calibrations）。
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal

from ..core.consult.report import ConsultReport
from ..core.consult import ConsultRequest
from ..core.context import ContextQuery
from ..core.regions import REGIONS_DIR
from ..core.wake.gate import wake_gate
from ..memory import MemoryProvider, MemoryScope
from ..memory.base import ExperienceEvent
from .check import check_advice_signal, detect_memory_cite
from ..server import (
    _build_consult_engine,
    _normalize_panel,
    _resolve_consult_panel,
    _resolve_endpoints,
)
from . import store
from .judge import advice_prompt_skeleton_hash, judge_task_advice
from .prices import ensure_doc_prices_registered
from .metadata import defaults_hash, git_sha
from .runner import aggregate_variant_stats
from .schema import BlindJudgement, EvalCaseRecord, EvalLedgerEntry
from .stats import (
    bootstrap_statistic,
    cost_ratio_stat,
    missed_critical_delta_stat,
    seed_for,
    useful_delta_stat,
)

logger = logging.getLogger("brainregion.eval.outcome")

# region → consultants 映射（扶正 workflow.py:_build_actions + server._CONSULT_MODE_CONSULTANTS
# 已有的设计意图——目前它们只进 suggested_args advisory、从没到引擎）。
# memory/research/review 无天然 consult specialist → 空，由 _resolve_variant_consultants 回退默认。
REGION_CONSULTANTS: dict[str, list[str]] = {
    "debugging": ["debugger"],
    "performance": ["performance", "critic"],
    "security": ["challenge", "critic"],  # security 无同名 consultant；对应质疑/边界挑战视角
    "unity_ecs": ["unity_ecs"],
    "planning": ["architect", "test_designer", "critic"],
    "memory": [],
    "research": [],
    "review": [],  # review 走 review 管线，不量 consult
}

# 对齐 defaults.py consult_consultants（fallback 用，也是 A 臂的 default 面板）
_DEFAULT_CONSULTANTS = ["debugger", "architect", "critic"]

MappingSource = Literal["routed", "routed_additive", "fallback", "default"]
Strategy = Literal["default", "routed", "routed_additive", "wake_all"]


def consultants_for_regions(woken: list[str]) -> list[str]:
    """woken region 并集 → consultants（纯映射，去重保序）。无 specialist 的 region 贡献空；
    并集为空返回 []。不掺 fallback（回退策略随 variant strategy 变，放 _resolve_variant_consultants）。"""
    out: list[str] = []
    for rid in woken or []:
        for c in REGION_CONSULTANTS.get(rid, []):
            if c not in out:
                out.append(c)
    return out


def _resolve_variant_consultants(
    variant: "OutcomeVariant", woken: list[str], dd: dict,
) -> tuple[list[str], MappingSource]:
    """变体 → (consultants, mapping_source)。

    - default=A：静态默认面板（config consult_consultants，前额叶常驻基座）。
    - routed=B（**替换式**，原设计）：wake 派生的 region 专题专家**替换**基座；空并集回退默认。
    - routed_additive=C（**叠加式**，ISS-009 + formal NO_GO 修复）：基座 **∪** region 专题专家，
      base 在前、去重保序。模拟「前额叶常驻 + 运动皮层按需激活」——不丢通用互补性，对症增量。
      这是 mapped ⊇ base 的超集：mapped = base + specialists。
    """
    defaults = list(dd.get("consult_consultants") or _DEFAULT_CONSULTANTS)
    if variant.strategy == "default":
        return defaults, "default"
    if variant.strategy == "routed":
        mapped = consultants_for_regions(woken)
        if mapped:
            return mapped, "routed"
        return defaults, "fallback"
    if variant.strategy == "routed_additive":
        mapped = consultants_for_regions(woken)
        out = list(defaults)  # base 在前
        for c in mapped:      # 叠加 region 专题专家，去重
            if c not in out:
                out.append(c)
        return out, "routed_additive"
    raise NotImplementedError("wake_all strategy 预留（roadmap §2）；follow-up")


@dataclass
class OutcomeVariant:
    name: str
    strategy: Strategy = "default"
    inject_memory: bool = False  # Phase2A RELEVANT 臂（不改 strategy，正交轴）
    inject_memory_irrelevant: bool = False  # Phase2A.5 IRRELEVANT 臂（跨域无关，控 token 长度）
    inject_memory_stale: bool = False        # Phase2A.5 STALE 臂（过时事实当现状）
    # scoped-eval 臂:woken 是 per-task 动态 → 持「woken→MemoryScope 的函数」(类型即 MemoryScope,
    # 无字符串/枚举增长)。runner 调 v.build_scope(shared_wake["woken"]);None=unscoped。GPT r2①。
    build_scope: Callable[[list[str]], MemoryScope | None] | None = None


DEFAULT_OUTCOME_VARIANTS = [
    OutcomeVariant("default", "default"),
    OutcomeVariant("routed", "routed"),
]


@dataclass
class GateConfig:
    """闸门阈值（默认对齐 eval_harness §6）。集中于此——实验/CI/论文改阈值不动代码。"""

    cost_ratio: float = 0.85              # primary: cost_ratio CI high ≤ 此值（整段满足降本）
    cost_primary: bool = True             # False = cost 不进 GO/NO_GO 判定（仅 diagnostic）。
    # 覆盖型 treatment（memory/additive）召回免费 + 两臂同 panel → cost 结构持平，降本非其目标 →
    # cost 闸门结构上不可能过（additive/memory 两例证明）。False 时看 useful + missed_critical。
    missed_wake_rate_max: float = 0.10    # hard: missed_wake_rate ≤ 此值（路由层，点估计）
    latency_p95_floor_ms: float = 6000.0  # hard: latency_p95 的绝对下限
    latency_ratio_max: float = 1.5        # hard: latency_p95 ≤ max(ratio×A, floor)
    min_tasks: int = 4                    # 低于此 → INCONCLUSIVE（样本不足）
    formal_min_n: int = 30                # 低于此 → pilot_ 前缀（不宣称"可信闸门"）
    confidence: float = 0.95              # CI 置信水平


@dataclass
class OutcomeRecord:
    """单任务 × 单变体 的 consult 产出。独立 dataclass（不 mimic EvalCaseRecord），字段按职责命名；
    喂 store 时走 to_case_record() 薄 adapter。"""

    run_id: str
    task_id: str
    variant: str
    report_summary: dict = field(default_factory=dict)  # consult 产出：advice_count/failed_count
    wake: dict = field(default_factory=dict)            # strategy/mapping_source/consultant trace/woken/wake_metrics
    cost: dict = field(default_factory=dict)            # {inference_usd, estimated_usd, total_tokens}
    latency_ms: float = 0.0
    outputs_json: str = ""
    error: str = ""

    def to_case_record(self) -> EvalCaseRecord:
        """映射到 store.record_case 期望的 EvalCaseRecord shape（wake 并入 report_summary 持久化）。"""
        return EvalCaseRecord(
            run_id=self.run_id, task_id=self.task_id, variant=self.variant,
            report_summary={**self.report_summary, "wake": self.wake},
            retrieved_case_ids=[],
            cost=self.cost, latency_ms=self.latency_ms,
            outputs_json=self.outputs_json, error=self.error,
        )


def build_outcome_engines(dd: dict):
    """从 server._build_consult_engine 拿真 ConsultEngine + backend（judge 复用 backend）。单引擎。"""
    engine = _build_consult_engine(dd)
    return engine, engine.backend


def _build_request(task) -> ConsultRequest:
    inp = task.input or {}
    return ConsultRequest(
        problem=inp.get("problem", ""),
        context=inp.get("context", ""),
        files=inp.get("files") or {},
        logs=inp.get("logs", ""),
        attempts=inp.get("attempts") or [],
        goal=inp.get("goal", ""),
        current_attempt=inp.get("current_attempt", ""),
        why_stuck=inp.get("why_stuck", ""),
        question=inp.get("question", ""),
        desired_output=inp.get("desired_output", ""),
        constraints=inp.get("constraints") or [],
    )


def _seed_to_event(m: dict) -> ExperienceEvent:
    """fixture seed_memory dict → ExperienceEvent（eval 不读 DB，纯内存召回 = 防伪记忆）。"""
    return ExperienceEvent(
        id=str(m.get("id") or ""),
        region=str(m.get("region") or ""),
        summary=str(m.get("summary") or ""),
        details=str(m.get("details") or ""),
        triggers=[str(t) for t in (m.get("triggers") or [])],
    )


def _task_context(task) -> str:
    """给 judge 的会诊问题摘要（problem/why_stuck/question/goal）——判相关性/missing_critical 的锚。"""
    inp = task.input or {}
    parts = []
    for k in ("problem", "why_stuck", "question", "goal"):
        v = inp.get(k)
        if v:
            parts.append(f"{k}: {v}")
    return "\n".join(parts)


def _run_wake_once(task, regions_dir) -> dict:
    """每 task 跑一次 wake_gate（所有 variant 共用），返回 woken + wake_metrics + shadow_promoted。"""
    inp = task.input or {}
    out = wake_gate(
        goal=inp.get("goal", ""),
        problem=inp.get("problem", ""),
        context=inp.get("context", ""),
        files=inp.get("files"),
        gold_regions=list(task.gold_regions or []),
        regions_dir=regions_dir,
    )
    ar = out.get("activated_regions") or {}
    return {
        "woken": list(ar.get("woken") or []),
        "wake_metrics": dict(out.get("wake_metrics") or {}),
        "shadow_promoted": int((out.get("trace") or {}).get("shadow_promoted") or 0),
    }


def _consultant_trace(consultants: list[str], report: ConsultReport) -> dict:
    """planned→returned→failed→dropped_by_budget 链路（全从 report 推导，无需改 engine）。
    一眼 debug"为什么没叫 architect"：wake 没醒 / budget 裁 / 执行超时。"""
    planned = set(consultants)
    returned = {a.consultant for a in (report.individual or []) if getattr(a, "consultant", None)}
    failed = {fm.get("consultant") for fm in (report.failed_models or []) if fm.get("consultant")}
    dropped = planned - returned - failed
    return {
        "planned": sorted(planned),
        "returned": sorted(returned),
        "failed": sorted(failed),
        "dropped_by_budget": sorted(dropped),
    }


async def run_outcome_variant(
    engine, request: ConsultRequest, panel: list, consultants: list[str],
    variant: OutcomeVariant, mapping_source: MappingSource, shared_wake: dict,
    effort, max_cost_usd, run_id: str, task_id: str,
    context_blocks: list | None = None,
) -> OutcomeRecord:
    t0 = time.perf_counter()
    wake_info = {
        "strategy": variant.strategy,
        "mapping_source": mapping_source,
        "consultants": list(consultants),
        "woken": shared_wake.get("woken", []),
        "wake_metrics": shared_wake.get("wake_metrics", {}),
        "shadow_promoted": shared_wake.get("shadow_promoted", 0),
    }
    try:
        report: ConsultReport = await engine.consult(
            request, panel=panel, consultants=consultants,
            max_cost_usd=max_cost_usd, effort=effort,
            context_blocks=context_blocks or [],
        )
        dt = (time.perf_counter() - t0) * 1000.0
        wake_info["consultant_trace"] = _consultant_trace(consultants, report)
        return OutcomeRecord(
            run_id=run_id, task_id=task_id, variant=variant.name,
            report_summary={
                "advice_count": len(report.individual),
                "failed_count": len(report.failed_models),
            },
            wake=wake_info,
            cost={
                "inference_usd": (report.usage or {}).get("cost_usd"),
                "estimated_usd": (report.budget or {}).get("estimated_usd"),
                "total_tokens": (report.usage or {}).get("total_tokens", 0),
            },
            latency_ms=round(dt, 1),
            outputs_json=json.dumps(report.to_dict(), ensure_ascii=False, default=str),
        )
    except Exception as e:  # noqa: BLE001 — 单变体失败不阻断整 run
        dt = (time.perf_counter() - t0) * 1000.0
        logger.warning("run_outcome_variant 失败 task=%s variant=%s: %s", task_id, variant.name, e)
        return OutcomeRecord(
            run_id=run_id, task_id=task_id, variant=variant.name,
            wake=wake_info, latency_ms=round(dt, 1),
            error=f"{type(e).__name__}: {e}",
        )


def _routed_default_overlap(records: list, variants: list[OutcomeVariant]) -> float | None:
    """routed vs default 的专家集重合率（Jaccard 均值）——高则 B 退化成 A、无信号。"""
    names = {v.strategy: v.name for v in variants}
    d_name = names.get("default")
    r_name = names.get("routed")
    if not d_name or not r_name:
        return None
    by_task: dict[str, dict[str, set]] = {}
    for r in records:
        by_task.setdefault(r.task_id, {})[r.variant] = set(((r.wake or {}).get("consultants") or []))
    rates = []
    for vm in by_task.values():
        d, r = vm.get(d_name, set()), vm.get(r_name, set())
        union = d | r
        if union:
            rates.append(len(d & r) / len(union))
    return round(statistics.mean(rates), 3) if rates else None


def compute_outcome_summary(records: list, judgements: list, variants: list[OutcomeVariant]) -> dict:
    """点估计 per_variant（用于展示）+ wake_stats + missed/false_wake。CI 在 evaluate_gate 单独算。"""
    per_variant: dict[str, dict] = {}
    for v in variants:
        recs = [r for r in records if r.variant == v.name]
        jdgs = [j for j in judgements if j.variant == v.name]
        stats = aggregate_variant_stats(recs, jdgs)
        stats["missed_critical_total"] = sum(
            int((j.scores or {}).get("missed_critical", 0) or 0) for j in jdgs
        )
        woken_counts = [len((r.wake or {}).get("woken") or []) for r in recs]
        expert_counts = [len((r.wake or {}).get("consultants") or []) for r in recs]
        shadow_total = sum(int((r.wake or {}).get("shadow_promoted") or 0) for r in recs)
        sources = [(r.wake or {}).get("mapping_source") for r in recs]
        fallback_n = sum(1 for s in sources if s == "fallback")
        stats["wake_stats"] = {
            "avg_regions_woken": round(statistics.mean(woken_counts), 2) if woken_counts else 0.0,
            "avg_experts_selected": round(statistics.mean(expert_counts), 2) if expert_counts else 0.0,
            "shadow_promoted_total": shadow_total,
            "fallback_rate": round(fallback_n / len(sources), 3) if sources else 0.0,
            "mapping_source_breakdown": {str(s): sources.count(s) for s in sorted(set(sources))},
        }
        missed_rates, false_rates = [], []
        for r in recs:
            wm = (r.wake or {}).get("wake_metrics") or {}
            missed = wm.get("missed") or []
            hit = wm.get("hit") or []
            false_wake = wm.get("false_wake") or []
            woken = (r.wake or {}).get("woken") or []
            gold_total = len(set(missed) | set(hit))
            if gold_total:
                missed_rates.append(len(missed) / gold_total)
            if woken:
                false_rates.append(len(false_wake) / len(woken))
        stats["missed_wake_rate"] = round(statistics.mean(missed_rates), 3) if missed_rates else 0.0
        stats["false_wake_rate"] = round(statistics.mean(false_rates), 3) if false_rates else 0.0
        per_variant[v.name] = stats
    return {
        "per_variant": per_variant,
        "routed_default_overlap_rate": _routed_default_overlap(records, variants),
    }


def outcome_sanity(records: list, judgements: list, variants: list[OutcomeVariant]) -> dict:
    """errors=结构性失败；warnings=观察（cost None / 运行失败 / 盲评解析失败）。"""
    errors: list[str] = []
    warnings: list[str] = []
    for r in records:
        if r.cost and r.cost.get("inference_usd") is None and not r.error:
            warnings.append(
                f"task={r.task_id} variant={r.variant} inference_usd=None（litellm 无单价，ISS-003）"
            )
        if r.error:
            warnings.append(f"task={r.task_id} variant={r.variant} 运行失败: {r.error}")
    parse_fails = [j for j in judgements if "parse" in (j.reason or "").lower() or not j.scores]
    if parse_fails:
        warnings.append(f"{len(parse_fails)} 条盲评解析失败/空（judge 输出非 JSON）")
    return {"errors": errors, "warnings": warnings}


def _per_judge_metrics(judgements: list, control: str, treatment: str) -> dict:
    """per-judge useful/missed_critical 的 mean/std/min/max（吸收 Rec 2：judge 分歧诊断）。
    返回 {judge_id: {metric: {mean,std,min,max,n}}}。"""
    by: dict[str, dict[str, list[float]]] = {}
    for j in judgements:
        for m in ("useful", "missed_critical"):
            v = float((j.scores or {}).get(m, 0) or 0)
            by.setdefault(j.judge_id, {}).setdefault(m, []).append(v)
    out = {}
    for jid, metrics in by.items():
        out[jid] = {}
        for m, vals in metrics.items():
            out[jid][m] = {
                "mean": round(statistics.mean(vals), 3) if vals else 0.0,
                "std": round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0,
                "min": min(vals) if vals else 0.0,
                "max": max(vals) if vals else 0.0,
                "n": len(vals),
            }
    return out


def per_task_metrics(records: list, judgements: list, control: str = "default",
                     treatment: str = "routed") -> tuple[list, dict]:
    """records/judgements → (task_rows, meta)。

    task_rows: [{control: {cost, useful, total_advice, missed_critical}, treatment: {...}}, ...]
    ——按 variant_name 键控（不写死，吸收 GPT：未来多变体不用改 stats）。useful/missed_critical 跨 judge
    **取 mean**（多 judge 测同一 advice，求和不该被 judge 数放大）；cost/total_advice 单值。
    meta: {effective_n, dropped_task_ids, missing}。仅两 variant 都在且有 judgement 的 task 入列。
    """
    rec_by = {(r.task_id, r.variant): r for r in records}
    jdg_by: dict[tuple, list] = {}
    for j in judgements:
        jdg_by.setdefault((j.task_id, j.variant), []).append(j)
    task_ids = sorted({tid for (tid, _v) in rec_by} | {tid for (tid, _v) in jdg_by})

    def _mean(jdgs, key):
        vals = [float((j.scores or {}).get(key, 0) or 0) for j in jdgs]
        return sum(vals) / len(vals) if vals else 0.0

    rows: list[dict] = []
    dropped: list[str] = []
    for tid in task_ids:
        rec_c, rec_t = rec_by.get((tid, control)), rec_by.get((tid, treatment))
        jdgs_c, jdgs_t = jdg_by.get((tid, control), []), jdg_by.get((tid, treatment), [])
        if not rec_c or not rec_t or not jdgs_c or not jdgs_t:
            dropped.append(tid)
            continue
        rows.append({
            control: {"cost": float((rec_c.cost or {}).get("inference_usd") or 0),
                      "useful": _mean(jdgs_c, "useful"),
                      "total_advice": float((rec_c.report_summary or {}).get("advice_count") or 0),
                      "missed_critical": _mean(jdgs_c, "missed_critical")},
            treatment: {"cost": float((rec_t.cost or {}).get("inference_usd") or 0),
                        "useful": _mean(jdgs_t, "useful"),
                        "total_advice": float((rec_t.report_summary or {}).get("advice_count") or 0),
                        "missed_critical": _mean(jdgs_t, "missed_critical")},
        })
    return rows, {"effective_n": len(rows), "dropped_task_ids": dropped,
                  "missing": [t for t in task_ids if t not in dropped]}


def _boot_ci(rows, run_id, control, treatment, confidence):
    """三个估计量的 bootstrap CI（每 metric 独立 seed 流）。"""
    return {
        "cost_ratio": bootstrap_statistic(
            rows, lambda rs: cost_ratio_stat(rs, control, treatment),
            confidence=confidence, seed=seed_for(run_id, "cost_ratio")),
        "useful_delta": bootstrap_statistic(
            rows, lambda rs: useful_delta_stat(rs, control, treatment),
            confidence=confidence, seed=seed_for(run_id, "useful_delta")),
        "missed_critical_delta": bootstrap_statistic(
            rows, lambda rs: missed_critical_delta_stat(rs, control, treatment),
            confidence=confidence, seed=seed_for(run_id, "missed_critical_delta")),
    }


def evaluate_gate(
    records: list, judgements: list, variants: list[OutcomeVariant], *,
    run_id: str, cfg: GateConfig | None = None,
    control: str = "default", treatment: str = "routed",
    calibration_ok: bool = True, confidence: float = 0.95,
) -> dict:
    """CI-aware gate（吸收 3 轮评审）。决策 ∈ {GO, NO_GO, INCONCLUSIVE, CALIBRATION_REQUIRED, pilot_*}。

    - GO：hard gates 全过 且三 primary（cost_ratio / useful_delta / missed_critical_delta）整段 CI 满足。
    - NO_GO（OR 语义）：任一 primary 整段 CI 确定失败（cost_ratio CI low>thr / useful CI high<0 /
      missed_critical CI low>0）或 hard gate 破。
    - INCONCLUSIVE：CI 跨阈值 / n<min / bootstrap None。
    - CALIBRATION_REQUIRED：校准 artifact 缺失/未达标（前置）。
    - pilot_ 前缀：n<formal_min_n（不宣称"可信闸门"）。
    """
    cfg = cfg or GateConfig()
    rows, meta = per_task_metrics(records, judgements, control, treatment)
    n = meta["effective_n"]
    boot = _boot_ci(rows, run_id, control, treatment, confidence)
    cr, ud, md = boot["cost_ratio"], boot["useful_delta"], boot["missed_critical_delta"]

    # hard gates（路由层 missed_wake + latency，点估计；从 records 取 treatment 侧）
    treat_recs = [r for r in records if r.variant == treatment]
    control_recs = [r for r in records if r.variant == control]
    missed_rates = []
    for r in treat_recs:
        wm = (r.wake or {}).get("wake_metrics") or {}
        gold = len(set(wm.get("missed") or []) | set(wm.get("hit") or []))
        if gold:
            missed_rates.append(len(wm.get("missed") or []) / gold)
    missed_wake_b = statistics.mean(missed_rates) if missed_rates else 0.0
    lat_a_vals = [float(r.latency_ms or 0) for r in control_recs]
    lat_b_vals = [float(r.latency_ms or 0) for r in treat_recs]
    lat_a = _pctl(lat_a_vals, 0.95)
    lat_b = _pctl(lat_b_vals, 0.95)
    lat_limit = max(cfg.latency_ratio_max * lat_a, cfg.latency_p95_floor_ms)
    hard = {
        "missed_wake_rate_B": round(missed_wake_b, 3),
        "missed_wake_ok": missed_wake_b <= cfg.missed_wake_rate_max,
        "latency_p95_B": round(lat_b, 1),
        "latency_limit": round(lat_limit, 1),
        "latency_ok": lat_b <= lat_limit,
    }
    hard_all_ok = hard["missed_wake_ok"] and hard["latency_ok"]

    reasons: list[str] = []
    pilot = n < cfg.formal_min_n

    if not calibration_ok:
        decision = "CALIBRATION_REQUIRED"
        reasons.append("advice judge 校准 artifact 缺失/未达标/不匹配 → 先 `brain-region calibrate --advice`")
    else:
        any_none = any(b["point"] is None for b in boot.values())
        cost_ok = cr["point"] is not None and cr["high"] is not None and cr["high"] <= cfg.cost_ratio
        useful_ok = ud["point"] is not None and ud["low"] is not None and ud["low"] >= 0
        missed_ok = md["point"] is not None and md["high"] is not None and md["high"] <= 0
        cost_fail = (cr["low"] is not None and cr["low"] > cfg.cost_ratio) if cfg.cost_primary else False
        useful_fail = ud["high"] is not None and ud["high"] < 0
        missed_fail = md["low"] is not None and md["low"] > 0

        if not hard_all_ok:
            decision = "NO_GO"
            if not hard["missed_wake_ok"]:
                reasons.append(f"hard: missed_wake_rate_B={missed_wake_b:.3f} > {cfg.missed_wake_rate_max}")
            if not hard["latency_ok"]:
                reasons.append(f"hard: latency_p95_B={lat_b:.1f}ms > limit {lat_limit:.1f}ms")
        elif any_none or n < cfg.min_tasks:
            decision = "INCONCLUSIVE"
            if any_none:
                reasons.append("某 primary bootstrap 点估计 None（Σuseful=0 / Σtotal=0）")
            if n < cfg.min_tasks:
                reasons.append(f"有效配对 n={n} < min_tasks={cfg.min_tasks}")
        elif cost_fail or useful_fail or missed_fail:
            decision = "NO_GO"
            if cost_fail:
                reasons.append(f"cost_ratio CI low={cr['low']:.4f} > {cfg.cost_ratio}（整段确定没降本）")
            if useful_fail:
                reasons.append(f"useful_delta CI high={ud['high']:.4f} < 0（整段确定劣化）")
            if missed_fail:
                reasons.append(f"missed_critical_delta CI low={md['low']:.4f} > 0（整段确定多漏关键）")
        elif (cost_ok if cfg.cost_primary else True) and useful_ok and missed_ok:
            decision = "GO"
            if cfg.cost_primary:
                reasons.append("三 primary 整段 CI 满足（cost 降 / useful 非劣 / missed_critical 不增）且 hard gates 过")
            else:
                reasons.append("覆盖型 operating point：useful 非劣 + missed_critical 不增（cost 非目标，仅 diagnostic）且 hard gates 过")
        else:
            decision = "INCONCLUSIVE"
            reasons.append("某 primary CI 跨阈值（无法确定满足或失败）")

        if pilot and decision in {"GO", "NO_GO"}:
            decision = f"pilot_{decision}"
            reasons.append(f"pilot：有效 n={n} < formal_min_n={cfg.formal_min_n}，仅 pilot 级（不宣称可信闸门）")

    diagnostics = {
        "effective_n": n,
        "dropped_task_ids": meta["dropped_task_ids"],
        "B": next((b["B"] for b in boot.values() if b["B"]), 0),
        "confidence": confidence,
        "pilot": pilot,
        "cost_ratio_ci": {k: boot["cost_ratio"][k] for k in ("point", "low", "high", "effective_rate")},
        "useful_delta_ci": {k: boot["useful_delta"][k] for k in ("point", "low", "high", "effective_rate")},
        "missed_critical_delta_ci": {k: boot["missed_critical_delta"][k] for k in ("point", "low", "high", "effective_rate")},
        # useful_absolute_delta：Σ(treatment useful) − Σ(control useful)，跨 task×judge 求和（计数，非 rate）。
        # 与 useful_delta_ci（rate=useful/total_advice）互补——additive 产更多 advice 时 rate 会被稀释误导，
        # 绝对值才是「treatment 是否给出更多有用建议」的直读信号（additive 验证：rate 负但绝对值正）。
        "useful_absolute_delta": (
            sum(int((j.scores or {}).get("useful", 0) or 0) for j in judgements if j.variant == treatment)
            - sum(int((j.scores or {}).get("useful", 0) or 0) for j in judgements if j.variant == control)
        ),
        "bootstrap_quantiles": {m: boot[m]["quantiles"] for m in boot},
        "per_judge_metrics": _per_judge_metrics(judgements, control, treatment),
        "routed_default_overlap_rate": _routed_default_overlap(records, variants),
    }
    return {"decision": decision, "hard_gates": hard, "reasons": reasons, "diagnostics": diagnostics}


def _pctl(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def _calibration_ok(judge_entries: list[dict], rubric_hash: str, prompt_hash: str, gold_version: str = "") -> bool:
    """所有 judge 都有匹配的 pass 校准 artifact 吗？（outcome gate 前置，吸收 I6/Blocker 3）

    按 judge+rubric+prompt 取最新（gold_version 不强制匹配——outcome 不知 gold 版本，仅追溯用）。"""
    for je in judge_entries or []:
        rec = store.lookup_calibration(je.get("label", ""), je.get("model", ""),
                                       rubric_hash, prompt_hash, gold_version=None)
        if not rec or not rec.get("passed"):
            return False
    return True


def compute_memory_diagnostics(judgements: list, variants: list[OutcomeVariant]) -> dict:
    """Phase2A.5 4 臂 memory 研究实验 diagnostic（**不进 gate**，GPT：关键词钝）。

    主比较 RELEVANT vs IRRELEVANT（控 token 长度，量 information quality 而非 context length）。
    只用 judge_id=="programmatic" 的 judgement（check_advice_signal 产出）。
    """
    prog = [j for j in judgements if getattr(j, "judge_id", "") == "programmatic"]
    if not prog:
        return {}
    by_task: dict[str, dict[str, dict]] = {}
    for j in prog:
        s = j.scores or {}
        by_task.setdefault(j.task_id, {})[j.variant] = {
            "violation": int(s.get("constraint_violation") or 0),
            "applies": int(s.get("applies_context") or 0),
            "cite": int(s.get("cite_count") or 0),
        }
    var_names = [v.name for v in variants]
    per_arm: dict = {}
    for vn in var_names:
        vals = [by_task[t][vn] for t in by_task if vn in by_task[t]]
        if vals:
            n = len(vals)
            per_arm[vn] = {
                "n": n,
                "violation_rate": round(sum(v["violation"] for v in vals) / n, 3),
                "applies_rate": round(sum(v["applies"] for v in vals) / n, 3),
                "cite_mean": round(sum(v["cite"] for v in vals) / n, 2),
            }
    # pairwise violation delta CI（delta = mean(treatment) − mean(control)；负=treatment 更少违反=好）
    pairs = [
        ("routed_memory_irrelevant", "routed_memory", "relevant_vs_irrelevant"),  # 主比较
        ("routed", "routed_memory", "relevant_vs_off"),
        ("routed", "routed_memory_stale", "stale_vs_off"),
    ]
    pairwise: dict = {}
    for control, treatment, label in pairs:
        if control not in var_names or treatment not in var_names:
            continue
        rows = [
            {"c": by_task[t][control]["violation"], "t": by_task[t][treatment]["violation"]}
            for t in by_task if control in by_task[t] and treatment in by_task[t]
        ]
        if len(rows) >= 2:
            def _delta(rs):
                return (sum(r["t"] for r in rs) - sum(r["c"] for r in rs)) / len(rs) if rs else None
            boot = bootstrap_statistic(rows, _delta, seed=seed_for("mem-diag", label))
            pairwise[label] = {k: boot.get(k) for k in ("point", "low", "high", "effective_rate")}
    return {
        "per_arm": per_arm,
        "pairwise_delta_ci": pairwise,
        "primary": "relevant_vs_irrelevant：CI 整段<0 = RELEVANT violation 显著少于 IRRELEVANT = memory 信息有效（超越纯上下文长度）",
    }


def _scoped_recall_summary(memory_instr: list) -> dict:
    """scoped-eval 召回率聚合(GPT r2③④):per-variant relevant_recall / distractor_leak_rate 的 median+mean。

    relevant_recall = relevant_injected/relevant_total(高=好);distractor_leak_rate = distractor_injected/
    distractor_total(scoped 该≈0、unscoped 高)。带 n + top_k 上下文(防 top_k 截断误读)。仅 scoped-eval
    臂有 relevant/distractor 计数;其他实验 memory_instr 无这些键 → 返回空 dict(向后兼容)。
    """
    by_arm: dict[str, list[dict]] = {}
    for m in memory_instr:
        if "relevant_total" not in m:
            continue
        rt, ri = int(m.get("relevant_total") or 0), int(m.get("relevant_injected") or 0)
        dt, di = int(m.get("distractor_total") or 0), int(m.get("distractor_injected") or 0)
        by_arm.setdefault(m.get("variant", ""), []).append({
            "relevant_recall": (ri / rt) if rt else None,
            "distractor_leak_rate": (di / dt) if dt else None,
        })
    out: dict = {}
    for arm, recs in by_arm.items():
        def _mm(key):
            vals = [r[key] for r in recs if r[key] is not None]
            if not vals:
                return {"median": None, "mean": None, "n": 0}
            return {"median": round(statistics.median(vals), 3),
                    "mean": round(statistics.mean(vals), 3), "n": len(vals)}
        out[arm] = {"relevant_recall": _mm("relevant_recall"),
                    "distractor_leak_rate": _mm("distractor_leak_rate")}
    return out


async def run_outcome_eval(
    tasks: list, variants: list[OutcomeVariant], judge_entries: list[dict],
    dd: dict, rubric_text: str, rubric_hash: str, run_id: str,
    effort=None, max_cost_usd: float = 1.0, panel_override: list | None = None,
    *, regions_dir=REGIONS_DIR, gold_version: str = "", require_calibration: bool = True,
) -> tuple[list, list, EvalLedgerEntry, dict]:
    """主编排（仿 runner.run_eval，但量 consult 而非 review）+ CI-aware gate。

    每 task：wake_gate 跑一次（所有 variant 共用）→ 每 variant 解析 consultants + run_outcome_variant +
    record_case → 每 judge judge_task_advice 盲评 + record_judgement → compute_outcome_summary +
    CI-aware evaluate_gate（前置校准校验）→ record_run。
    """
    engine, backend = build_outcome_engines(dd)
    ensure_doc_prices_registered()
    endpoint_ids = set((_resolve_endpoints(dd.get("endpoints") or {}) or {}).keys())
    records: list[OutcomeRecord] = []
    judgements: list = []
    memory_instr: list = []  # Phase2A.5：per (task,variant) memory 召回 instrumentation

    for task in tasks:
        request = _build_request(task)
        panel_src, _ = _resolve_consult_panel(panel_override, dd)
        panel = _normalize_panel(panel_src, endpoint_ids, dd.get("endpoints"))
        shared_wake = _run_wake_once(task, regions_dir)  # 每 task 一次，variant 共用
        variant_outputs: dict[str, str] = {}
        for v in variants:
            consultants, mapping_source = _resolve_variant_consultants(v, shared_wake["woken"], dd)
            context_blocks: list = []
            # Phase2A.5 4 臂 seed 选择：RELEVANT/IRRELEVANT/STALE 各从对应冻结 seed 纯内存召回。
            seed_list, arm = None, "OFF"
            if getattr(v, "inject_memory", False):
                seed_list, arm = getattr(task, "seed_memory", None), "RELEVANT"
            elif getattr(v, "inject_memory_irrelevant", False):
                seed_list, arm = getattr(task, "seed_memory_irrelevant", None), "IRRELEVANT"
            elif getattr(v, "inject_memory_stale", False):
                seed_list, arm = getattr(task, "seed_memory_stale", None), "STALE"
            if seed_list:
                evs = [_seed_to_event(m) for m in seed_list]
                top_k = int(dd.get("memory_recall_top_k", 5))
                # scoped-eval:build_scope(woken)→MemoryScope;None=unscoped(现状)。GPT r2①
                scope = v.build_scope(shared_wake["woken"]) if v.build_scope else None
                rr = MemoryProvider.from_records(evs, scope=scope).retrieve(
                    ContextQuery(text=_task_context(task), top_k=top_k)
                )
                context_blocks = rr.blocks
                # 召回率 + 混淆(scoped-eval,GPT r1④/r2③⑤):id→role 推召回;block metadata 已带 region
                id_role = {str(m.get("id")): m.get("role") for m in seed_list}
                inj_ids = [b.metadata.get("id") for b in rr.blocks]
                rel_total = sum(1 for r in id_role.values() if r == "relevant")
                dist_total = sum(1 for r in id_role.values() if r == "distractor")
                memory_instr.append({
                    "task_id": task.id, "variant": v.name, "arm": arm,
                    "scoped": scope is not None, "top_k": top_k,
                    "candidates": rr.meta.get("candidates_before_top_k"),
                    "after_scope": rr.meta.get("candidates_after_scope"),
                    "after_governance": rr.meta.get("candidates_after_governance"),
                    "injected": len(rr.blocks),
                    "relevant_total": rel_total, "distractor_total": dist_total,
                    "relevant_injected": sum(1 for i in inj_ids if id_role.get(i) == "relevant"),
                    "distractor_injected": sum(1 for i in inj_ids if id_role.get(i) == "distractor"),
                    "injected_regions": sorted({(b.metadata.get("region") or "(global)")
                                                for b in rr.blocks}),
                })
            rec = await run_outcome_variant(
                engine, request, panel, consultants, v, mapping_source, shared_wake,
                effort, max_cost_usd, run_id, task.id, context_blocks=context_blocks,
            )
            records.append(rec)
            store.record_case(rec.to_case_record())
            variant_outputs[v.name] = rec.outputs_json
        for je in judge_entries:
            try:
                js = await judge_task_advice(
                    backend, je, rubric_text, rubric_hash, run_id, task.id,
                    variant_outputs, _task_context(task),
                )
                for j in js:
                    store.record_judgement(j)
                    judgements.append(j)
            except Exception as e:  # noqa: BLE001
                logger.warning("judge_task_advice 失败 task=%s judge=%s: %s", task.id, je.get("label"), e)
        # Phase2A.5 程序 diagnostic（仅 diagnostic 不进 gate）：gold_check → 合成 programmatic judgement。
        if getattr(task, "gold_check", None):
            for v in variants:
                out = variant_outputs.get(v.name)
                if not out:
                    continue
                try:
                    rep = json.loads(out)
                    sig = check_advice_signal(rep, task.gold_check)
                    cite = detect_memory_cite(rep)
                    pj = BlindJudgement(
                        run_id=run_id, task_id=task.id, judge_id="programmatic", judge_model="rule",
                        rubric_hash="", variant=v.name, scores={**sig, **cite})
                    store.record_judgement(pj)
                    judgements.append(pj)
                except Exception as e:  # noqa: BLE001
                    logger.warning("check_advice_signal 失败 task=%s variant=%s: %s", task.id, v.name, e)

    summary = compute_outcome_summary(records, judgements, variants)
    summary["sanity"] = outcome_sanity(records, judgements, variants)

    prompt_hash = advice_prompt_skeleton_hash(rubric_text)
    calib_ok = (not require_calibration) or _calibration_ok(
        judge_entries, rubric_hash, prompt_hash, gold_version)
    # Phase2A：memory A/B 单变量——control=routed, treatment=routed_memory。
    # 修 gate 静默：原调用漏传 control/treatment → 新 arm 对 GO/NO_GO 不可见（用默认 default/routed）。
    has_scoped = any(v.build_scope is not None for v in variants)  # scoped-eval(GPT r2①)
    has_memory = any(getattr(v, "inject_memory", False) for v in variants)
    gate_kwargs = {"run_id": run_id, "calibration_ok": calib_ok}
    if has_scoped:
        # scoped vs unscoped(两臂都有 memory,同 panel)→ cost 持平非目标;control=unscoped/treatment=scoped
        gate_kwargs["control"] = "routed_memory"
        gate_kwargs["treatment"] = "routed_memory_scoped"
        gate_kwargs["cfg"] = GateConfig(cost_primary=False)
    elif has_memory:
        # memory 召回免费 + 两臂同 panel → cost 结构持平，降本非其目标 → cost 不当 primary
        # （additive/memory 两例证明 cost≤0.85 闸门对覆盖型 treatment 结构上不可能过）。
        gate_kwargs["control"] = "routed"
        gate_kwargs["treatment"] = "routed_memory"
        gate_kwargs["cfg"] = GateConfig(cost_primary=False)
    else:
        gate_kwargs["confidence"] = GateConfig().confidence
    gate = evaluate_gate(records, judgements, variants, **gate_kwargs)
    summary["gate"] = gate
    summary["memory_diagnostics"] = compute_memory_diagnostics(judgements, variants)
    summary["memory_instrumentation"] = memory_instr
    summary["scoped_recall"] = _scoped_recall_summary(memory_instr)

    entry = EvalLedgerEntry(
        run_id=run_id,
        date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=git_sha(),
        variants=[v.name for v in variants],
        judge_models=[je["model"] for je in judge_entries],
        rubric_hash=rubric_hash,
        knowledge_hash="",  # consult 无知识库检索
        reviewer_hash="",   # consult 无 reviewer
        defaults_hash=defaults_hash(dd),
        n_tasks=len(tasks),
        summary=summary,
    )
    store.record_run(entry)
    return records, judgements, entry, gate
