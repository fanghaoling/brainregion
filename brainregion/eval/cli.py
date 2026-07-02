"""`brain-region eval` 子命令的编排：加载 fixtures → 解析变体/judge → 建引擎 → run_eval → 导出。

复用 server 工厂（_resolve_adapter/_resolve_endpoints/_normalize_one）+ defaults.apply，
不重造。runner 做实际运行。
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .. import defaults as _defaults_mod
from ..core.regions import REGIONS_DIR
from ..providers.litellm import LiteLLMBackend
from ..server import _normalize_one, _resolve_adapter, _resolve_endpoints
from . import store
from .calibrate import calibrate, calibrate_advice, load_gold, load_gold_advice, summarize, summarize_advice
from .judge import advice_prompt_skeleton_hash
from .metadata import rubric_hash
from .routing import (
    DEFAULT_ROUTING_VARIANTS,
    compute_routing_summary,
    make_routing_run_id,
    routing_sanity,
    run_routing_eval,
)
from .outcome import DEFAULT_OUTCOME_VARIANTS, OutcomeVariant, run_outcome_eval
from .runner import build_engines, make_run_id, run_eval
from .schema import CalibrationRecord, EvalTask, VariantSpec

logger = logging.getLogger("brainregion.eval.cli")

_DEFAULT_RUBRIC = Path(__file__).parent / "rubrics" / "review_v1.md"
_DEFAULT_OUTCOME_RUBRIC = Path(__file__).parent / "rubrics" / "advice_v1.md"


def load_tasks(fixtures_dir: str) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    for p in sorted(Path(fixtures_dir).glob("*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        for d in items:
            if not isinstance(d, dict):
                continue
            tasks.append(EvalTask(
                id=d.get("id", p.stem),
                task_type=d.get("task_type", "review"),
                difficulty=d.get("difficulty", ""),
                input=d.get("input") or {},
                gold_regions=[str(g) for g in (d.get("gold_regions") or [])],
                seed_memory=list(d.get("seed_memory") or []),
                seed_memory_irrelevant=list(d.get("seed_memory_irrelevant") or []),
                seed_memory_stale=list(d.get("seed_memory_stale") or []),
                gold_check=dict(d.get("gold_check") or {}),
                exp_type=str(d.get("exp_type") or ""),
                notes=d.get("notes", ""),
                frozen=bool(d.get("frozen", True)),
            ))
    return tasks


def parse_variants(spec: str) -> list[VariantSpec]:
    """'retrieve_off:0,retrieve_on:5,retrieve_garbage:5g' → VariantSpec 列表。

    每项 'name:k' 或 'name:kg'(g=garbage 负对照) 或 'name:k:g'。
    """
    out: list[VariantSpec] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        name = bits[0]
        k_tok = bits[1] if len(bits) > 1 else "5"
        garbage = False
        if k_tok.endswith("g"):
            garbage = True
            k_tok = k_tok[:-1]
        if len(bits) > 2 and bits[2] == "g":
            garbage = True
        try:
            k = int(k_tok)
        except ValueError:
            k = 5
        out.append(VariantSpec(name=name, retrieve_top_k=k, garbage=garbage))
    return out


def parse_judges(judges: list[str] | None, endpoint_ids: set, endpoints_cfg) -> list[dict]:
    specs = judges or []
    return [_normalize_one(s, endpoint_ids, endpoints_cfg) for s in specs]


async def run(args) -> dict:
    dd = _defaults_mod.apply()
    root = os.environ.get("UNITY_PROJECT_ROOT", ".")
    adapter = _resolve_adapter(args.adapter, root)

    variants = parse_variants(args.variants)
    if not variants:
        variants = [VariantSpec("retrieve_off", 0), VariantSpec("retrieve_on", 5),
                    VariantSpec("retrieve_garbage", 5, garbage=True)]

    endpoints_cfg = dd.get("endpoints") or {}
    endpoint_ids = set((_resolve_endpoints(endpoints_cfg) or {}).keys())
    judge_specs = args.judges or [dd.get("normalizer_model", "claude-opus-4-8")]
    judge_entries = parse_judges(judge_specs, endpoint_ids, endpoints_cfg)

    engines, backend = build_engines(adapter, dd, variants)

    rubric_path = Path(args.rubric) if getattr(args, "rubric", None) else _DEFAULT_RUBRIC
    rubric_text = rubric_path.read_text(encoding="utf-8") if rubric_path.exists() else ""
    rhash = rubric_hash(rubric_text)

    tasks = load_tasks(args.fixtures_dir)
    if not tasks:
        raise SystemExit(f"fixtures 目录无 *.yaml 任务: {args.fixtures_dir}")

    run_id = make_run_id()
    _, _, entry = await run_eval(
        tasks, variants, judge_entries, backend, engines, dd, adapter,
        rubric_text, rhash, run_id,
        effort=args.effort, max_cost_usd=args.max_cost_usd if args.max_cost_usd is not None else 1.0,
        panel_override=getattr(args, "panel", None),
    )

    exported = None
    if getattr(args, "export", None):
        exported = store.export_jsonl(run_id, args.export)

    return {
        "run_id": run_id,
        "n_tasks": entry.n_tasks,
        "variants": entry.variants,
        "judge_models": entry.judge_models,
        "metadata": {
            "git_sha": entry.git_sha, "rubric_hash": entry.rubric_hash,
            "knowledge_hash": entry.knowledge_hash, "reviewer_hash": entry.reviewer_hash,
            "defaults_hash": entry.defaults_hash,
        },
        "summary": entry.summary,
        "exported_jsonl": exported,
    }


def run_routing(args) -> dict:
    """`brain-region routing`：量 wake_gate 路由精度（免费，不调模型）。

    对带 gold_regions 的任务跑 wake_gate（A=no_defense vs B=full），聚合
    precision/recall/missed_wake_rate/false_wake_rate，sanity 检兜底是否降 missed-wake。
    """
    tasks = load_tasks(args.fixtures_dir)
    if not tasks:
        raise SystemExit(f"fixtures 目录无 *.yaml 任务: {args.fixtures_dir}")
    scored = [t for t in tasks if t.gold_regions]
    if not scored:
        raise SystemExit(f"fixtures 无 gold_regions 的任务（routing 需要）: {args.fixtures_dir}")

    regions_dir = getattr(args, "regions_dir", None) or REGIONS_DIR
    variants = DEFAULT_ROUTING_VARIANTS
    run_id = make_routing_run_id()
    records = run_routing_eval(scored, variants, run_id=run_id, regions_dir=regions_dir)
    summary = compute_routing_summary(records)
    summary["sanity"] = routing_sanity(records, summary)

    return {
        "run_id": run_id,
        "n_tasks": len(scored),
        "variants": [v.name for v in variants],
        "summary": summary,
        "per_task": [
            {
                "task_id": r.task_id,
                "variant": r.variant,
                "gold": r.gold_regions,
                "woken": r.woken,
                "hit": r.hit,
                "missed": r.missed,
                "false_wake": r.false_wake,
            }
            for r in records
        ],
    }


async def run_calibrate(args) -> dict:
    """`brain-region calibrate`：用 gold 对测盲评 judge 能否稳定把好的排在前面。"""
    dd = _defaults_mod.apply()
    registry = _resolve_endpoints(dd.get("endpoints") or {})
    backend = LiteLLMBackend(timeout=float(dd.get("timeout", 90)), endpoint_registry=registry)

    endpoint_ids = set((registry or {}).keys())
    judge_specs = args.judges or [dd.get("normalizer_model", "claude-opus-4-8")]
    judge_entries = [_normalize_one(s, endpoint_ids, dd.get("endpoints")) for s in judge_specs]

    rubric_path = Path(args.rubric) if getattr(args, "rubric", None) else _DEFAULT_RUBRIC
    rubric_text = rubric_path.read_text(encoding="utf-8") if rubric_path.exists() else ""
    rhash = rubric_hash(rubric_text)

    gold = load_gold(args.gold)
    if not gold:
        raise SystemExit(f"gold 无 *.yaml 条目: {args.gold}")

    threshold = float(getattr(args, "threshold", 0.7))
    run_id = make_run_id()
    rows = await calibrate(gold, backend, judge_entries, rubric_text, rhash, run_id)
    summary = summarize(rows, threshold=threshold)

    return {
        "run_id": run_id,
        "judge_models": [je["model"] for je in judge_entries],
        "rubric_hash": rhash,
        "n_pairs": len(gold),
        "summary": summary,
    }


async def run_calibrate_advice(args) -> dict:
    """`brain-region calibrate --advice`：用 advice gold 测 advice judge（judge_task_advice）能否稳定
    good>bad。落 CalibrationRecord artifact（outcome gate 前置）。Wilson 下界过门槛才 calibrated
    （n=10 是 smoke）。"""
    import hashlib

    dd = _defaults_mod.apply()
    registry = _resolve_endpoints(dd.get("endpoints") or {})
    backend = LiteLLMBackend(timeout=float(dd.get("timeout", 90)), endpoint_registry=registry)

    endpoint_ids = set((registry or {}).keys())
    judge_specs = args.judges or [dd.get("normalizer_model", "claude-opus-4-8")]
    judge_entries = [_normalize_one(s, endpoint_ids, dd.get("endpoints")) for s in judge_specs]

    rubric_path = Path(args.rubric) if getattr(args, "rubric", None) else _DEFAULT_OUTCOME_RUBRIC
    rubric_text = rubric_path.read_text(encoding="utf-8") if rubric_path.exists() else ""
    rhash = rubric_hash(rubric_text)
    prompt_hash = advice_prompt_skeleton_hash(rubric_text)

    gold = load_gold_advice(args.gold)
    if not gold:
        raise SystemExit(f"advice gold 无条目: {args.gold}")
    gp = Path(args.gold)
    gold_version = hashlib.sha256(gp.read_bytes()).hexdigest()[:16] if gp.is_file() else ""

    threshold = float(getattr(args, "threshold", 0.7))
    run_id = make_run_id()
    rows = await calibrate_advice(gold, backend, judge_entries, rubric_text, rhash, run_id)
    summary = summarize_advice(rows, threshold=threshold)

    date = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for je in judge_entries:
        store.record_calibration(
            CalibrationRecord(
                judge_id=je["label"], judge_model=je["model"],
                rubric_hash=rhash, prompt_hash=prompt_hash, gold_version=gold_version,
                agreement_rate=summary["agreement_rate"], wilson_lower=summary["wilson_lower"],
                threshold=threshold, passed=bool(summary["calibrated"]),
                run_id=run_id, date=date,
            ),
            summary,
        )

    return {
        "run_id": run_id,
        "judge_models": [je["model"] for je in judge_entries],
        "rubric_hash": rhash,
        "prompt_hash": prompt_hash,
        "gold_version": gold_version,
        "n_pairs": len(gold),
        "calibrated": summary["calibrated"],
        "wilson_lower": summary["wilson_lower"],
        "summary": summary,
    }


def build_outcome_variants(args) -> list:
    """据 args 标志构造 outcome 变体集(抽出来便测)。

    - 默认:default vs routed。
    - --additive:加 routed_additive(叠加式映射)做 3-way。
    - --memory:Phase2A.5 4 臂(OFF/RELEVANT/IRRELEVANT/STALE)。
    - --scoped:scoped-eval 两臂(unscoped control + scoped treatment,单变量=scope)。
    --memory / --scoped 互斥(elif,整体替换);--additive 在默认之上追加。
    """
    variants = list(DEFAULT_OUTCOME_VARIANTS)
    if getattr(args, "additive", False):
        variants.append(OutcomeVariant("routed_additive", "routed_additive"))
    if getattr(args, "memory", False):
        variants = [
            OutcomeVariant("routed", "routed"),
            OutcomeVariant("routed_memory", "routed", inject_memory=True),
            OutcomeVariant("routed_memory_irrelevant", "routed", inject_memory_irrelevant=True),
            OutcomeVariant("routed_memory_stale", "routed", inject_memory_stale=True),
        ]
    elif getattr(args, "scoped", False):
        # scoped-eval:单变量=scope。两臂同一份 seed_memory(relevant+global+distractor),
        # scoped 臂 build_scope(woken)→MemoryScope 过滤跨 region distractor;unscoped 全注入。
        from ..memory import MemoryScope
        variants = [
            OutcomeVariant("routed_memory", "routed", inject_memory=True),  # control(unscoped)
            OutcomeVariant("routed_memory_scoped", "routed", inject_memory=True,
                           build_scope=lambda woken: MemoryScope(frozenset(woken))),  # treatment
        ]
    return variants


async def run_outcome(args) -> dict:
    """`brain-region outcome`：量 wake_gate→consult 建议质量（A=default vs B=routed，真调模型+盲评+gate）。

    让 wake_gate 的 woken 真正驱动 consult 选 consultants，盲评 judge 量 useful，对照
    cost_per_useful_advice（roadmap §8 v5.5 主指标），evaluate_gate 出 GO/NO_GO/INCONCLUSIVE。
    """
    dd = _defaults_mod.apply()
    variants = build_outcome_variants(args)

    endpoints_cfg = dd.get("endpoints") or {}
    endpoint_ids = set((_resolve_endpoints(endpoints_cfg) or {}).keys())
    judge_specs = args.judges or [dd.get("normalizer_model", "claude-opus-4-8")]
    judge_entries = parse_judges(judge_specs, endpoint_ids, endpoints_cfg)

    rubric_path = Path(args.rubric) if getattr(args, "rubric", None) else _DEFAULT_OUTCOME_RUBRIC
    rubric_text = rubric_path.read_text(encoding="utf-8") if rubric_path.exists() else ""
    rhash = rubric_hash(rubric_text)

    tasks = load_tasks(args.fixtures_dir)
    if not tasks:
        raise SystemExit(f"fixtures 目录无 *.yaml 任务: {args.fixtures_dir}")

    regions_dir = getattr(args, "regions_dir", None) or REGIONS_DIR
    run_id = make_run_id()
    _, _, entry, gate = await run_outcome_eval(
        tasks, variants, judge_entries, dd, rubric_text, rhash, run_id,
        effort=args.effort,
        max_cost_usd=args.max_cost_usd if args.max_cost_usd is not None else 1.0,
        panel_override=getattr(args, "panel", None),
        regions_dir=regions_dir,
    )

    exported = None
    if getattr(args, "export", None):
        exported = store.export_jsonl(run_id, args.export)

    return {
        "run_id": run_id,
        "n_tasks": entry.n_tasks,
        "variants": entry.variants,
        "judge_models": entry.judge_models,
        "metadata": {
            "git_sha": entry.git_sha, "rubric_hash": entry.rubric_hash,
            "defaults_hash": entry.defaults_hash,
        },
        "summary": entry.summary,
        "gate": gate,
        "exported_jsonl": exported,
    }
