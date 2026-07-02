"""scoped vs unscoped outcome eval 测试(验证 Phase A region scoping thesis)。

单变量 = scope:两臂同一份 seed_memory(relevant+global+distractor),scoped 臂 build_scope(woken)
→ MemoryScope 过滤跨 region distractor。测:召回率机制( scoped 滤 distractor / unscoped 全注入)、
gate control/treatment、CLI --scoped 变体、scoped_recall 聚合、fixture wake-validity 与结构。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from brainregion.core.wake.gate import wake_gate
from brainregion.eval import cli as eval_cli
from brainregion.eval import outcome
from brainregion.eval.cli import load_tasks
from brainregion.eval.outcome import OutcomeVariant, run_outcome_eval
from brainregion.eval.schema import EvalTask
from brainregion.memory import MemoryScope
from brainregion.providers.base import ModelResponse

FIXTURES = "brainregion/eval/scoped_fixtures"


# ── fake harness(复用 test_outcome_memory_eval 套路)──────────────────────────

class _FakeJudgeBackend:
    async def complete(self, *, model, system, user, temperature=0.1, max_tokens=2048, effort=None, endpoint_id=None):
        content = json.dumps({
            "X": {"useful": 3, "missed_critical": 0}, "Y": {"useful": 2, "missed_critical": 1},
        })
        return ModelResponse(model=model, content=content, cost_usd=0.001)


class _CapturingEngine:
    """记录每次 consult 的 context_blocks(按调用序 = variant 序)。"""

    def __init__(self) -> None:
        self.calls: list[list] = []
        self.backend = _FakeJudgeBackend()

    async def consult(self, request, *, panel, consultants, max_cost_usd=None, effort=None,
                      consultation_id=None, context_blocks=None):
        from brainregion.core.consult.report import ConsultAdvice, ConsultReport
        self.calls.append(list(context_blocks or []))
        return ConsultReport(
            consultation_id="c", summary="s",
            individual=[ConsultAdvice(id="c0", model="m", consultant="debugger", summary="s")],
            usage={"cost_usd": 0.001},
        )


def _scoped_task() -> EvalTask:
    """relevant(debugging)+ global + 2 distractor(unity_ecs/planning)。wake 只醒 debugging。"""
    return EvalTask(
        id="sc-test-1", task_type="consult",
        input={"problem": "Flaky AssertionError race condition around DB reconnect, intermittent",
               "why_stuck": "can't reproduce", "question": "race fix?"},
        gold_regions=["debugging"],
        seed_memory=[
            {"id": "r1", "role": "relevant", "region": "debugging",
             "summary": "DB race: connection-bound semaphore 串行化",
             "details": "semaphore 串行化 reconnect 与事务", "triggers": ["race", "reconnect", "flaky", "AssertionError"]},
            {"id": "g1", "role": "relevant", "region": "",
             "summary": "先抓交错日志再改", "details": "log interleaving",
             "triggers": ["race", "reproduce", "flaky"]},
            {"id": "d1", "role": "distractor", "region": "unity_ecs",
             "summary": "用 EntityCommandBuffer playback 顺序",
             "details": "ecs race fix", "triggers": ["race", "flaky", "reconnect"]},
            {"id": "d2", "role": "distractor", "region": "planning",
             "summary": "排个 rollback milestone",
             "details": "plan it", "triggers": ["race", "rollback", "milestone"]},
        ],
    )


SCOPED_VARIANTS = [
    OutcomeVariant("routed_memory", "routed", inject_memory=True),  # unscoped control
    OutcomeVariant("routed_memory_scoped", "routed", inject_memory=True,
                   build_scope=lambda woken: MemoryScope(frozenset(woken))),  # treatment
]


# ── 召回率机制(scoped 滤 distractor;unscoped 全注入)──────────────────────────

@pytest.mark.asyncio
async def test_scoped_filters_distractors_unscoped_injects_all(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    engine = _CapturingEngine()
    monkeypatch.setattr(outcome, "_build_consult_engine", lambda dd: engine)
    judge_entries = [{"label": "j", "model": "fake-judge", "endpoint_id": None}]

    records, _jdgs, entry, _gate = await run_outcome_eval(
        [_scoped_task()], SCOPED_VARIANTS, judge_entries, dd={},
        rubric_text="", rubric_hash="h", run_id="run-scoped",
        max_cost_usd=0.5, require_calibration=False,
    )

    # consult 调用序 = variants 序:unscoped 先(4 块),scoped 后(2 块)
    assert len(engine.calls) == 2
    unscoped_blocks, scoped_blocks = engine.calls[0], engine.calls[1]
    assert len(unscoped_blocks) == 4              # relevant+global+2 distractor
    assert len(scoped_blocks) == 2                # relevant+global,distractor 被 region 滤
    scoped_ids = {b.metadata.get("id") for b in scoped_blocks}
    assert scoped_ids == {"r1", "g1"}             # distractor d1/d2 不在
    assert "d1" not in {b.metadata.get("id") for b in scoped_blocks}

    # memory_instrumentation:per (task,variant) 召回率字段
    instr = {m["variant"]: m for m in entry.summary["memory_instrumentation"]}
    assert instr["routed_memory"]["distractor_injected"] == 2
    assert instr["routed_memory"]["relevant_injected"] == 2
    assert instr["routed_memory_scoped"]["distractor_injected"] == 0   # 机制铁证
    assert instr["routed_memory_scoped"]["relevant_injected"] == 2
    assert instr["routed_memory_scoped"]["scoped"] is True
    assert instr["routed_memory_scoped"]["top_k"] == 5                 # GPT r2③ 透明


@pytest.mark.asyncio
async def test_scoped_recall_aggregation_median_mean(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(outcome, "_build_consult_engine", lambda dd: _CapturingEngine())
    judge_entries = [{"label": "j", "model": "fake-judge", "endpoint_id": None}]
    _records, _jdgs, entry, _gate = await run_outcome_eval(
        [_scoped_task()], SCOPED_VARIANTS, judge_entries, dd={},
        rubric_text="", rubric_hash="h", run_id="run-scoped2",
        max_cost_usd=0.5, require_calibration=False,
    )
    rec = entry.summary["scoped_recall"]
    # scoped 臂 distractor_leak median/mean = 0;unscoped = 1.0
    assert rec["routed_memory_scoped"]["distractor_leak_rate"]["median"] == 0.0
    assert rec["routed_memory"]["distractor_leak_rate"]["median"] == 1.0
    assert rec["routed_memory_scoped"]["relevant_recall"]["mean"] == 1.0


# ── gate control/treatment(GPT r2① has_scoped)─────────────────────────────────

@pytest.mark.asyncio
async def test_gate_scoped_control_treatment(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    captured: dict = {}

    def _spy(*a, **kw):
        captured.update(kw)
        return {"decision": "pilot_INCONCLUSIVE", "diagnostics": {}, "hard_gates": {}, "reasons": ["stub"]}

    monkeypatch.setattr(outcome, "evaluate_gate", _spy)
    monkeypatch.setattr(outcome, "_build_consult_engine", lambda dd: _CapturingEngine())
    judge_entries = [{"label": "j", "model": "fake-judge", "endpoint_id": None}]
    await run_outcome_eval(
        [_scoped_task()], SCOPED_VARIANTS, judge_entries, dd={},
        rubric_text="", rubric_hash="h", run_id="run-scoped-gate",
        max_cost_usd=0.5, require_calibration=False,
    )
    assert captured["control"] == "routed_memory"
    assert captured["treatment"] == "routed_memory_scoped"
    assert captured["cfg"].cost_primary is False


# ── CLI --scoped 变体构造 ──────────────────────────────────────────────────────

def test_cli_scoped_builds_two_variants():
    """--scoped → 2 臂(unscoped control + scoped treatment);build_scope 产 MemoryScope。"""
    args = SimpleNamespace(memory=False, additive=False, scoped=True)
    variants = eval_cli.build_outcome_variants(args)
    assert [v.name for v in variants] == ["routed_memory", "routed_memory_scoped"]
    assert variants[0].build_scope is None                       # unscoped control
    assert variants[1].build_scope is not None                   # scoped treatment
    # build_scope(woken) 产 MemoryScope(GPT r2① 类型,非字符串)
    assert variants[1].build_scope(["debugging"]) == MemoryScope(frozenset({"debugging"}))


def test_cli_scoped_mutually_exclusive_with_memory():
    """--memory 优先于 --scoped(elif);memory 在则不出 scoped 臂。"""
    args = SimpleNamespace(memory=True, additive=False, scoped=True)
    variants = eval_cli.build_outcome_variants(args)
    assert any(v.name == "routed_memory_stale" for v in variants)   # memory 4 臂
    assert not any(v.build_scope is not None for v in variants)     # 无 scoped 臂


# ── fixture wake-validity(GPT r2②)+ 结构 ──────────────────────────────────────

def test_fixtures_wake_does_not_wake_distractor_regions():
    """每条 fixture:wake 唤醒 gold region,但**不唤醒任何 distractor region**(实验有效)。"""
    tasks = load_tasks(FIXTURES)
    assert tasks, "scoped_fixtures 未加载到任务"
    for t in tasks:
        inp = t.input or {}
        out = wake_gate(problem=inp.get("problem", ""), goal=inp.get("goal", ""),
                        context=inp.get("why_stuck", ""), gold_regions=list(t.gold_regions or []))
        woken = set((out.get("activated_regions") or {}).get("woken") or [])
        distractor_regions = {str(m.get("region")) for m in (t.seed_memory or [])
                              if m.get("role") == "distractor" and m.get("region")}
        leak = woken & distractor_regions
        assert not leak, f"{t.id}: wake 误醒了 distractor region {leak}(实验失效)"


def test_fixtures_structure():
    """每 task:≥1 relevant(woken/gold region)+ 1 global relevant + ≥2 distractor;
    distractor region ≠ gold region;seed 都有 unique id + role;seed 数 ≤ top_k(5)。"""
    tasks = load_tasks(FIXTURES)
    for t in tasks:
        seeds = t.seed_memory or []
        ids = [str(m.get("id")) for m in seeds]
        assert len(ids) == len(set(ids)), f"{t.id}: seed id 不唯一"
        assert all(m.get("role") in ("relevant", "distractor") for m in seeds), f"{t.id}: role 缺/非法"
        relevant = [m for m in seeds if m.get("role") == "relevant"]
        distractors = [m for m in seeds if m.get("role") == "distractor"]
        assert any(m.get("region") in (t.gold_regions or [""]) or m.get("region") == "" for m in relevant), \
            f"{t.id}: 无 relevant 落在 gold/global"
        assert any(m.get("region") == "" for m in relevant), f"{t.id}: 无 global relevant(include_global 未测)"
        assert len(distractors) >= 2, f"{t.id}: distractor < 2(GPT r1① 多 distractor)"
        for d in distractors:
            assert d.get("region") not in (t.gold_regions or []), f"{t.id}: distractor region == gold"
        assert len(seeds) <= 5, f"{t.id}: seed 数 {len(seeds)} > top_k 5(召回被 cutoff,GPT r2③)"
