"""Phase 4 formal A/B harness 测试。

覆盖 review 双强(2026-07-08,gpt-5.5 + opus-4-8)硬化簇:
- EchoStrategy 控制臂:无 LLM(cost=0)、复述主脑上一句 thought(剥离 tool-call)、忽略 memory_rough_map。
- D.3 回归:real StrategyRegion.reason 接受 prev_assistant(忽略),_plan_via_strategy 透传不崩 real 路径。
- 过程指标:revisit_rate(循环高/线性低/零移动 None 除零守卫)、coverage(size² 分母 cap 1.0)、positions 重放。
- config 级 bootstrap:delta 方向 + CI + gate;按 config(非 run)聚合(pseudo-replication 守卫)。
- cost-cap:matched-set 截断 → 实际 n_runs 聚合 + 不完整矩阵 → gate INCONCLUSIVE。
- signal_regime:全解/全未解 flag。
- run_env_eval 端到端:give-up backend × configs×arms×repeats → 报告结构 + CSV 行数。
"""
from __future__ import annotations

import asyncio
import csv as _csv
import json
import tempfile

from brainregion.providers.base import ModelResponse
from brainregion.sandbox.env_eval import (
    ARMS_MEMORY_STRATEGY,
    ARMS_METRONOME,
    ARM_PRESETS,
    EchoStrategy,
    EnvArm,
    EnvConfig,
    _aggregate,
    _coverage,
    _positions_from_traj,
    _revisit_rate,
    _status_referenced,
    build_regions_for_arm,
    make_status_injector,
    render_env_eval_summary,
    run_env_eval,
    write_report,
)
from brainregion.sandbox.envs import GridWorld
from brainregion.sandbox.loop import run_agent, scoped_env, scoped_memory_mode
from brainregion.sandbox.regions import MemoryRegion, StrategyRegion
from brainregion.sandbox.regions.strategy_region import _strip_to_thought
from brainregion.sandbox import cleanup_run_dir, make_run_dir
from brainregion.sandbox.envs import build_env_system_prompt
from brainregion.sandbox.task import SandboxTask


# ---------- helpers ----------


def _J(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


class MockBackend:
    def __init__(self, script, cost=0.001):
        self.script = script
        self.i = 0
        self.cost = cost
        self.calls = 0

    async def complete_messages(self, messages, **kw):
        content = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        self.calls += 1
        return ModelResponse(model=kw.get("model", "mock"), content=content, usage={}, cost_usd=self.cost)


class _GiveUpBackend:
    """每集都立即 done(放弃)→ solved=False;run_env_eval 端到端 plumbing 用。"""

    def __init__(self):
        self.calls = 0

    async def complete_messages(self, messages, **kw):
        self.calls += 1
        return ModelResponse(
            model="mock",
            content=_J({"thought": "放弃", "done": True, "answer": "不找了"}),
            usage={}, cost_usd=0.001,
        )


_MEMORY_JSON = _J({"current_position": "(0,0)", "rough_map": "东边开阔,起点在西角",
                   "looping_detected": "否", "goal_direction_estimate": "东南"})
_STRATEGY_JSON = _J({"intent": "向东南探索未见区", "rationale": "东边开阔",
                     "expected_outcome": "可能找到通路"})


def _make_env_verify(env):
    def verify(task, run_dir, *, python_exe=None):
        return {
            "tests_green": bool(env.solved), "solve_status": "solved" if env.solved else "tests_fail",
            "pytest": None, "gold_diff": getattr(task, "gold_diff", ""),
        }
    return verify


def _run(backend, env, *, memory_region=None, strategy_region=None, goal="找到 G", max_steps=8):
    task = SandboxTask(id="env-eval-test", goal=goal)
    run_dir = make_run_dir()
    try:
        with scoped_env(env), scoped_memory_mode():
            return asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm="none", max_steps=max_steps,
                system_prompt=build_env_system_prompt(
                    env, goal, memory=True, strategy=strategy_region is not None),
                verify_fn=_make_env_verify(env),
                memory_region=memory_region, strategy_region=strategy_region,
            ))
    finally:
        cleanup_run_dir(run_dir)


def _fake_run(config, arm, solved, *, cost=0.001, steps=10, revisit=0.2, coverage=0.5, n_plan=1, n_recall=1):
    return {"config": config, "arm": arm, "solved": solved, "steps": steps, "cost": cost,
            "termination": "done", "n_recall": n_recall, "n_plan": n_plan,
            "revisit_rate": revisit, "coverage": coverage}


def _agg(runs, configs, arms, *, cost_capped=False, cost_capped_at=None):
    return _aggregate("test-run", "mock", configs, arms, 2, runs, 0.01, cost_capped, cost_capped_at,
                      0.0, False, None, None, 2048)


# ---------- EchoStrategy 单元 + _strip_to_thought ----------


def test_strip_to_thought_extracts_reasoning_not_action():
    """review gpt-3/opus-6:echo 剥 tool-call/action,只取 thought 推理(防旧动作再当新 plan)。"""
    assert _strip_to_thought('{"thought":"我在(1,2)东有墙","tool":"plan","args":{}}') == "我在(1,2)东有墙"
    assert _strip_to_thought("纯文本无 JSON") == "纯文本无 JSON"
    assert _strip_to_thought('{"nothought":1}') == '{"nothought":1}'
    assert _strip_to_thought(None) == ""


def test_echo_strategy_no_llm_returns_prev_throat_ignores_map():
    """review 双强:Echo 无 LLM(cost=0)、复述主脑上一句 thought、忽略 memory_rough_map。"""
    backend = MockBackend(["should-not-be-called"])  # 若被调,i 会增
    region = EchoStrategy()
    res = asyncio.run(region.reason(
        backend, "mock",
        memory_rough_map="IGNORE_ALL_THIS_DATA", current_view="@.", rough_position=(1, 2),
        prev_assistant='{"thought":"我在(1,2)东有墙","tool":"plan","args":{}}',
    ))
    assert backend.calls == 0                      # 不调 backend
    assert res["cost_usd"] == 0.0 and res["ok"] is True
    assert "我在(1,2)东有墙" in res["intent"]       # 复述 thought
    assert "IGNORE_ALL_THIS_DATA" not in res["intent"]  # 忽略 map


def test_echo_strategy_prev_none_fallback_neutral():
    region = EchoStrategy()
    res = asyncio.run(region.reason(
        MockBackend([]), "mock", memory_rough_map="x", current_view="@.", rough_position=(0, 0),
        prev_assistant=None,
    ))
    assert res["cost_usd"] == 0.0
    assert res["intent"]  # 非空中性串兜底


# ---------- D.3 回归:real StrategyRegion 接受 prev_assistant ----------


def test_real_strategy_accepts_prev_assistant_kwarg():
    """review gpt-4/opus-5:统一签名加 prev_assistant(real 忽略)→ D.3 real 路径不崩。"""
    region = StrategyRegion()
    res = asyncio.run(region.reason(
        MockBackend([_STRATEGY_JSON]), "mock",
        memory_rough_map="东边开阔", current_view="@.", rough_position=(0, 0),
        prev_assistant="whatever-the-main-brain-said",
    ))
    assert res["ok"] is True and "向东南" in res["intent"]


def test_echo_via_run_agent_plan_no_extra_llm_call():
    """plan 经 EchoStrategy:主脑 plan-call 轮 thought 被复述为 intent;region 不额外调 backend。"""
    env = GridWorld(size=3, start=(0, 0), goal=(2, 2), visibility_radius=1, strict_obs=True)
    mem = MemoryRegion(start=env.start)
    strat = EchoStrategy()
    backend = MockBackend([
        _J({"thought": "看", "tool": "observe", "args": {}}),
        _J({"thought": "查记忆", "tool": "recall_map", "args": {}}),
        _MEMORY_JSON,                                                            # memory.region.reason 消费
        _J({"thought": "问策略我在(1,1)", "tool": "plan", "args": {}}),           # 主脑 plan-call(prev_assistant)
        _J({"thought": "到", "done": True, "answer": "记下"}),
    ])
    traj = _run(backend, env, memory_region=mem, strategy_region=strat)
    plan_step = next(s for s in traj.steps if s.tool == "plan")
    assert '"strategy": true' in plan_step.result_preview
    assert "问策略我在(1,1)" in plan_step.result_preview   # echo 复述主脑 plan-call 轮 thought


# ---------- 过程指标 ----------


def test_revisit_rate_loop_linear_zero_move():
    assert _revisit_rate([(0, 0), (1, 0), (0, 0), (1, 0)]) == 2 / 3   # 循环:2 次重访 / 3 次移动
    assert _revisit_rate([(0, 0), (1, 0), (2, 0), (3, 0)]) == 0.0     # 线性无重访
    assert _revisit_rate([(0, 0)]) is None                            # 零移动(give-up 首步)→ None(review 除零)
    assert _revisit_rate([(0, 0), (0, 0)]) is None                    # 全 blocked 无成功移动 → None


def test_coverage_size_squared_denominator_capped():
    env = GridWorld(size=4, start=(0, 0), goal=(3, 3), visibility_radius=1, strict_obs=True)
    cov = _coverage(env)  # 初始 _explored = start 可见域 4 格 / 16
    assert cov is not None and 0 < cov <= 1.0
    assert abs(cov - 4 / 16) < 1e-9


def test_positions_from_traj_replays_act_steps():
    env = GridWorld(size=4, start=(0, 0), goal=(3, 3), visibility_radius=1, strict_obs=True)

    class _S:
        def __init__(self, tool, args):
            self.tool = tool
            self.args = args

    class _T:
        def __init__(self, steps):
            self.steps = steps

    traj = _T([_S("act", {"action": "right"}), _S("act", {"action": "down"}), _S("observe", {})])
    assert _positions_from_traj(traj, env) == [(0, 0), (1, 0), (1, 1)]  # observe 不追


# ---------- arm 装配 ----------


def test_build_regions_for_arm_assembly():
    env = GridWorld(size=4, start=(0, 0), goal=(3, 3), visibility_radius=1, strict_obs=True)
    mem, strat, mm = build_regions_for_arm(EnvArm("memory_tool", memory_tool=True), env)
    assert mem is None and strat is None and mm is True
    mem, strat, mm = build_regions_for_arm(EnvArm("memory_only", memory_region=True), env)
    assert isinstance(mem, MemoryRegion) and strat is None and mm is True
    mem, strat, mm = build_regions_for_arm(EnvArm("memory_strategy", memory_region=True, strategy="real"), env)
    assert isinstance(strat, StrategyRegion) and mm is True
    mem, strat, mm = build_regions_for_arm(EnvArm("memory_echo", memory_region=True, strategy="echo"), env)
    assert isinstance(strat, EchoStrategy)


def test_arm_presets_shape():
    assert [a.name for a in ARMS_MEMORY_STRATEGY] == ["memory_only", "memory_strategy", "memory_echo"]
    assert ARM_PRESETS["memory-strategy"][-1].strategy == "echo"   # 控制臂在


# ---------- config 级 bootstrap 正确性 ----------


def test_aggregate_delta_direction_and_per_arm():
    """b 每 config 都比 a 解得多 → solve_rate_delta point=0.5,CI 整段>0 → pilot_GO(n=4<30)。"""
    arms = (EnvArm("a", memory_region=True), EnvArm("b", memory_region=True, strategy="real"))
    configs = [EnvConfig(size=3, seed=s) for s in (1, 2, 3, 4)]
    runs = []
    for cfg in configs:
        runs += [_fake_run(cfg.label, "a", False), _fake_run(cfg.label, "a", True)]   # a: 1/2
        runs += [_fake_run(cfg.label, "b", True), _fake_run(cfg.label, "b", True)]   # b: 2/2
    report = _agg(runs, configs, arms)
    assert report["per_arm"]["a"]["solve_rate"] == 0.5
    assert report["per_arm"]["b"]["solve_rate"] == 1.0
    d = report["pairwise"]["a_vs_b"]["solve_rate_delta"]
    assert d["point"] == 0.5
    assert d["n"] == 4                                  # config 级(n=4 configs,非 8 runs)
    assert "GO" in report["pairwise"]["a_vs_b"]["gate"]["decision"]   # pilot_GO
    assert report["signal_regime"] == "ok"


def test_aggregate_cost_capped_incomplete_forces_inconclusive():
    """review 双强 cost-cap:config2 缺 arm b → 不完整矩阵 → gate INCONCLUSIVE + incomplete_pairs。"""
    arms = (EnvArm("a", memory_region=True), EnvArm("b", memory_region=True, strategy="real"))
    configs = [EnvConfig(size=3, seed=1), EnvConfig(size=3, seed=2)]
    runs = [_fake_run("3x3_seed1", "a", True), _fake_run("3x3_seed1", "b", False),
            _fake_run("3x3_seed2", "a", True)]   # seed2 缺 b(cap 截)
    report = _agg(runs, configs, arms, cost_capped=True, cost_capped_at="3x3_seed2 arm b")
    assert report["incomplete_pairs"] is True
    assert "INCONCLUSIVE" in report["pairwise"]["a_vs_b"]["gate"]["decision"]


def test_aggregate_signal_regime_all_solve_all_fail():
    arms = (EnvArm("a", memory_region=True),)
    configs = [EnvConfig(size=3, seed=1), EnvConfig(size=3, seed=2)]
    runs_all_solve = [_fake_run("3x3_seed1", "a", True), _fake_run("3x3_seed2", "a", True)]
    runs_all_fail = [_fake_run("3x3_seed1", "a", False), _fake_run("3x3_seed2", "a", False)]
    assert _agg(runs_all_solve, configs, arms)["signal_regime"] == "all_solve"
    assert _agg(runs_all_fail, configs, arms)["signal_regime"] == "all_fail"


# ---------- run_env_eval 端到端 plumbing ----------


def test_run_env_eval_plumbing_structure_and_csv():
    """give-up backend × 2 configs × 2 arms × 2 repeats → 报告字段齐 + CSV 行数 = runs+表头 + all_fail。"""
    configs = [EnvConfig(size=5, seed=1, visibility_radius=1), EnvConfig(size=5, seed=2, visibility_radius=1)]
    arms = (EnvArm("memory_only", memory_region=True),
            EnvArm("memory_strategy", memory_region=True, strategy="real"))
    report = asyncio.run(run_env_eval(
        _GiveUpBackend(), "mock", configs, arms, repeats=2, max_cost_usd=2.0, log_progress=False,
    ))
    assert set(report["per_arm"]) == {"memory_only", "memory_strategy"}
    assert report["repeats"] == 2
    assert len(report["runs"]) == 2 * 2 * 2          # configs × arms × repeats
    assert report["signal_regime"] == "all_fail"     # 全 give-up
    assert report["per_arm"]["memory_only"]["solve_rate"] == 0.0
    assert report["cost_capped"] is False
    # 生成配置记录(gpt-5 可复现)
    assert report["temperature"] == 0.0 and report["model"] == "mock"
    with tempfile.TemporaryDirectory() as d:
        jp, cp = write_report(report, d)
        assert jp.exists() and cp.exists()
        rows = list(_csv.reader(open(cp, encoding="utf-8")))
        assert len(rows) == len(report["runs"]) + 1   # 表头 + 每 run 一行
        assert rows[0][:2] == ["config", "arm"]


def test_run_env_eval_cost_cap_stops_early():
    """max_cost_usd 极小 → cost_capped=True,runs < configs×arms×repeats。"""
    configs = [EnvConfig(size=5, seed=1, visibility_radius=1), EnvConfig(size=5, seed=2, visibility_radius=1)]
    arms = (EnvArm("memory_only", memory_region=True),
            EnvArm("memory_strategy", memory_region=True, strategy="real"))
    report = asyncio.run(run_env_eval(
        _GiveUpBackend(), "mock", configs, arms, repeats=3, max_cost_usd=0.001, log_progress=False,
    ))
    assert report["cost_capped"] is True
    assert report["cost_capped_at"]
    assert len(report["runs"]) < len(configs) * len(arms) * 3


def test_render_env_eval_summary_smoke():
    configs = [EnvConfig(size=3, seed=1), EnvConfig(size=3, seed=2)]
    arms = (EnvArm("a", memory_region=True), EnvArm("b", memory_region=True, strategy="real"))
    runs = [_fake_run("3x3_seed1", "a", True), _fake_run("3x3_seed1", "b", False),
            _fake_run("3x3_seed2", "a", False), _fake_run("3x3_seed2", "b", True)]
    report = _agg(runs, configs, arms)
    text = render_env_eval_summary(report)
    assert "env-eval" in text and "per-arm" in text and "pairwise" in text
    assert "a_vs_b" in text


# ---------- Phase 4.1 metronome push ----------


def test_arms_metronome_preset():
    assert [a.name for a in ARMS_METRONOME] == ["push_real", "push_dummy", "push_echo"]
    assert ARM_PRESETS["metronome"][-1].metronome is True
    assert all(a.metronome for a in ARMS_METRONOME)


def test_injector_real_and_dummy_matched_calls_echo_none():
    """review gpt #1+#2:real 与 dummy 都调 memory+strategy(同源同成本同调用数);echo 不调 LLM(不同源)。"""
    env = GridWorld(size=4, start=(0, 0), goal=(3, 3), visibility_radius=1, strict_obs=True)

    def make(strategy):
        mem = MemoryRegion(start=env.start)
        strat = StrategyRegion() if strategy in ("real", "dummy") else EchoStrategy()
        return EnvArm("t", memory_region=True, strategy=strategy, metronome=True), mem, strat

    class _Spy:
        def __init__(self):
            self.calls = 0
            self.i = 0

        async def complete_messages(self, messages, **kw):
            self.calls += 1
            self.i += 1
            content = _MEMORY_JSON if self.i % 2 == 1 else _STRATEGY_JSON  # 奇=memory 偶=strategy
            return ModelResponse(model="m", content=content, usage={}, cost_usd=0.001)

    with scoped_env(env):
        # real:2 调用,返真 rough_map + intent
        arm, mem, strat = make("real")
        spy = _Spy()
        inj = make_status_injector(arm, mem, strat, spy, "m", endpoint_id=None, thinking=False, effort=None)
        status_r, cost_r = asyncio.run(inj(3, []))
        assert spy.calls == 2 and "东边开阔" in status_r

        # dummy:同样 2 调用(同源同成本),但返固定模板(content-null)
        arm, mem, strat = make("dummy")
        spy = _Spy()
        inj = make_status_injector(arm, mem, strat, spy, "m", endpoint_id=None, thinking=False, effort=None)
        status_d, cost_d = asyncio.run(inj(3, []))
        assert spy.calls == 2                          # 与 real 同调用数(matched cost)
        assert "无具体地图解读" in status_d and "东边开阔" not in status_d  # 模板,非 real 输出
        assert cost_d == cost_r                        # 成本一致

        # echo:不调 LLM(不同源),返主脑上一句
        arm, mem, strat = make("echo")
        spy = _Spy()
        inj = make_status_injector(arm, mem, strat, spy, "m", endpoint_id=None, thinking=False, effort=None)
        status_e, cost_e = asyncio.run(inj(3, [{"role": "assistant", "content": '{"thought":"我在(1,1)东有墙","tool":"plan","args":{}}'}]))
        assert spy.calls == 0 and cost_e == 0.0
        assert "我在(1,1)" in status_e


def test_status_injector_fires_every_period_not_step0():
    """run_agent 钩子:period=3 → step 3,6 注入 <region_status>;step 0,1,2 不注。"""
    env = GridWorld(size=8, start=(0, 0), goal=(7, 0), visibility_radius=1, strict_obs=True)
    captured = []

    class _Rec:
        async def complete_messages(self, messages, **kw):
            captured.append([m["content"] for m in messages])
            return ModelResponse(model="m", content=_J({"thought": "走", "tool": "act", "args": {"action": "right"}}),
                                 usage={}, cost_usd=0.0)

    async def inj(step, messages):
        return f"MARKER_{step}", 0.0

    task = SandboxTask(id="t", goal="到 G")
    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            asyncio.run(run_agent(
                _Rec(), "m", task, run_dir=run_dir, arm="none", max_steps=7,
                system_prompt=build_env_system_prompt(env, "到 G"), verify_fn=_make_env_verify(env),
                status_injector=inj, status_period=3,
            ))
    finally:
        cleanup_run_dir(run_dir)
    assert len(captured) >= 7                                   # 跑满 7 步
    assert not any("MARKER" in c for c in captured[0])          # step 0 无注入
    assert not any("MARKER" in c for c in captured[1])          # step 1 无
    assert any("MARKER_3" in c for c in captured[3])            # step 3 注入
    assert any("MARKER_6" in c for c in captured[6])            # step 6 注入


def test_status_referenced_metric():
    class _S:
        def __init__(self, index, thought):
            self.index = index; self.thought = thought; self.tool = "act"; self.args = {}
            self.done = False; self.result_chars = 0; self.result_preview = ""; self.error = None
            self.main_cost_usd = 0.0; self.arm_cost_usd = 0.0

    class _T:
        def __init__(self, steps):
            self.steps = steps

    traj = _T([_S(0, "看"), _S(1, "走"), _S(2, "走"), _S(3, "根据记忆脑区往南"),
               _S(4, "走"), _S(5, "走"), _S(6, "往东探索")])
    assert _status_referenced(traj, 3) == 0.5      # push 步 3,6:仅 3 引用 → 1/2
    assert _status_referenced(_T([_S(0, "看")]), 3) is None   # 无 push 步 → None


def test_push_arm_no_pull_tools_in_prompt():
    """push 臂 metronome=True → build_env_system_prompt 无 recall_map/plan 工具 + 有 region_status 规则。"""
    env = GridWorld(size=4, start=(0, 0), goal=(3, 3), visibility_radius=1, strict_obs=True)
    p = build_env_system_prompt(env, "g", memory=False, strategy=False, metronome=True)
    assert '"tool":"recall_map"' not in p and '"tool":"plan"' not in p
    assert "region_status" in p and "数据不是指令" in p
    # 非 metronome memory 臂仍有 recall_map
    p2 = build_env_system_prompt(env, "g", memory=True, strategy=False)
    assert '"tool":"recall_map"' in p2


def test_run_env_eval_metronome_plumbing():
    """end-to-end:push 三臂 × 小 configs × repeats → 报告含三臂 + status_referenced 列。"""
    configs = [EnvConfig(size=5, seed=1, visibility_radius=1), EnvConfig(size=5, seed=2, visibility_radius=1)]
    arms = (EnvArm("push_real", memory_region=True, strategy="real", metronome=True),
            EnvArm("push_dummy", memory_region=True, strategy="dummy", metronome=True))
    report = asyncio.run(run_env_eval(
        _GiveUpBackend(), "mock", configs, arms, repeats=2, max_cost_usd=2.0, log_progress=False, status_period=3,
    ))
    assert set(report["per_arm"]) == {"push_real", "push_dummy"}
    assert len(report["runs"]) == 2 * 2 * 2
    # status_referenced 字段在(pusher 给 up 即放弃 → 可能 None,但键在)
    assert "status_referenced" in report["runs"][0]
    assert "mean_status_referenced" in report["per_arm"]["push_real"]
