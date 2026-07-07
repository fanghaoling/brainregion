"""§15.1 认知环外环(run_cognitive_loop)单测。

monkeypatch loop.run_agent 返回罐装 Trajectory(不跑真 agent loop),验外环逻辑:
重跑条件、终止原因、directive 注入、budget、失败隔离、序列化。+ _run_expert 分支(CLI)。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from brainregion.sandbox import cli as sandbox_cli
from brainregion.sandbox import loop
from brainregion.sandbox.loop import (
    CognitiveIteration,
    Trajectory,
    run_cognitive_loop,
)


def _task():
    return SimpleNamespace(id="t", goal="g", gold_diff="")


def _traj(*, tests_green, action, subgoal="", cost=0.01):
    t = Trajectory(task_id="t", arm="none")
    t.tests_green = tests_green
    t.solve_status = "solved" if tests_green else "tests_fail"
    t.n_steps = 1
    t.total_main_cost_usd = cost
    t.delegate = (None if action is None else {"action": action, "next_subgoal": subgoal})
    return t


def _patch_seq(monkeypatch, seq):
    """fake run_agent 按 seq 顺序返;记每次调用 kw(含 directive)。"""
    calls = []

    async def fake(*a, **kw):
        calls.append(kw)
        return seq[len(calls) - 1]

    monkeypatch.setattr(loop, "run_agent", fake)
    return calls


def _patch_always(monkeypatch, traj):
    async def fake(*a, **kw):
        return traj
    monkeypatch.setattr(loop, "run_agent", fake)


# ---------------- 重跑 / 终止 矩阵 ----------------
def test_loop_redelegate_then_accept(monkeypatch):
    seq = [_traj(tests_green=False, action="redelegate", subgoal="补 os.fsync"),
           _traj(tests_green=True, action="accept")]
    calls = _patch_seq(monkeypatch, seq)
    traj = asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=3, max_cost_usd=1.0))
    assert traj.termination_reason == "accepted"
    assert len(traj.iterations) == 2
    assert traj.iterations[1].directive == "补 os.fsync"          # it2 记录了注入的 directive
    assert calls[1]["directive"] == "补 os.fsync"                  # it2 run_agent 收到 directive
    assert traj.cumulative_cost_usd == round(0.02, 6)
    assert traj.iterations[0].delegate_action == "redelegate"


def test_loop_accept_first_iteration(monkeypatch):
    _patch_always(monkeypatch, _traj(tests_green=True, action="accept"))
    traj = asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=3, max_cost_usd=1.0))
    assert traj.termination_reason == "accepted"
    assert len(traj.iterations) == 1                              # 不进 it2


def test_loop_max_iterations(monkeypatch):
    _patch_always(monkeypatch, _traj(tests_green=False, action="redelegate", subgoal="x"))
    traj = asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=2, max_cost_usd=1.0))
    assert traj.termination_reason == "max_iterations"
    assert len(traj.iterations) == 2


def test_loop_budget_exceeded(monkeypatch):
    _patch_always(monkeypatch, _traj(tests_green=False, action="redelegate", subgoal="x", cost=0.01))
    traj = asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=5, max_cost_usd=0.005))
    assert traj.termination_reason == "budget_exceeded"          # it1 后 cumulative(0.01)≥0.005
    assert len(traj.iterations) == 1


def test_loop_budget_does_not_mask_accept(monkeypatch):
    # fix 3:it1 花超预算但 action=accept → term=accepted(budget 不 mask 终态判定)
    _patch_always(monkeypatch, _traj(tests_green=True, action="accept", cost=0.06))
    traj = asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=3, max_cost_usd=0.05))
    assert traj.termination_reason == "accepted"
    assert len(traj.iterations) == 1


def test_loop_delegate_no_subgoal(monkeypatch):
    _patch_always(monkeypatch, _traj(tests_green=False, action="redelegate", subgoal=""))
    traj = asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=3, max_cost_usd=1.0))
    assert traj.termination_reason == "delegate_no_subgoal"
    assert len(traj.iterations) == 1


def test_loop_delegate_failed_action_none(monkeypatch):
    # delegate 步失败 → run_agent 兜成 {error, action:None};外环 action is None → delegate_failed
    _patch_always(monkeypatch, _traj(tests_green=False, action=None))
    traj = asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=3, max_cost_usd=1.0))
    assert traj.termination_reason == "delegate_failed"


def test_loop_escalate_terminal_no_rerun(monkeypatch):
    # escalate(测试过+weak)→ accepted_weak_test,不重跑
    _patch_always(monkeypatch, _traj(tests_green=True, action="escalate"))
    traj = asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=3, max_cost_usd=1.0))
    assert traj.termination_reason == "accepted_weak_test"
    assert len(traj.iterations) == 1


def test_loop_policy_drift_redelegate_on_passed_test_no_rerun(monkeypatch):
    # I10: tests_green=True 但 delegate 返 redelegate(policy drift)→ 不重跑,inconsistent_delegate
    _patch_always(monkeypatch, _traj(tests_green=True, action="redelegate", subgoal="x"))
    traj = asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=3, max_cost_usd=1.0))
    assert traj.termination_reason == "inconsistent_delegate"
    assert len(traj.iterations) == 1                              # 没重跑


def test_loop_give_up(monkeypatch):
    _patch_always(monkeypatch, _traj(tests_green=False, action="give_up"))
    traj = asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=3, max_cost_usd=1.0))
    assert traj.termination_reason == "give_up"


# ---------------- C1 / I8 ----------------
def test_loop_max_iterations_zero_no_crash():
    # C1: max_iterations≤0 → stub,不崩(不引用未定义 traj)
    traj = asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=0, max_cost_usd=1.0))
    assert traj.termination_reason == "max_iterations"
    assert traj.iterations == []
    assert traj.cumulative_cost_usd == 0.0


def test_loop_inner_exception_isolated(monkeypatch):
    # I8: 内层 run_agent 抛异常 → term=error,返已累积(无迭代)
    async def boom(*a, **kw):
        raise RuntimeError("inner explosion")
    monkeypatch.setattr(loop, "run_agent", boom)
    traj = asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=3, max_cost_usd=1.0))
    assert traj.termination_reason == "error"
    assert traj.iterations == []                                  # it1 异常,未 append


def test_loop_inner_exception_after_iteration(monkeypatch):
    # it1 成功(redelegate)→ it2 异常 → term=error,iterations 含 it1
    seq = [_traj(tests_green=False, action="redelegate", subgoal="x")]
    state = {"i": 0}

    async def fake(*a, **kw):
        state["i"] += 1
        if state["i"] == 1:
            return seq[0]
        raise RuntimeError("it2 boom")

    monkeypatch.setattr(loop, "run_agent", fake)
    traj = asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=3, max_cost_usd=1.0))
    assert traj.termination_reason == "error"
    assert len(traj.iterations) == 1                              # it1 已 append


# ---------------- I4: 内层传剩余预算 ----------------
def test_loop_inner_gets_remaining_budget(monkeypatch):
    seq = [_traj(tests_green=False, action="redelegate", subgoal="x", cost=0.03),
           _traj(tests_green=True, action="accept", cost=0.01)]
    calls = _patch_seq(monkeypatch, seq)
    asyncio.run(run_cognitive_loop(None, "m", _task(), run_dir=".", max_iterations=3, max_cost_usd=0.1))
    assert calls[0]["max_cost_usd"] == 0.1                        # it1: 全额
    assert abs(calls[1]["max_cost_usd"] - 0.07) < 1e-9            # it2: 剩余 0.1-0.03


# ---------------- Trajectory 序列化 ----------------
def test_trajectory_iterations_serialization():
    t = Trajectory(task_id="t", arm="none")
    assert t.iterations is None and t.cumulative_cost_usd is None
    d = t.to_dict()
    assert d["iterations"] is None and d["cumulative_cost_usd"] is None
    t.iterations = [CognitiveIteration(iteration=0, directive="", solve_status="tests_fail",
                                       tests_green=False, n_steps=3, cost_usd=0.01,
                                       delegate_action="redelegate", next_subgoal="补 X")]
    t.cumulative_cost_usd = 0.03
    out = t.to_dict()
    assert out["iterations"][0]["next_subgoal"] == "补 X"
    assert out["iterations"][0]["delegate_action"] == "redelegate"
    assert out["cumulative_cost_usd"] == 0.03
    # total_main_cost_usd 不被外环覆盖(语义保留)—— 这里未设外环,仍为默认
    assert out["total_main_cost_usd"] == 0.0


# ---------------- CLI 分支(_run_expert)----------------
def _fake_args(**over):
    base = dict(arm="none", max_steps=None, max_cost_usd=None, max_tokens=None,
                effort=None, thinking="off", brain_verify=False, brain_delegate=False,
                brain_loop=False, max_iterations=3)
    base.update(over)
    return SimpleNamespace(**base)


def test_run_expert_default_calls_run_agent(monkeypatch):
    called = {}

    async def fake_agent(*a, **kw):
        called["agent"] = kw
        return Trajectory(task_id="t", arm="none")

    async def fake_loop(*a, **kw):
        called["loop"] = kw
        return Trajectory(task_id="t", arm="none")

    # _run_expert 调 sandbox_cli 命名空间里的 run_agent/run_cognitive_loop(import 名),非 loop.*
    monkeypatch.setattr(sandbox_cli, "run_agent", fake_agent)
    monkeypatch.setattr(sandbox_cli, "run_cognitive_loop", fake_loop)
    asyncio.run(sandbox_cli._run_expert(_fake_args(), None, "m", _task(), ".", {}, None))
    assert "agent" in called and "loop" not in called


def test_run_expert_brain_loop_calls_cognitive_loop(monkeypatch):
    called = {}

    async def fake_agent(*a, **kw):
        called["agent"] = kw
        return Trajectory(task_id="t", arm="none")

    async def fake_loop(*a, **kw):
        called["loop"] = kw
        return Trajectory(task_id="t", arm="none")

    monkeypatch.setattr(sandbox_cli, "run_agent", fake_agent)
    monkeypatch.setattr(sandbox_cli, "run_cognitive_loop", fake_loop)
    asyncio.run(sandbox_cli._run_expert(
        _fake_args(brain_loop=True, max_iterations=5), None, "m", _task(), ".", {}, None))
    assert "loop" in called and "agent" not in called
    assert called["loop"]["max_iterations"] == 5
    # run_cognitive_loop 不吃 brain_verify/brain_delegate(强制 True)→ 不在 kw
    assert "brain_verify" not in called["loop"]
    assert "brain_delegate" not in called["loop"]


def test_run_expert_brain_loop_threads_python_exe(monkeypatch):
    called = {}

    async def fake_loop(*a, **kw):
        called["loop"] = kw
        return Trajectory(task_id="t", arm="none")

    monkeypatch.setattr(sandbox_cli, "run_cognitive_loop", fake_loop)
    asyncio.run(sandbox_cli._run_expert(
        _fake_args(brain_loop=True), None, "m", _task(), ".", {}, None, python_exe="/py"))
    assert called["loop"]["python_exe"] == "/py"


def test_run_expert_max_iterations_zero_not_silently_three(monkeypatch):
    # fix 2:--max-iterations 0 应传 0(走 stub),不该被 `0 or 3` 静默变 3
    called = {}

    async def fake_loop(*a, **kw):
        called["loop"] = kw
        return Trajectory(task_id="t", arm="none")

    monkeypatch.setattr(sandbox_cli, "run_cognitive_loop", fake_loop)
    asyncio.run(sandbox_cli._run_expert(
        _fake_args(brain_loop=True, max_iterations=0), None, "m", _task(), ".", {}, None))
    assert called["loop"]["max_iterations"] == 0
