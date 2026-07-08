"""env-loop 注入(Phase A)测试 —— 验证「不复制 loop」:observe/act 作 tool 进现有 dispatch_tool,
run_agent 经 system_prompt/verify_fn 注入 + scoped_env ContextVar 驱动 env。

覆盖 review 双强(2026-07-08)硬化:dispatch None-env 显式报错不崩(consensus high)、
act action 净化(opus high)、并发/嵌套 scoped_env 隔离(consensus medium)、
code-regime 幻觉 observe/act 被拒不崩、system_prompt 注入跳过默认、code-regime prompt 不泄漏 observe/act(opus #4)。
"""
from __future__ import annotations

import asyncio
import json
import sys

import pytest

from brainregion.providers.base import ModelResponse
from brainregion.sandbox import cleanup_run_dir, make_run_dir
from brainregion.sandbox.envs import GridWorld, build_env_system_prompt
from brainregion.sandbox.loop import (
    _build_system_prompt,
    _current_env,
    dispatch_tool,
    run_agent,
    scoped_env,
    ToolCall,
)
from brainregion.sandbox.task import SandboxTask


# ---------- helpers ----------


def _J(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


class MockBackend:
    """按脚本返 tool-call;不调模型。镜像 test_sandbox.MockBackend。"""

    def __init__(self, script, cost=0.001):
        self.script = script
        self.i = 0
        self.cost = cost

    async def complete_messages(self, messages, **kw):
        content = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return ModelResponse(model=kw.get("model", "mock"), content=content, usage={}, cost_usd=self.cost)


def _make_env_verify(env):
    """env-grounded verify_fn(返完整 verify_solution shape;tests_green := env.solved)。"""
    def verify(task, run_dir, *, python_exe=None):
        return {
            "tests_green": bool(env.solved),
            "solve_status": "solved" if env.solved else "tests_fail",
            "pytest": None,
            "gold_diff": getattr(task, "gold_diff", ""),
        }
    return verify


def _tc(tool: str, args: dict | None = None) -> ToolCall:
    return ToolCall(thought="", tool=tool, args=args or {}, done=False, answer="")


# ---------- dispatch:None-env 显式报错(consensus high)+ action 净化(opus high) ----------


def test_dispatch_observe_without_env_errors_not_crash():
    """code-regime(_current_env None)下 observe → 明确错误,不抛 AttributeError 崩 loop。"""
    assert _current_env.get() is None
    result, error = dispatch_tool(_tc("observe"))
    assert result == ""
    assert "当前无 env" in error and "RuntimeError" in error


def test_dispatch_act_without_env_errors_not_crash():
    result, error = dispatch_tool(_tc("act", {"action": "right"}))
    assert result == ""
    assert "当前无 env" in error


def test_dispatch_observe_with_env_returns_render():
    env = GridWorld(size=3, start=(0, 0), goal=(2, 2))
    with scoped_env(env):
        result, error = dispatch_tool(_tc("observe"))
    assert error is None
    payload = json.loads(result)
    assert "@" in payload["observation"]  # agent 标记在渲染里


def test_dispatch_act_with_env_moves_and_emits():
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0))  # 一步达
    with scoped_env(env):
        result, error = dispatch_tool(_tc("act", {"action": "right"}))
    assert error is None
    payload = json.loads(result)
    assert payload["reward"] == 1.0 and payload["terminated"] is True and payload["solved"] is True
    assert env.solved is True


def test_dispatch_act_invalid_action_rejected_before_env_step():
    """opus high:非法 action 在 dispatch 净化拦下,不进 env.step(模型可输出任意串)。"""
    env = GridWorld(size=3, start=(0, 0), goal=(2, 2))
    agent_before = env._agent
    with scoped_env(env):
        result, error = dispatch_tool(_tc("act", {"action": "fly"}))
    assert "非法 action" in error and "ValueError" in error
    assert env._agent == agent_before  # 未动


def test_dispatch_act_normalizes_case_and_whitespace():
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0))
    with scoped_env(env):
        result, error = dispatch_tool(_tc("act", {"action": "  RIGHT  "}))
    assert error is None
    assert json.loads(result)["reward"] == 1.0  # 归一化后合法 → 到达 goal


# ---------- scoped_env 嵌套/并发隔离(consensus medium) ----------


def test_scoped_env_nested_isolation_and_reset():
    a = GridWorld(size=3, start=(0, 0), goal=(2, 2))
    b = GridWorld(size=3, start=(0, 0), goal=(2, 2))
    assert _current_env.get() is None
    with scoped_env(a):
        assert _current_env.get() is a
        with scoped_env(b):
            assert _current_env.get() is b
        assert _current_env.get() is a
    assert _current_env.get() is None


def test_scoped_env_concurrent_no_cross_talk():
    """两个并发任务各持自己的 env,ContextVar 副本隔离,不串台。"""
    a = GridWorld(size=3, start=(0, 0), goal=(2, 2))
    b = GridWorld(size=3, start=(0, 0), goal=(2, 2))
    seen = {}

    async def hold(env, key):
        with scoped_env(env):
            await asyncio.sleep(0.005)
            seen[key] = _current_env.get() is env

    async def main():  # gather 须在 running loop 内调用(3.14:外部调 gather 会 get_event_loop 崩)
        await asyncio.gather(hold(a, "a"), hold(b, "b"))

    asyncio.run(main())
    assert seen == {"a": True, "b": True}


# ---------- run_agent env 求解(MockBackend,确定)----------


def test_run_agent_env_solves_mockbackend():
    """复用 run_agent(arm=none)+ env 注入:scripted act→done,断言 solved/terminated/reward=1/末帧到 G。"""
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0))  # 一步达
    task = SandboxTask(id="env-solve", goal="到达目标 G")
    backend = MockBackend([
        _J({"thought": "向右一步到目标", "tool": "act", "args": {"action": "right"}}),
        _J({"thought": "到达", "done": True, "answer": "已到目标 G"}),
    ])
    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            traj = asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm="none", max_steps=5,
                system_prompt=build_env_system_prompt(env, task.goal),
                verify_fn=_make_env_verify(env),
            ))
    finally:
        cleanup_run_dir(run_dir)
    # grounding(env.solved)+ 终态 + reward
    assert traj.tests_green is True
    assert traj.solve_status == "solved"
    assert traj.termination_reason == "done"
    assert env.solved is True
    assert env.total_reward == 1.0
    # 末帧 agent 在 goal 位(1,0)→ row0 col1 == '@'
    assert env.frames[-1].split("\n")[0][1] == "@"
    assert traj.n_steps == 2


def test_env_verify_fn_returns_full_shape():
    """verify_fn env 路径返完整 verify_solution shape(防下游 KeyError;opus #10)。"""
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0))
    env.step("right")  # solved
    task = SandboxTask(id="x", goal="g", gold_diff="diff")
    verify = _make_env_verify(env)
    out = verify(task, run_dir=".", python_exe=None)
    assert set(out.keys()) == {"tests_green", "solve_status", "pytest", "gold_diff"}
    assert out["tests_green"] is True and out["solve_status"] == "solved" and out["gold_diff"] == "diff"


# ---------- system_prompt 注入 + code-regime 不泄漏(opus #4)----------


def test_system_prompt_injected_skips_default(monkeypatch):
    """system_prompt 非 None → 不调 _build_system_prompt(直接用注入 prompt)。"""
    calls = {"n": 0}

    def spy(task, python_exe):
        calls["n"] += 1
        return "DEFAULT"

    monkeypatch.setattr("brainregion.sandbox.loop._build_system_prompt", spy)
    task = SandboxTask(id="x", goal="g")
    backend = MockBackend([_J({"thought": "done", "done": True, "answer": "x"})])
    run_dir = make_run_dir()
    try:
        with scoped_env(GridWorld(size=3, goal=(2, 2))):
            asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm="none", max_steps=3,
                system_prompt="CUSTOM", verify_fn=_make_env_verify(GridWorld(size=3, goal=(2, 2))),
            ))
    finally:
        cleanup_run_dir(run_dir)
    assert calls["n"] == 0  # 注入了 → 默认 prompt 未调


def test_code_regime_system_prompt_does_not_leak_env_tools():
    """opus #4:code-regime system prompt 不列 observe/act(code agent 不知其存在)。"""
    task = SandboxTask(id="x", goal="修 bug")
    prompt = _build_system_prompt(task, sys.executable)
    assert "observe" not in prompt  # env 专属工具不泄漏
    assert "read_text" in prompt    # code 工具仍在(tool 列表快照不变)


# ---------- Phase B review 硬化:act 非法输入 / already_done 跳过 emit / CLI argparse ----------


def test_dispatch_act_none_empty_nonstr_graceful():
    """review opus(B2):act None/空串/非 str/缺 key → 优雅错误,不崩(strip/lower 前已 isinstance 守卫)。"""
    env = GridWorld(size=3, start=(0, 0), goal=(2, 2))
    with scoped_env(env):
        for bad in (None, "", 123, ["right"], {"k": "v"}):
            _, error = dispatch_tool(_tc("act", {"action": bad}))
            assert error and ("action" in error or "非法" in error or "must be a string" in error), bad
        _, error = dispatch_tool(_tc("act", {}))  # 缺 action key
        assert error and "action" in error


def test_dispatch_act_already_done_skips_emit(monkeypatch):
    """review opus(B5):terminal 后冗余 act(already_done)不重复发 env.step 事件。"""
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0))
    emits: list[dict] = []
    monkeypatch.setattr(
        "brainregion.sandbox.loop.emit_event",
        lambda event_type, **kw: emits.append({"type": event_type, "kw": kw}),
    )
    with scoped_env(env):
        dispatch_tool(_tc("act", {"action": "right"}))  # reach goal → emit 1
        dispatch_tool(_tc("act", {"action": "right"}))  # already_done → skip emit
    env_events = [e for e in emits if e["type"] == "env.step"]
    assert len(env_events) == 1  # 只有 goal 那次,already_done 不发


def test_sandbox_env_cli_argparse():
    """review gpt(B4):sandbox env 子命令 argparse —— fog/random-goal/seed/visibility-radius/默认。"""
    from brainregion.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args([
        "sandbox", "env", "--env", "gridworld", "--size", "6",
        "--fog", "--random-goal", "--seed", "5", "--visibility-radius", "3",
        "--main-brain", "deepseek-v4-flash", "--debug",
    ])
    assert ns.command == "sandbox" and ns.sandbox_command == "env"
    assert ns.fog is True and ns.random_goal is True
    assert ns.seed == 5 and ns.size == 6 and ns.visibility_radius == 3
    assert ns.debug is True and ns.main_brain == "deepseek-v4-flash"

    ns2 = parser.parse_args(["sandbox", "env", "--main-brain", "glm-5.2"])
    assert ns2.fog is False and ns2.visibility_radius is None  # 默认全可见(Phase A 回归)
