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
    CODE_REGIME_TOOLS,
    _build_system_prompt,
    _current_env,
    _memory_mode,
    dispatch_tool,
    run_agent,
    scoped_env,
    scoped_memory_mode,
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
        self.last_messages = None

    async def complete_messages(self, messages, **kw):
        self.last_messages = [dict(m) for m in messages]
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


def test_env_action_budget_does_not_charge_observe_or_recall():
    """主脑轮次与环境动作预算分离:observe/recall 占 turn,不挤掉 2 次 act 机会。"""
    env = GridWorld(
        size=5, start=(0, 0), goal=(4, 4),
        visibility_radius=1, strict_obs=True,
    )
    task = SandboxTask(id="env-budget", goal="探索")
    backend = MockBackend([
        _J({"thought": "看", "tool": "observe", "args": {}}),
        _J({"thought": "回忆", "tool": "recall_map", "args": {}}),
        _J({"thought": "走1", "tool": "act", "args": {"action": "right"}}),
        _J({"thought": "再回忆", "tool": "recall_map", "args": {}}),
        _J({"thought": "走2", "tool": "act", "args": {"action": "right"}}),
    ])
    run_dir = make_run_dir()
    try:
        with scoped_env(env), scoped_memory_mode():
            traj = asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm="none",
                max_steps=10, max_env_actions=2,
                system_prompt=build_env_system_prompt(env, task.goal, memory=True),
                verify_fn=_make_env_verify(env),
            ))
    finally:
        cleanup_run_dir(run_dir)

    assert traj.n_steps == 5 and traj.env_actions == 2
    assert traj.successful_moves == 2 and env._agent == (2, 0)
    assert traj.region_tool_calls == 2 and traj.region_model_calls == 0
    assert traj.termination_reason == "env_action_budget"
    assert traj.to_dict()["main_turns"] == 5


def test_env_action_metrics_count_turn_block_and_move():
    """ego 原始动作口径:撞墙、转向、成功 forward 都占动作额度,但分类分开。"""
    env = GridWorld(
        size=4, start=(0, 0), goal=(3, 3), walls=((1, 0),),
        visibility_radius=1, strict_obs=True, ego_actions=True,
    )
    task = SandboxTask(id="env-ego-budget", goal="探索")
    backend = MockBackend([
        _J({"thought": "前方撞墙", "tool": "act", "args": {"action": "forward"}}),
        _J({"thought": "右转", "tool": "act", "args": {"action": "turn_right"}}),
        _J({"thought": "向南前进", "tool": "act", "args": {"action": "forward"}}),
    ])
    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            traj = asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm="none",
                max_steps=8, max_env_actions=3,
                system_prompt=build_env_system_prompt(env, task.goal),
                verify_fn=_make_env_verify(env),
            ))
    finally:
        cleanup_run_dir(run_dir)

    assert traj.env_actions == 3
    assert traj.blocked_actions == 1
    assert traj.turn_actions == 1
    assert traj.successful_moves == 1
    assert env._agent == (0, 1)


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


def test_code_regime_prompt_exposes_only_actor_owned_tools():
    task = SandboxTask(id="owned-evidence", goal="修 bug")
    prompt = _build_system_prompt(
        task,
        sys.executable,
        allowed_tools=CODE_REGIME_TOOLS - {"read_text", "search_text"},
    )

    assert "- read_text(" not in prompt
    assert "- search_text(" not in prompt
    assert "- apply_text_patch(" in prompt
    assert "- workspace_run_check(" in prompt
    assert "<region_workbench>" in prompt
    assert "不要重复请求已委派的读取或搜索" in prompt


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

    ns3 = parser.parse_args(["sandbox", "env", "--memory", "--size", "8", "--debug-port", "9000"])
    assert ns3.memory is True and ns3.size == 8 and ns3.debug_port == 9000

    ns4 = parser.parse_args([
        "sandbox", "env-eval", "--max-steps", "20", "--max-main-turns", "90",
    ])
    assert ns4.max_steps == 20 and ns4.max_main_turns == 90

    ns5 = parser.parse_args([
        "sandbox", "env", "--env", "urban-delivery", "--size", "13",
        "--orders", "4", "--vehicles", "3", "--seed", "9", "--main-brain", "sonnet",
    ])
    assert ns5.env == "urban-delivery" and ns5.size == 13
    assert ns5.orders == 4 and ns5.vehicles == 3 and ns5.seed == 9


# ---------- Phase C 记忆脑区:recall_map + strict observation ----------


def test_recall_map_without_memory_mode_errors():
    """recall_map 在 _memory_mode False(默认)→ 显式错误,不崩(镜像 observe/act None-guard)。"""
    env = GridWorld(size=4, start=(0, 0), goal=(3, 3), visibility_radius=1)
    assert _memory_mode.get() is False
    with scoped_env(env):
        result, error = dispatch_tool(_tc("recall_map"))
    assert result == ""
    assert "记忆脑区未激活" in error and "RuntimeError" in error


def test_recall_map_returns_accumulated_map():
    """memory 模式:recall_map 返累积探索图(render)+ explored 计数。"""
    env = GridWorld(size=4, start=(0, 0), goal=(3, 3), visibility_radius=1, strict_obs=True)
    env.step("right")  # 探索扩
    with scoped_env(env), scoped_memory_mode():
        result, error = dispatch_tool(_tc("recall_map"))
    assert error is None
    payload = json.loads(result)
    assert "map" in payload and payload["explored_cells"] >= 1 and payload["of_total"] == 16
    assert "?" in payload["map"]  # 累积图里仍有未探索格


def test_recall_map_reflects_exploration_growth():
    """recall_map 的 explored_cells 随 agent 探索增长(记忆脑区跟随 env 状态)。"""
    env = GridWorld(size=5, start=(0, 0), goal=(4, 4), visibility_radius=1, strict_obs=True)
    with scoped_env(env), scoped_memory_mode():
        _, _ = dispatch_tool(_tc("recall_map"))
        n0 = json.loads(dispatch_tool(_tc("recall_map"))[0])["explored_cells"]
        dispatch_tool(_tc("act", {"action": "right"}))
        n1 = json.loads(dispatch_tool(_tc("recall_map"))[0])["explored_cells"]
    assert n1 > n0  # 移动后探索域增长


def test_scoped_memory_mode_nesting_and_reset():
    assert _memory_mode.get() is False
    with scoped_memory_mode():
        assert _memory_mode.get() is True
        with scoped_memory_mode():  # 嵌套
            assert _memory_mode.get() is True
        assert _memory_mode.get() is True
    assert _memory_mode.get() is False


def test_code_regime_prompt_does_not_leak_recall_map():
    """code-regime system prompt 不列 recall_map(code agent 不知其存在;同 observe/act 隔离)。"""
    task = SandboxTask(id="x", goal="修 bug")
    prompt = _build_system_prompt(task, sys.executable)
    assert "recall_map" not in prompt


def test_recall_map_hallucination_in_code_regime_graceful():
    """code-regime(_memory_mode False)幻觉调 recall_map → 优雅错误,不崩不 leak memory。"""
    assert _memory_mode.get() is False
    result, error = dispatch_tool(_tc("recall_map"))  # 无 scoped_env、无 memory_mode
    assert result == "" and "记忆脑区未激活" in error


def test_run_agent_memory_loop_mockbackend():
    """MockBackend memory-loop(确定性):observe(当前视野)→ recall_map(累积图)→ act→ done。
    断言 strict observe 返当前视野、recall_map 返累积图、trajectory 含 recall_map、solved。"""
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0), visibility_radius=1, strict_obs=True)
    task = SandboxTask(id="env-mem", goal="找到 G")
    backend = MockBackend([
        _J({"thought": "看当前视野", "tool": "observe", "args": {}}),
        _J({"thought": "查记忆地图", "tool": "recall_map", "args": {}}),
        _J({"thought": "移动到目标", "tool": "act", "args": {"action": "right"}}),
        _J({"thought": "到达", "done": True, "answer": "到 G"}),
    ])
    run_dir = make_run_dir()
    try:
        with scoped_env(env), scoped_memory_mode():
            traj = asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm="none", max_steps=6,
                system_prompt=build_env_system_prompt(env, task.goal, memory=True),
                verify_fn=_make_env_verify(env),
            ))
    finally:
        cleanup_run_dir(run_dir)
    tools_called = [s.tool for s in traj.steps]
    assert "recall_map" in tools_called  # agent 确调了记忆脑区
    recall_step = next(s for s in traj.steps if s.tool == "recall_map")
    assert '"map"' in recall_step.result_preview  # 返累积图
    observe_step = next(s for s in traj.steps if s.tool == "observe")
    assert '"observation"' in observe_step.result_preview
    assert traj.tests_green is True and env.solved is True  # grounded


# ---------- Phase 4.9 导航执行委托 ----------


def test_delegate_navigation_executes_dfs_with_actor_trace():
    """导航脑区走入死路后正确回溯,主脑只调用一次工具；每个原始动作保留 actor provenance。"""
    from brainregion.sandbox.regions import NavigationRegion

    env = GridWorld(
        size=4, start=(0, 0), goal=(0, 2), walls=((2, 0), (1, 1)),
        visibility_radius=1, strict_obs=True,
    )
    task = SandboxTask(id="env-nav-delegate", goal="找到 G")
    backend = MockBackend([
        _J({"thought": "委托局部探索", "tool": "delegate_navigation", "args": {"action_budget": 8}}),
        _J({"thought": "轨迹显示已到达", "done": True, "answer": "到 G"}),
    ])
    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            traj = asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm="none",
                max_steps=4, max_env_actions=8,
                system_prompt=build_env_system_prompt(env, task.goal, navigation=True),
                verify_fn=_make_env_verify(env),
                navigation_region=NavigationRegion(start=env.start),
            ))
    finally:
        cleanup_run_dir(run_dir)

    assert env.solved is True and traj.tests_green is True
    assert [x["action"] for x in traj.env_action_trace] == ["right", "left", "down", "down"]
    assert all(x["actor"] == "navigation_region" for x in traj.env_action_trace)
    assert traj.env_actions == 4 and traj.delegated_actions == 4
    assert traj.successful_moves == 4 and traj.blocked_actions == 0
    assert traj.region_tool_calls == 1 and traj.region_model_calls == 0
    assert traj.navigation_delegations == 1 and traj.automatic_region_activations == 0
    assert '"actor": "navigation_region"' in traj.steps[0].result_preview


def test_delegate_navigation_respects_global_env_action_budget():
    from brainregion.sandbox.regions import NavigationRegion

    env = GridWorld(size=4, start=(0, 0), goal=(3, 3), visibility_radius=1, strict_obs=True)
    task = SandboxTask(id="env-nav-budget", goal="找到 G")
    backend = MockBackend([
        _J({"thought": "委托", "tool": "delegate_navigation", "args": {"action_budget": 16}}),
    ])
    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            traj = asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm="none",
                max_steps=3, max_env_actions=1,
                system_prompt=build_env_system_prompt(env, task.goal, navigation=True),
                verify_fn=_make_env_verify(env),
                navigation_region=NavigationRegion(start=env.start),
            ))
    finally:
        cleanup_run_dir(run_dir)

    assert traj.env_actions == 1 and traj.delegated_actions == 1
    assert traj.termination_reason == "env_action_budget"


def test_delegate_navigation_without_region_is_graceful():
    result, error = dispatch_tool(_tc("delegate_navigation", {"action_budget": 2}))
    assert result == "" and "导航执行脑区未激活" in error


def test_navigation_autorun_executes_before_main_and_injects_attributed_trace():
    from brainregion.sandbox.regions import NavigationRegion

    env = GridWorld(size=3, start=(0, 0), goal=(2, 0), visibility_radius=1, strict_obs=True)
    task = SandboxTask(id="env-nav-autorun", goal="找到 G")
    backend = MockBackend([_J({"thought": "脑区已经到达", "done": True, "answer": "到 G"})])
    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            traj = asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm="none",
                max_steps=3, max_env_actions=8,
                system_prompt=build_env_system_prompt(env, task.goal, navigation=True),
                verify_fn=_make_env_verify(env),
                navigation_region=NavigationRegion(start=env.start),
                navigation_autorun_actions=8,
            ))
    finally:
        cleanup_run_dir(run_dir)

    assert env.solved is True and traj.delegated_actions == 2
    assert traj.navigation_delegations == 1 and traj.automatic_region_activations == 1
    assert traj.region_tool_calls == 0  # automatic activation is not a main-model tool call
    injected = "\n".join(m["content"] for m in backend.last_messages if m["role"] == "user")
    assert '<region_execution actor="navigation_region"' in injected
    assert "不是你亲自执行的动作" in injected


def test_grounded_navigation_autorun_solves_dead_end_without_env_access():
    """Grounded policy only sees text observations yet can backtrack out of a visible dead end."""
    from brainregion.sandbox.regions import GroundedNavigationRegion

    class TextOnlyRegion(GroundedNavigationRegion):
        def next_action(self, observation):
            assert isinstance(observation, str)
            return super().next_action(observation)

    env = GridWorld(
        size=4, start=(0, 0), goal=(0, 2), walls=((2, 0), (1, 1)),
        visibility_radius=1, strict_obs=True,
    )
    task = SandboxTask(id="env-nav-grounded", goal="找到 G")
    backend = MockBackend([_J({"thought": "脑区已完成", "done": True, "answer": "到 G"})])
    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            traj = asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm="none",
                max_steps=3, max_env_actions=8,
                system_prompt=build_env_system_prompt(env, task.goal, navigation=True),
                verify_fn=_make_env_verify(env),
                option_region=TextOnlyRegion(), option_autorun_actions=8,
            ))
    finally:
        cleanup_run_dir(run_dir)

    assert env.solved is True
    assert [x["action"] for x in traj.env_action_trace] == ["right", "left", "down", "down"]
    assert all(x["actor"] == "navigation_region" for x in traj.env_action_trace)


def test_grounded_navigation_yields_at_junction():
    from brainregion.sandbox.regions import GroundedNavigationRegion

    env = GridWorld(size=5, start=(0, 0), goal=(4, 4), visibility_radius=1, strict_obs=True)
    task = SandboxTask(id="env-nav-junction", goal="找到 G")
    backend = MockBackend([_J({"thought": "接管后续决策", "done": True, "answer": "暂停"})])
    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            traj = asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm="none",
                max_steps=2, max_env_actions=8,
                system_prompt=build_env_system_prompt(env, task.goal, navigation=True),
                verify_fn=_make_env_verify(env),
                navigation_region=GroundedNavigationRegion(), navigation_autorun_actions=8,
            ))
    finally:
        cleanup_run_dir(run_dir)

    assert traj.delegated_actions == 1
    injected = "\n".join(m["content"] for m in backend.last_messages if m["role"] == "user")
    assert '"stop_reason": "decision_boundary:junction"' in injected


def test_grounded_navigation_does_not_store_hidden_cells():
    from brainregion.sandbox.regions import GroundedNavigationRegion

    env = GridWorld(size=5, start=(0, 0), goal=(4, 4), visibility_radius=1, strict_obs=True)
    region = GroundedNavigationRegion()
    region.next_action(env.observation())
    assert (4, 4) not in region.known
    assert set(region.known) == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_continuous_navigation_reactivates_only_after_main_action():
    from brainregion.sandbox.regions import GroundedNavigationRegion

    env = GridWorld(size=5, start=(0, 0), goal=(4, 4), visibility_radius=1, strict_obs=True)
    task = SandboxTask(id="env-nav-continuous", goal="找到 G")
    backend = MockBackend([
        _J({"thought": "在岔路选择向下", "tool": "act", "args": {"action": "down"}}),
        _J({"thought": "观察第二次交权", "done": True, "answer": "暂停"}),
    ])
    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            traj = asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm="none",
                max_steps=3, max_env_actions=8,
                system_prompt=build_env_system_prompt(env, task.goal, navigation=True),
                verify_fn=_make_env_verify(env),
                navigation_region=GroundedNavigationRegion(), navigation_autorun_actions=8,
                navigation_continuous=True,
            ))
    finally:
        cleanup_run_dir(run_dir)

    assert traj.automatic_region_activations == 2
    assert traj.delegated_actions == 2 and traj.env_actions == 3
    assert [item["trigger"] for item in traj.navigation_options] == ["initial", "after_main_action"]
    assert all(item["stop_reason"] == "decision_boundary:junction" for item in traj.navigation_options)
    assert [item["actor"] for item in traj.env_action_trace] == [
        "navigation_region", "main", "navigation_region",
    ]


def test_option_region_failure_isolated_and_reported_to_main():
    class BrokenRegion:
        name = "broken"
        access_mode = "grounded"

        def next_action(self, observation):
            raise RuntimeError("region exploded")

        def observe_transition(self, *, action, observation, status):
            pass

        def option_boundary(self, observation, *, actions_executed):
            return None

        def snapshot(self):
            return {}

    env = GridWorld(size=3, start=(0, 0), goal=(2, 2), visibility_radius=1, strict_obs=True)
    task = SandboxTask(id="env-option-failure", goal="找到 G")
    backend = MockBackend([_J({"thought": "脑区失败后继续", "done": True, "answer": "停止"})])
    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            traj = asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm="none",
                max_steps=2, max_env_actions=4,
                system_prompt=build_env_system_prompt(env, task.goal, navigation=True),
                verify_fn=_make_env_verify(env),
                navigation_region=BrokenRegion(), navigation_autorun_actions=2,
            ))
    finally:
        cleanup_run_dir(run_dir)

    assert traj.done is True and traj.n_steps == 1
    assert traj.automatic_region_activations == 0 and traj.navigation_options == []
    injected = "\n".join(m["content"] for m in backend.last_messages if m["role"] == "user")
    assert 'region_execution actor="broken_region" error="true"' in injected
    assert "region exploded" in injected


def test_option_region_and_legacy_navigation_region_are_mutually_exclusive():
    from brainregion.sandbox.regions import GroundedNavigationRegion, NavigationRegion

    env = GridWorld(size=3, start=(0, 0), goal=(2, 2), visibility_radius=1, strict_obs=True)
    task = SandboxTask(id="env-option-conflict", goal="找到 G")
    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            with pytest.raises(ValueError, match="cannot both be set"):
                asyncio.run(run_agent(
                    MockBackend([_J({"done": True})]), "mock", task,
                    run_dir=run_dir, max_steps=1,
                    system_prompt=build_env_system_prompt(env, task.goal, navigation=True),
                    verify_fn=_make_env_verify(env),
                    option_region=GroundedNavigationRegion(),
                    navigation_region=NavigationRegion(start=env.start),
                ))
    finally:
        cleanup_run_dir(run_dir)
