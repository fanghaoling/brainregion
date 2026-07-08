"""Phase D 记忆脑区(真 LLM,region-as-tool)测试。

覆盖 review 双强(2026-07-08)硬化:
- region.reason 返结构化 dict + 解析鲁棒(consensus/gpt 正确性);
- recall 失败/超预算/超 cap → 降级 Phase C(不崩主 run,定义 result_str)(consensus medium);
- positions/attempts 分轨(位置 delta 判 status;blocked/invalid/already_done 不入 positions)(gpt/opus medium);
- region 输出 no-advice(无动作指令)(gpt/opus 实验效度);
- region 臂 = 基线信息超集(raw map 永在 + interpretation);
- 组合 arm=brainregion + memory_region 不崩(opus medium)。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from brainregion.providers.base import ModelResponse
from brainregion.sandbox import cleanup_run_dir, make_run_dir
from brainregion.sandbox.envs import GridWorld, build_env_system_prompt
from brainregion.sandbox.loop import run_agent, scoped_env, scoped_memory_mode
from brainregion.sandbox.regions import MemoryRegion, build_memory_region_system_prompt
from brainregion.sandbox.regions.memory_region import _extract_interpretation
from brainregion.sandbox.task import SandboxTask


# ---------- helpers(镜像 test_sandbox_env_loop,避免跨模块耦合)----------


def _J(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


class MockBackend:
    """按脚本返 content;不调模型。region 调用也走此(按序消费一条脚本)。"""

    def __init__(self, script, cost=0.001):
        self.script = script
        self.i = 0
        self.cost = cost

    async def complete_messages(self, messages, **kw):
        content = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return ModelResponse(model=kw.get("model", "mock"), content=content, usage={}, cost_usd=self.cost)


def _make_env_verify(env):
    def verify(task, run_dir, *, python_exe=None):
        return {
            "tests_green": bool(env.solved),
            "solve_status": "solved" if env.solved else "tests_fail",
            "pytest": None,
            "gold_diff": getattr(task, "gold_diff", ""),
        }
    return verify


class _RecordingRegion:
    """假记忆脑区:记录每次 reason 收到的 positions/attempts,返 canned interpretation(不调 backend)。"""

    def __init__(self):
        self.seen = []

    async def reason(self, backend, model, **kw):
        self.seen.append({
            "positions": list(kw["positions"]),
            "attempts": list(kw["attempts"]),
            "spatial": kw.get("spatial", ""),
        })
        return {"interpretation": "current_position: (0,0)\ngoal_direction_estimate: 东", "cost_usd": 0.0, "ok": True}


_REGION_JSON = _J({
    "current_position": "(1,0)", "path_summary": "右走一步", "looping_detected": "否", "goal_direction_estimate": "东",
})


def _run(backend, env, *, memory_region=None, goal="找到 G", arm="none", max_steps=6, max_recall_calls=None):
    task = SandboxTask(id="env-region", goal=goal)
    run_dir = make_run_dir()
    try:
        with scoped_env(env), scoped_memory_mode():
            return asyncio.run(run_agent(
                backend, "mock", task, run_dir=run_dir, arm=arm, max_steps=max_steps,
                system_prompt=build_env_system_prompt(env, goal, memory=True), verify_fn=_make_env_verify(env),
                memory_region=memory_region, max_recall_calls=max_recall_calls,
            ))
    finally:
        cleanup_run_dir(run_dir)


# ---------- MemoryRegion 单元 ----------


def test_memory_region_reason_parses_interpretation():
    backend = MockBackend([_REGION_JSON])
    region = MemoryRegion()
    res = asyncio.run(region.reason(
        backend, "mock", spatial="@.G", positions=[(0, 0)], attempts=[], current_view="@.", query="",
    ))
    assert res["ok"] is True
    assert "current_position" in res["interpretation"] and "东" in res["interpretation"]
    assert res["cost_usd"] == 0.001


def test_memory_region_reason_raises_on_empty(monkeypatch):
    """backend 返空 content → reason 抛 RuntimeError(上层 _recall_via_region 兜底降级)。"""
    backend = MockBackend([""])
    region = MemoryRegion()
    with pytest.raises(RuntimeError, match="empty output|backend failed"):
        asyncio.run(region.reason(backend, "mock", spatial="@", positions=[], attempts=[], current_view="@", query=""))


def test_extract_interpretation_json_and_garbage():
    out = _extract_interpretation(_REGION_JSON)
    assert "current_position: (1,0)" in out and "goal_direction_estimate: 东" in out
    garbage = _extract_interpretation("完全不是 JSON 的一段话")
    assert garbage == "完全不是 JSON 的一段话"


def test_memory_region_prompt_is_no_advice():
    """review gpt/opus 实验效度:v1 no-advice —— 提示词明示不下动作指令;输出 schema 是记忆事实无 action/move 键。"""
    p = build_memory_region_system_prompt()
    assert "记忆脑区" in p
    assert "不下动作指令" in p
    # 输出 schema 四键均为记忆事实,无动作指令键
    for k in ("current_position", "path_summary", "looping_detected", "goal_direction_estimate"):
        assert k in p
    assert '"action"' not in p and '"move"' not in p.lower() or "不写" in p


# ---------- run_agent recall 拦截(region vs Phase C)----------


def test_recall_via_region_returns_interpretation_and_costs():
    """region 臂:recall_map → 记忆脑区 LLM → 结果含 interpretation + region;region cost 记账。"""
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0), visibility_radius=1, strict_obs=True)
    backend = MockBackend([
        _J({"thought": "看", "tool": "observe", "args": {}}),
        _J({"thought": "查记忆", "tool": "recall_map", "args": {}}),
        _REGION_JSON,  # region LLM 输出(被 _recall_via_region 消费)
        _J({"thought": "走", "tool": "act", "args": {"action": "right"}}),
        _J({"thought": "到", "done": True, "answer": "到 G"}),
    ])
    traj = _run(backend, env, memory_region=MemoryRegion())
    recall_step = next(s for s in traj.steps if s.tool == "recall_map")
    assert "interpretation" in recall_step.result_preview and '"region": true' in recall_step.result_preview
    # region cost(0.001)进了 main cost(observe/recall/act 3 主调 + region 1 = 0.004)
    assert traj.total_main_cost_usd >= 0.004
    assert env.solved is True


def test_recall_without_region_is_phase_c_regression():
    """无 memory_region → recall_map 走 dispatch(Phase C):结果 = map + explored_cells,无 interpretation。"""
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0), visibility_radius=1, strict_obs=True)
    backend = MockBackend([
        _J({"thought": "看", "tool": "observe", "args": {}}),
        _J({"thought": "查记忆", "tool": "recall_map", "args": {}}),
        _J({"thought": "走", "tool": "act", "args": {"action": "right"}}),
        _J({"thought": "到", "done": True, "answer": "到 G"}),
    ])
    traj = _run(backend, env, memory_region=None)
    recall_step = next(s for s in traj.steps if s.tool == "recall_map")
    assert "explored_cells" in recall_step.result_preview
    assert "interpretation" not in recall_step.result_preview  # Phase C 形状,无脑区解释
    assert env.solved is True


# ---------- 失败降级(consensus medium + gpt medium)----------


def test_recall_region_degrades_on_backend_failure():
    """region 调用失败(空 content→reason 抛)→ 降级 Phase C:result 有 region_degraded + map,无 interpretation;
    主 run 不崩,后续 act 仍可达 goal。"""
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0), visibility_radius=1, strict_obs=True)
    backend = MockBackend([
        _J({"thought": "看", "tool": "observe", "args": {}}),
        _J({"thought": "查记忆", "tool": "recall_map", "args": {}}),
        "",  # region LLM 空 → reason 抛 → 降级
        _J({"thought": "走", "tool": "act", "args": {"action": "right"}}),
        _J({"thought": "到", "done": True, "answer": "到 G"}),
    ])
    traj = _run(backend, env, memory_region=MemoryRegion())
    recall_step = next(s for s in traj.steps if s.tool == "recall_map")
    assert "region_degraded" in recall_step.result_preview
    assert "interpretation" not in recall_step.result_preview
    assert env.solved is True  # 降级未毁主 run


def test_recall_region_degrades_on_cap():
    """max_recall_calls=0 → 首次 recall 即超 cap 降级(budget_or_cap),不调 region backend。"""
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0), visibility_radius=1, strict_obs=True)
    backend = MockBackend([
        _J({"thought": "看", "tool": "observe", "args": {}}),
        _J({"thought": "查记忆", "tool": "recall_map", "args": {}}),
        # 无 region 脚本条目:cap 拦下不消费 backend
        _J({"thought": "走", "tool": "act", "args": {"action": "right"}}),
        _J({"thought": "到", "done": True, "answer": "到 G"}),
    ])
    traj = _run(backend, env, memory_region=MemoryRegion(), max_recall_calls=0)
    recall_step = next(s for s in traj.steps if s.tool == "recall_map")
    assert "budget_or_cap" in recall_step.result_preview
    assert "interpretation" not in recall_step.result_preview
    assert env.solved is True


# ---------- positions/attempts 分轨(gpt/opus medium)----------


def test_positions_attempts_split_blocked_vs_moved():
    """act 撞墙(blocked)不入 positions 但入 attempts;成功移动(moved)入两者。位置 delta 判 status。"""
    env = GridWorld(size=3, start=(0, 0), goal=(2, 2), walls=((1, 0),), visibility_radius=1, strict_obs=True)
    region = _RecordingRegion()
    backend = MockBackend([
        _J({"thought": "右(撞墙)", "tool": "act", "args": {"action": "right"}}),   # (1,0) 墙 → blocked
        _J({"thought": "下", "tool": "act", "args": {"action": "down"}}),          # (0,0)->(0,1) moved
        _J({"thought": "查记忆", "tool": "recall_map", "args": {}}),               # 触发 region(record)
        _J({"thought": "到", "done": True, "answer": "记下"}),
    ])
    _run(backend, env, memory_region=region)
    assert len(region.seen) == 1
    attempts = region.seen[0]["attempts"]
    assert attempts[0]["action"] == "right" and attempts[0]["status"] == "blocked"
    assert attempts[1]["action"] == "down" and attempts[1]["status"] == "moved"
    # positions 只含 start + moved 位(blocked 不入)
    assert region.seen[0]["positions"] == [(0, 0), (0, 1)]


# ---------- 组合:memory_region × BrainRegion arm(opus medium)----------


def test_memory_region_composes_with_brainregion_arm():
    """arm=brainregion(wake_gate/种子)与 memory_region(region 拦截/path 追踪)同启不崩,两者都活。"""
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0), visibility_radius=1, strict_obs=True)
    region = _RecordingRegion()
    backend = MockBackend([
        _J({"thought": "看", "tool": "observe", "args": {}}),
        _J({"thought": "查记忆", "tool": "recall_map", "args": {}}),
        _J({"thought": "走", "tool": "act", "args": {"action": "right"}}),
        _J({"thought": "到", "done": True, "answer": "到 G"}),
    ])
    traj = _run(backend, env, memory_region=region, arm="brainregion")
    assert len(region.seen) == 1  # region 确被调
    assert hasattr(traj, "wake_calls")  # arm 路径也跑了
    assert env.solved is True
