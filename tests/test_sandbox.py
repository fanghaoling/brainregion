"""沙盒(agent loop + BrainRegion 臂 + A/B)测试。

覆盖 plan 验证项:backend complete_messages + json_object 回退、materialize 路径校验、
verifier(tests-green 为主)、loop(solve/parse-error 早停/budget/未知工具/brainregion 臂)、
parse_tool_call 校验、scoped_workspace_root 收容 + 优先级、eval gate 决策。
loop 用 mock backend(不调真模型);verifier/isolation 真跑 pytest。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from brainregion.providers.base import ModelResponse
from brainregion.providers.litellm import LiteLLMBackend
from brainregion.sandbox import (
    materialize_fixture,
    make_run_dir,
    cleanup_run_dir,
    run_agent,
    verify_solution,
)
from brainregion.sandbox.eval import evaluate_gate, run_sandbox_eval
from brainregion.sandbox.fixtures import SANDBOX_FIXTURES, get_fixture
from brainregion.sandbox.loop import ALLOWED_TOOLS, parse_tool_call
from brainregion.sandbox.isolation import FixturePathError
from brainregion.sandbox.task import SandboxTask
from brainregion.workspace import apply_text_patch, read_text
from brainregion.workspace.files import scoped_workspace_root


# ---------- helpers ----------


def test_complete_messages_passes_history_through(monkeypatch):
    """complete_messages 把完整 messages 列表透传给 litellm.acompletion(跨步带历史)。"""
    import litellm

    captured = {}

    class _FakeResp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
            self.usage = None
            self._hidden_params = {}

    async def fake_acompletion(**kw):
        captured.update(kw)
        return _FakeResp('{"thought":"x","done":true}')

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    backend = LiteLLMBackend()
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]
    resp = asyncio.run(backend.complete_messages(msgs, model="some/model"))
    assert resp.ok
    assert captured["messages"] == msgs  # history passed through verbatim
    assert resp.content == '{"thought":"x","done":true}'


def test_complete_messages_json_object_fallback(monkeypatch):
    """provider 拒 response_format=json_object → 去掉 response_format 重试 + 仍解析成功。"""
    import litellm

    calls = []

    class _FakeResp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
            self.usage = None
            self._hidden_params = {}

    async def fake_acompletion(**kw):
        calls.append(kw.get("response_format"))
        if kw.get("response_format"):  # 第一次带 json_object → 模拟 provider 拒绝
            raise ValueError("response_format json_object not supported on this endpoint")
        return _FakeResp('{"done":true}')

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    backend = LiteLLMBackend()  # 默认 response_format=json_object
    resp = asyncio.run(backend.complete_messages([{"role": "user", "content": "hi"}], model="some/model"))
    assert resp.ok
    assert resp.content == '{"done":true}'
    assert len(calls) == 2  # 回退重试了一次
    assert calls[0] == {"type": "json_object"}
    assert calls[1] is None  # 第二次不带 response_format


def test_deepseek_thinking_kwargs():
    """_effort_kwargs / _sampling_for 对 deepseek 的思考模式控制(关=便宜快;开=reasoning_effort)。"""
    from brainregion.providers.litellm import LiteLLMBackend, _effort_kwargs

    # thinking off → disabled, 无 reasoning_effort
    assert _effort_kwargs("openai/deepseek-v4-flash", effort=None, thinking=False) == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }
    # thinking on + effort → enabled + reasoning_effort
    assert _effort_kwargs("openai/deepseek-v4-pro", effort="high", thinking=True) == {
        "extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": "high",
    }
    # thinking None(未显式)→ 保持原契约:effort 对 deepseek no-op(§15.6)
    assert _effort_kwargs("openai/deepseek-v4-flash", effort="high", thinking=None) == {}
    assert _effort_kwargs("openai/deepseek-v4-flash", effort=None, thinking=None) == {}
    # 非 deepseek 不受 thinking 影响
    assert _effort_kwargs("zhipu/glm-5.2", effort=None, thinking=False) == {}
    # sampling:deepseek 思考关(False)/默认(None) → 正常采样;显式开(True) → 不采样(文档:思考忽略 temp/top_p)
    assert LiteLLMBackend._sampling_for("openai/deepseek-v4-flash", 0.0, 0.95, None, False) == {"temperature": 0.0, "top_p": 0.95}
    assert LiteLLMBackend._sampling_for("openai/deepseek-v4-flash", 0.0, 0.95, None, None) == {"temperature": 0.0, "top_p": 0.95}
    assert LiteLLMBackend._sampling_for("openai/deepseek-v4-flash", 0.0, 0.95, None, True) == {}



class MockBackend:
    """按脚本返 tool-call;不调模型。"""

    def __init__(self, script, cost=0.001):
        self.script = script
        self.i = 0
        self.cost = cost

    async def complete_messages(self, messages, **kw):
        content = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return ModelResponse(model=kw.get("model", "mock"), content=content, usage={}, cost_usd=self.cost)


def _J(d):
    return json.dumps(d, ensure_ascii=False)


def _materialized(task_id):
    task = get_fixture(task_id)
    run_dir = make_run_dir()
    materialize_fixture(task, Path(run_dir))
    return task, run_dir


def _sha(run_dir, path):
    with scoped_workspace_root(run_dir):
        return read_text(path)["sha256"]


# ---------- materialize path validation ----------

def test_materialize_rejects_absolute_path(tmp_path):
    task = SandboxTask(id="x", goal="g", files={"abs": "x"})
    task = SandboxTask(id="x", goal="g", files={str(tmp_path / "evil.py"): "x"})
    with pytest.raises(FixturePathError, match="absolute"):
        materialize_fixture(task, tmp_path)


def test_materialize_rejects_parent_traversal(tmp_path):
    task = SandboxTask(id="x", goal="g", files={"../evil.py": "x"})
    with pytest.raises(FixturePathError, match="traversal|outside"):
        materialize_fixture(task, tmp_path)


def test_materialize_writes_files(tmp_path):
    task = SandboxTask(id="x", goal="g", files={"a/b.py": "print(1)"}, tests={"test_t.py": "x"})
    materialize_fixture(task, tmp_path)
    assert (tmp_path / "a" / "b.py").read_text(encoding="utf-8") == "print(1)"
    assert (tmp_path / "test_t.py").exists()


# ---------- verifier ----------

@pytest.mark.parametrize("task_id", [t.id for t in SANDBOX_FIXTURES])
def test_verifier_buggy_fails(task_id):
    task, run_dir = _materialized(task_id)
    try:
        v = verify_solution(task, run_dir)
        assert v["solve_status"] == "tests_fail"
        assert v["tests_green"] is False
    finally:
        cleanup_run_dir(run_dir)


def test_verifier_solved_when_green():
    task, run_dir = _materialized("off_by_one")
    try:
        with scoped_workspace_root(run_dir):
            sha = read_text("ranges.py")["sha256"]
            apply_text_patch(
                "ranges.py", expected_sha256=sha,
                replacements=[{"old_text": "range(start, end)", "new_text": "range(start, end + 1)"}],
                dry_run=False,
            )
        v = verify_solution(task, run_dir)
        assert v["solve_status"] == "solved"
        assert v["tests_green"] is True
        assert v["gold_diff"]  # diagnostic recorded
    finally:
        cleanup_run_dir(run_dir)


# ---------- parse_tool_call validation ----------

def test_parse_done():
    call, err = parse_tool_call(_J({"thought": "x", "done": True, "answer": "done"}))
    assert err is None and call.done and call.answer == "done"


def test_parse_tool():
    call, err = parse_tool_call(_J({"thought": "x", "tool": "read_text", "args": {"path": "a"}}))
    assert err is None and call.tool == "read_text" and call.args == {"path": "a"}


def test_parse_done_and_tool_mutually_exclusive():
    _, err = parse_tool_call(_J({"thought": "x", "done": True, "tool": "read_text", "args": {}}))
    assert err and "mutually exclusive" in err


def test_parse_unknown_tool_rejected():
    _, err = parse_tool_call(_J({"thought": "x", "tool": "rm_rf", "args": {}}))
    assert err and "unknown tool" in err


def test_parse_no_json_returns_error():
    _, err = parse_tool_call("not json at all")
    assert err


def test_allowed_tools_exact():
    # env 工具(observe/act=Phase A,recall_map=Phase C 记忆脑区,plan=Phase D.3 策略脑区,
    # recall_topo=Phase 4.6 拓扑记忆脑区,recall_path=Phase 4.7 路径轨迹记忆脑区)在 union;
    # code agent 的 system prompt 不列它们(不泄漏),仅幻觉调用时触发 → dispatch 显式报错。
    assert ALLOWED_TOOLS == frozenset(
        {"read_text", "search_text", "inspect_file", "apply_text_patch", "workspace_run_check",
         "list_allowed_roots", "observe", "act", "recall_map", "plan", "recall_topo", "recall_path"}
    )


# ---------- loop driver (mock backend) ----------

def _solve_script(run_dir):
    sha = _sha(run_dir, "ranges.py")
    return [
        _J({"thought": "read", "tool": "read_text", "args": {"path": "ranges.py"}}),
        _J({"thought": "fix", "tool": "apply_text_patch", "args": {
            "path": "ranges.py", "expected_sha256": sha,
            "replacements": [{"old_text": "range(start, end)", "new_text": "range(start, end + 1)"}],
            "dry_run": False}}),
        _J({"thought": "test", "tool": "workspace_run_check", "args": {"argv": [sys.executable, "-m", "pytest", "-q"]}}),
        _J({"thought": "done", "done": True, "answer": "fixed"}),
    ]


def test_loop_solves_happy_path():
    task, run_dir = _materialized("off_by_one")
    try:
        traj = asyncio.run(run_agent(MockBackend(_solve_script(run_dir)), "mock", task, run_dir=run_dir, arm="none"))
        assert traj.solve_status == "solved"
        assert traj.done and traj.n_steps == 4
        assert traj.termination_reason == "done"
        assert traj.steps[1].tool == "apply_text_patch"
    finally:
        cleanup_run_dir(run_dir)


def test_loop_consecutive_parse_error_early_stop():
    task, run_dir = _materialized("off_by_one")
    try:
        traj = asyncio.run(run_agent(
            MockBackend(["nope", "still nope", "again nope"]), "mock", task,
            run_dir=run_dir, arm="none", max_steps=10, consecutive_error_limit=3))
        assert traj.solve_status == "parse_error"
        assert traj.termination_reason == "parse_error"
        assert traj.n_steps == 3
    finally:
        cleanup_run_dir(run_dir)


def test_loop_budget_exceeded_terminates():
    task, run_dir = _materialized("off_by_one")
    try:
        traj = asyncio.run(run_agent(
            MockBackend([_J({"thought": "x", "tool": "list_allowed_roots", "args": {}})] * 50, cost=0.5),
            "mock", task, run_dir=run_dir, arm="none", max_steps=20, max_cost_usd=0.5))
        assert traj.termination_reason == "budget_exceeded"
        assert traj.solve_status in ("budget_exceeded", "tests_fail")
    finally:
        cleanup_run_dir(run_dir)


def test_loop_unknown_tool_error_feedback_no_crash():
    task, run_dir = _materialized("off_by_one")
    try:
        traj = asyncio.run(run_agent(
            MockBackend([_J({"thought": "x", "tool": "delete_all", "args": {}})] + _solve_script(run_dir)),
            "mock", task, run_dir=run_dir, arm="none"))
        assert traj.steps[0].error and "unknown tool" in traj.steps[0].error
        assert traj.solve_status == "solved"  # recovered and solved
    finally:
        cleanup_run_dir(run_dir)


def test_loop_brainregion_arm_calls_wake_gate():
    task, run_dir = _materialized("off_by_one")
    try:
        traj = asyncio.run(run_agent(
            MockBackend(_solve_script(run_dir)), "mock", task, run_dir=run_dir, arm="brainregion"))
        assert traj.solve_status == "solved"
        assert traj.wake_calls == 1
    finally:
        cleanup_run_dir(run_dir)


def test_loop_max_steps_terminates():
    task, run_dir = _materialized("off_by_one")
    try:
        traj = asyncio.run(run_agent(
            MockBackend([_J({"thought": "x", "tool": "list_allowed_roots", "args": {}})] * 50, cost=0.0),
            "mock", task, run_dir=run_dir, arm="none", max_steps=3, max_cost_usd=10.0))
        assert traj.termination_reason == "max_steps"
        assert traj.n_steps == 3
    finally:
        cleanup_run_dir(run_dir)


# ---------- scoped_workspace_root ----------

def test_scoped_root_overrides_env(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    monkeypatch.setenv("BRAIN_REGION_WORKSPACE_ROOTS", str(a))
    from brainregion.workspace.files import list_allowed_roots
    assert list_allowed_roots()["roots"][0]["path"] == str(a.resolve())
    with scoped_workspace_root(str(b)):
        roots = list_allowed_roots()["roots"]
        assert len(roots) == 1 and roots[0]["path"] == str(b.resolve())
        assert roots[0]["source"] == "contextvar:override"
    # 出 with → 回 env
    assert list_allowed_roots()["roots"][0]["path"] == str(a.resolve())


def test_scoped_root_confines_to_scoped_dir(tmp_path):
    sub = tmp_path / "sandbox"
    sub.mkdir()
    (sub / "inside.py").write_text("x = 1", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("x = 2", encoding="utf-8")
    from brainregion.workspace import inspect_file
    with scoped_workspace_root(str(sub)):
        # 内部路径 OK
        assert inspect_file("inside.py")["is_file"] is True
        # 越界路径(absolute,outside root)被拒
        with pytest.raises(PermissionError):
            inspect_file(str(outside))


# ---------- eval gate ----------

def test_gate_go_when_solve_rate_ci_above_zero():
    sr = {"point": 0.5, "low": 0.1, "high": 0.8, "n": 30}
    deltas = {"solve_rate_delta": sr, "cost_delta": {}, "steps_delta": {}}
    g = evaluate_gate(deltas, n=30)
    assert g["decision"] == "GO"


def test_gate_inconclusive_when_ci_crosses_zero():
    sr = {"point": 0.1, "low": -0.2, "high": 0.3, "n": 30}
    g = evaluate_gate({"solve_rate_delta": sr, "cost_delta": {}, "steps_delta": {}}, n=30)
    assert g["decision"] == "INCONCLUSIVE"


def test_gate_pilot_prefix_when_n_small():
    sr = {"point": 0.5, "low": 0.1, "high": 0.8, "n": 3}
    g = evaluate_gate({"solve_rate_delta": sr, "cost_delta": {}, "steps_delta": {}}, n=3)
    assert g["decision"] == "pilot_GO"


def test_gate_inconclusive_when_not_estimable():
    sr = {"point": None, "low": None, "high": None, "n": 1}
    g = evaluate_gate({"solve_rate_delta": sr, "cost_delta": {}, "steps_delta": {}}, n=1)
    assert "INCONCLUSIVE" in g["decision"]


def test_run_sandbox_eval_end_to_end_mock():
    """两臂都用解出脚本 → delta 0 → INCONCLUSIVE;验证报告结构 + trajectory 落 dict。"""
    task = get_fixture("off_by_one")
    # 预解 sha(两臂 run_dir 不同,但 fixture 文件内容相同 → sha 相同)
    probe_dir = make_run_dir()
    materialize_fixture(task, Path(probe_dir))
    sha = _sha(probe_dir, "ranges.py")
    cleanup_run_dir(probe_dir)

    def script_factory():
        return [
            _J({"thought": "fix", "tool": "apply_text_patch", "args": {
                "path": "ranges.py", "expected_sha256": sha,
                "replacements": [{"old_text": "range(start, end)", "new_text": "range(start, end + 1)"}],
                "dry_run": False}}),
            _J({"thought": "test", "tool": "workspace_run_check", "args": {"argv": [sys.executable, "-m", "pytest", "-q"]}}),
            _J({"thought": "done", "done": True, "answer": "fixed"}),
        ]

    # 每 run_agent 调用从脚本头开始:每次 complete_messages 重置脚本游标。
    class PerCallScriptBackend:
        def __init__(self):
            self._inner = None

        async def complete_messages(self, messages, **kw):
            self._inner = MockBackend(script_factory())
            return await self._inner.complete_messages(messages, **kw)

    report = asyncio.run(run_sandbox_eval(
        PerCallScriptBackend(), "mock", [task], max_steps=6, max_cost_usd=1.0,
    ))
    assert report["n"] == 1
    assert set(report["per_arm"]) == {"none", "brainregion"}
    assert report["per_arm"]["none"]["solve_rate"] == 1.0
    assert report["per_arm"]["brainregion"]["solve_rate"] == 1.0
    assert "decision" in report["gate"]
    assert len(report["trajectories"]) == 2
