"""brain_verify(§15.8 trace-first + test-backstop)单测。

hermetic:mock backend(不调真模型)+ monkeypatch verify_solution(不跑 pytest)。
"""
from __future__ import annotations

import asyncio

from brainregion.sandbox import brain_verify as bv
from brainregion.sandbox.brain_verify import (
    BrainVerifyResult,
    TraceResult,
    composite_verify,
    extract_final_patch,
    forced_trace,
    verify_with_brain,
)


class _Resp:
    def __init__(self, content, error=None):
        self.content = content
        self.error = error


class _Backend:
    """记录调用 + 返回预设 content/error 的假 backend。"""

    def __init__(self, content="", error=None):
        self._content = content
        self._error = error
        self.last_kwargs = None

    async def complete(self, **kwargs):
        self.last_kwargs = kwargs
        return _Resp(self._content, self._error)


# ---------------- extract_final_patch ----------------
def _step(tool="read_text", error=None, **args):
    return {"tool": tool, "args": args, "error": error}


def test_extract_final_patch_prefers_last_non_error():
    traj = {"steps": [
        _step("apply_text_patch", error="old_text not found",
              path="a.py", replacements=[{"old_text": "x", "new_text": "y"}]),
        _step("apply_text_patch", error=None,
              path="b.py", replacements=[{"old_text": "1", "new_text": "2"}]),
    ]}
    patch = extract_final_patch(traj)
    assert patch["path"] == "b.py"
    assert patch["replacements"][0]["new_text"] == "2"


def test_extract_final_patch_all_error_returns_none():
    traj = {"steps": [
        _step("apply_text_patch", error="boom", path="a.py", replacements=[{"old_text": "x", "new_text": "y"}]),
    ]}
    assert extract_final_patch(traj) is None


def test_extract_final_patch_no_patch_returns_none():
    assert extract_final_patch({"steps": [_step("read_text", path="a.py")]}) is None
    assert extract_final_patch({"steps": []}) is None


def test_extract_final_patch_ignores_apply_without_replacements():
    # apply_text_patch 但无 replacements(空补丁)→ 不算
    traj = {"steps": [_step("apply_text_patch", error=None, path="a.py", replacements=[])]}
    assert extract_final_patch(traj) is None


def test_extract_final_patch_handles_steprecord_objects():
    # 活 trajectory 的 steps 是 StepRecord 对象(非 dict);duck-typed 取值
    from types import SimpleNamespace
    traj = {"steps": [
        SimpleNamespace(tool="read_text", args={"path": "a.py"}, error=None),
        SimpleNamespace(tool="apply_text_patch", error=None,
                        args={"path": "b.py", "replacements": [{"old_text": "1", "new_text": "2"}]}),
    ]}
    patch = extract_final_patch(traj)
    assert patch["path"] == "b.py"
    assert patch["replacements"][0]["new_text"] == "2"


# ---------------- composite_verify 矩阵 ----------------
def _tr(verdict):
    return TraceResult(verdict=verdict, parse_ok=verdict is not None)


def test_composite_agree_solved():
    r = composite_verify(_tr("SOLVED"), True)
    assert r.agree is True
    assert not r.weak_test_signal and not r.trace_missed
    assert r.verdict == "SOLVED"


def test_composite_agree_failed():
    r = composite_verify(_tr("FAILED"), False)
    assert r.agree is True
    assert r.verdict == "FAILED"


def test_composite_weak_test_signal():
    # ISS-016 场景:客观测试过(弱测试)但 trace 判 FAILED
    r = composite_verify(_tr("FAILED"), True)
    assert r.agree is False
    assert r.weak_test_signal is True
    assert r.trace_missed is False
    assert any("弱测试" in n for n in r.notes)
    # 最终判定仍以客观测试为准(过)
    assert r.verdict == "SOLVED"


def test_composite_trace_missed():
    # 测试败但 trace 判 SOLVED(trace 不可靠)
    r = composite_verify(_tr("SOLVED"), False)
    assert r.agree is False
    assert r.trace_missed is True
    assert r.weak_test_signal is False
    assert any("漏检" in n for n in r.notes)
    assert r.verdict == "FAILED"


def test_composite_trace_unparseable():
    r = composite_verify(_tr(None), True)
    assert r.agree is None
    assert r.trace_verdict is None
    assert any("未解析" in n for n in r.notes)
    assert r.verdict == "SOLVED"  # test_green 兜底


def test_composite_no_test_falls_back_to_trace():
    r = composite_verify(_tr("FAILED"), None)
    assert r.agree is None
    assert r.verdict == "FAILED"  # 无测试 → 退回 trace
    assert any("无客观测试" in n for n in r.notes)


def test_verdict_property_none_both():
    assert composite_verify(_tr(None), None).verdict is None


# ---------------- forced_trace(解析)----------------
def test_forced_trace_parses_clean_json():
    backend = _Backend(content='{"verdict":"FAILED","trace":"exit_code=0→ok=True","check":"ok=False 不满足","confidence":0.9}')
    r = asyncio.run(forced_trace(backend, model="m", endpoint_id="e",
                                 goal="g", test_req="r", patch={"path": "p", "replacements": []}))
    assert r.verdict == "FAILED"
    assert r.parse_ok is True
    assert "exit_code" in r.trace
    assert r.confidence == 0.9


def test_forced_trace_parses_json_embedded_in_prose():
    backend = _Backend(content='经分析,结论如下:\n{"verdict":"SOLVED","trace":"t","check":"c"}\n以上。')
    r = asyncio.run(forced_trace(backend, model="m", endpoint_id=None,
                                 goal="g", test_req="r", patch={"path": "p", "replacements": []}))
    assert r.verdict == "SOLVED"


def test_forced_trace_fallback_regex_when_no_json():
    backend = _Backend(content="我觉得结果是 FAILED,因为缺 fsync。")
    r = asyncio.run(forced_trace(backend, model="m", endpoint_id=None,
                                 goal="g", test_req="r", patch={"path": "p", "replacements": []}))
    assert r.verdict == "FAILED"
    assert r.parse_ok is True  # fallback 命中


def test_forced_trace_unparseable_returns_none():
    backend = _Backend(content="无法判断,信息不足。")
    r = asyncio.run(forced_trace(backend, model="m", endpoint_id=None,
                                 goal="g", test_req="r", patch={"path": "p", "replacements": []}))
    assert r.verdict is None
    assert r.parse_ok is False
    assert any("未解析" in n for n in composite_verify(r, True).notes)


def test_forced_trace_passes_through_backend_error():
    backend = _Backend(content="", error="timeout")
    r = asyncio.run(forced_trace(backend, model="m", endpoint_id=None,
                                 goal="g", test_req="r", patch={"path": "p", "replacements": []}))
    assert r.verdict is None
    assert r.error == "timeout"


def test_forced_trace_sends_trace_system_prompt():
    backend = _Backend(content='{"verdict":"SOLVED","trace":"","check":""}')
    asyncio.run(forced_trace(backend, model="m", endpoint_id="e",
                             goal="g", test_req="r", patch={"path": "p", "replacements": []}))
    assert backend.last_kwargs["system"] == bv.SYS_TRACE
    assert backend.last_kwargs["endpoint_id"] == "e"
    assert "g" in backend.last_kwargs["user"] and "r" in backend.last_kwargs["user"]


# ---------------- verify_with_brain(编排)----------------
def test_verify_with_brain_weak_test_signal(monkeypatch):
    # 客观测试过(弱测试)+ trace 判 FAILED → weak_test_signal
    monkeypatch.setattr(bv, "verify_solution", lambda task, run_dir, **kw: {"tests_green": True})
    backend = _Backend(content='{"verdict":"FAILED","trace":"缺 fsync","check":"(b) 不满足"}')
    task = type("T", (), {"goal": "原子写", "test_args": ["t"], "gold_diff": ""})()
    traj = {"steps": [_step("apply_text_patch", error=None, path="f.py",
                            replacements=[{"old_text": "write_bytes", "new_text": "temp+replace"}])]}
    r = asyncio.run(verify_with_brain(backend, task=task, run_dir=".", trajectory=traj,
                                      model="m", endpoint_id=None))
    assert isinstance(r, BrainVerifyResult)
    assert r.test_green is True
    assert r.trace_verdict == "FAILED"
    assert r.weak_test_signal is True
    assert r.agree is False


def test_verify_with_brain_no_patch_skips_trace(monkeypatch):
    monkeypatch.setattr(bv, "verify_solution", lambda task, run_dir, **kw: {"tests_green": False})
    backend = _Backend(content='{"verdict":"SOLVED"}')  # 不该被调用
    task = type("T", (), {"goal": "g", "test_args": ["t"], "gold_diff": ""})()
    traj = {"steps": [_step("read_text", path="f.py")]}  # 无 patch
    r = asyncio.run(verify_with_brain(backend, task=task, run_dir=".", trajectory=traj,
                                      model="m", endpoint_id=None))
    assert r.trace_verdict is None
    assert r.test_green is False
    assert backend.last_kwargs is None  # forced_trace 没跑
    assert any("无 apply_text_patch" in n for n in r.notes)


def test_verify_with_brain_test_req_defaults_to_goal(monkeypatch):
    captured = {}

    def fake_verify(task, run_dir, **kw):
        return {"tests_green": True}

    monkeypatch.setattr(bv, "verify_solution", fake_verify)

    class _B(_Backend):
        async def complete(self, **kwargs):
            captured["user"] = kwargs["user"]
            return _Resp('{"verdict":"SOLVED","trace":"","check":""}')

    backend = _B()
    task = type("T", (), {"goal": "GOAL_TEXT", "test_args": ["t"], "gold_diff": ""})()
    traj = {"steps": [_step("apply_text_patch", error=None, path="f.py",
                            replacements=[{"old_text": "a", "new_text": "b"}])]}
    asyncio.run(verify_with_brain(backend, task=task, run_dir=".", trajectory=traj,
                                  model="m", endpoint_id=None))
    # test_req 默认 = goal → goal 出现在 user 里
    assert "GOAL_TEXT" in captured["user"]


# ---------------- brain_verify_from_trajectory(run loop 用)----------------
def test_brain_verify_from_trajectory_weak_test_signal():
    # 有 patch + 测试过 + trace FAILED → 弱测试信号
    backend = _Backend(content='{"verdict":"FAILED","trace":"缺 fsync","check":"(b) 不满足","confidence":0.9}')
    steps = [_step("apply_text_patch", error=None, path="f.py",
                   replacements=[{"old_text": "write_bytes", "new_text": "temp+replace"}])]
    out = asyncio.run(bv.brain_verify_from_trajectory(
        backend, model="m", endpoint_id=None, goal="原子写",
        steps=steps, test_green=True))
    assert out["trace_verdict"] == "FAILED"
    assert out["test_green"] is True
    assert out["weak_test_signal"] is True
    assert out["agree"] is False
    assert out["final_verdict"] == "SOLVED"  # 以客观测试为准


def test_brain_verify_from_trajectory_agree_failed():
    backend = _Backend(content='{"verdict":"FAILED","trace":"t","check":"c"}')
    steps = [_step("apply_text_patch", error=None, path="f.py",
                   replacements=[{"old_text": "a", "new_text": "b"}])]
    out = asyncio.run(bv.brain_verify_from_trajectory(
        backend, model="m", endpoint_id=None, goal="g", steps=steps, test_green=False))
    assert out["trace_verdict"] == "FAILED"
    assert out["test_green"] is False
    assert out["agree"] is True
    assert out["final_verdict"] == "FAILED"


def test_brain_verify_from_trajectory_no_patch_skips_trace():
    backend = _Backend(content='{"verdict":"SOLVED"}')  # 不该被调
    out = asyncio.run(bv.brain_verify_from_trajectory(
        backend, model="m", endpoint_id=None, goal="g",
        steps=[_step("read_text", path="f.py")], test_green=False))
    assert out["trace_verdict"] is None
    assert out["test_green"] is False
    assert backend.last_kwargs is None  # forced_trace 没跑
    assert any("无 apply_text_patch" in n for n in out["notes"])


# ---------------- Trajectory.brain_verify 序列化 ----------------
def test_trajectory_brain_verify_field_serializes():
    from brainregion.sandbox.loop import Trajectory

    traj = Trajectory(task_id="t", arm="none")
    assert traj.brain_verify is None  # 默认 None
    assert traj.to_dict()["brain_verify"] is None
    traj.brain_verify = {"trace_verdict": "SOLVED", "test_green": True}
    assert traj.to_dict()["brain_verify"] == {"trace_verdict": "SOLVED", "test_green": True}
