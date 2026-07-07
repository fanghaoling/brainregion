"""brain_delegate(§15.1 Delegate 步)单测。hermetic:mock backend + 纯 policy。"""
from __future__ import annotations

import asyncio

from brainregion.sandbox.brain_delegate import (
    DELEGATE_ACTIONS,
    DelegateDecision,
    delegate_policy,
    delegate_step,
)
from brainregion.sandbox import brain_delegate as bd


class _Resp:
    def __init__(self, content, error=None, cost_usd=0.0):
        self.content = content
        self.error = error
        self.cost_usd = cost_usd


class _Backend:
    def __init__(self, content="", error=None, cost_usd=0.0):
        self._content = content
        self._error = error
        self._cost_usd = cost_usd
        self.last_kwargs = None
        self.calls = 0

    async def complete(self, **kwargs):
        self.last_kwargs = kwargs
        self.calls += 1
        return _Resp(self._content, self._error, self._cost_usd)


# ---------------- delegate_policy(纯函数矩阵)----------------
def test_policy_accept():
    assert delegate_policy(test_green=True, trace_verdict="SOLVED",
                           weak_test_signal=False, trace_missed=False) == "accept"


def test_policy_escalate_on_weak_test_signal():
    # 测试过但 trace 怀疑(弱测试)→ escalate
    assert delegate_policy(test_green=True, trace_verdict="FAILED",
                           weak_test_signal=True, trace_missed=False) == "escalate"


def test_policy_redelegate_on_test_fail():
    assert delegate_policy(test_green=False, trace_verdict="FAILED",
                           weak_test_signal=False, trace_missed=False) == "redelegate"
    # 即便 trace 误判 SOLVED(trace_missed),测试败仍 redelegate
    assert delegate_policy(test_green=False, trace_verdict="SOLVED",
                           weak_test_signal=False, trace_missed=True) == "redelegate"


def test_policy_give_up_when_no_test():
    assert delegate_policy(test_green=None, trace_verdict=None,
                           weak_test_signal=False, trace_missed=False) == "give_up"


def test_policy_actions_are_valid_vocab():
    for tg in (True, False, None):
        a = delegate_policy(test_green=tg, trace_verdict="SOLVED",
                            weak_test_signal=False, trace_missed=False)
        assert a in DELEGATE_ACTIONS


# ---------------- delegate_step ----------------
def _patch():
    return {"path": "f.py", "replacements": [{"old_text": "a", "new_text": "b"}]}


def test_delegate_step_accept_skips_llm():
    backend = _Backend(content='{"next_subgoal":"x"}')  # 不该被调
    bv = {"test_green": True, "trace_verdict": "SOLVED", "weak_test_signal": False, "trace_missed": False}
    d = asyncio.run(delegate_step(backend, model="m", endpoint_id=None,
                                  task_goal="g", patch=_patch(), brain_verify=bv))
    assert d.action == "accept"
    assert backend.calls == 0  # accept 纯确定性,不调 LLM
    assert d.parse_ok is True
    assert d.to_dict()["action"] == "accept"


def test_delegate_step_give_up_skips_llm():
    backend = _Backend()
    bv = {"test_green": None, "trace_verdict": None, "weak_test_signal": False, "trace_missed": False}
    d = asyncio.run(delegate_step(backend, model="m", endpoint_id=None,
                                  task_goal="g", patch=_patch(), brain_verify=bv))
    assert d.action == "give_up"
    assert backend.calls == 0


def test_delegate_step_redelegate_calls_llm_and_parses():
    backend = _Backend(content='{"next_subgoal":"补 os.fsync(fd) 在 os.replace 前","target":"same expert","reason":"trace 指出缺 fsync","confidence":0.9}')
    bv = {"test_green": False, "trace_verdict": "FAILED", "weak_test_signal": False,
          "trace_missed": False, "trace": "os.write→close→replace", "check": "(b) 缺 os.fsync"}
    d = asyncio.run(delegate_step(backend, model="m", endpoint_id="e",
                                  task_goal="原子写", patch=_patch(), brain_verify=bv))
    assert d.action == "redelegate"
    assert "fsync" in d.next_subgoal
    assert d.target == "same expert"
    assert d.confidence == 0.9
    assert d.parse_ok is True
    # 送的是 SYS_DELEGATE + 信号进了 user
    assert backend.last_kwargs["system"] == bd.SYS_DELEGATE
    assert "(b) 缺 os.fsync" in backend.last_kwargs["user"]
    assert "原子写" in backend.last_kwargs["user"]


def test_delegate_step_escalate_calls_llm():
    backend = _Backend(content='{"next_subgoal":"强化测试断言 fsync","target":"test-strengthener","reason":"弱测试","confidence":0.7}')
    bv = {"test_green": True, "trace_verdict": "FAILED", "weak_test_signal": True,
          "trace_missed": False, "check": "缺 fsync"}
    d = asyncio.run(delegate_step(backend, model="m", endpoint_id=None,
                                  task_goal="g", patch=_patch(), brain_verify=bv))
    assert d.action == "escalate"
    assert "fsync" in d.next_subgoal
    assert backend.calls == 1


def test_delegate_step_parse_failure_keeps_action():
    # LLM 返回不可解析 → action 仍由 policy 定(redelegate),next_subgoal 空,parse_ok False
    backend = _Backend(content="嗯,我觉得应该加 fsync 吧。")  # 无 JSON
    bv = {"test_green": False, "trace_verdict": "FAILED", "weak_test_signal": False, "trace_missed": False,
          "check": "缺 os.fsync"}  # 有 check → 走 LLM 路径(测解析失败)
    d = asyncio.run(delegate_step(backend, model="m", endpoint_id=None,
                                  task_goal="g", patch=_patch(), brain_verify=bv))
    assert d.action == "redelegate"  # policy 仍生效
    assert d.next_subgoal == ""
    assert d.parse_ok is False


def test_delegate_step_sends_action_in_user():
    backend = _Backend(content='{"next_subgoal":"x"}')
    bv = {"test_green": False, "trace_verdict": "FAILED", "weak_test_signal": False, "trace_missed": False,
          "check": "缺 os.fsync"}  # 有 check → 走 LLM
    asyncio.run(delegate_step(backend, model="m", endpoint_id=None,
                              task_goal="g", patch=_patch(), brain_verify=bv))
    # user 里写明已确定的 action
    assert "redelegate" in backend.last_kwargs["user"]


def test_delegate_step_tracks_cost_into_dict():
    # sidecar cost 跟踪:delegate_step 从 resp.cost_usd 取 → DelegateDecision.cost_usd → to_dict
    backend = _Backend(content='{"next_subgoal":"补 fsync","target":"same expert","reason":"r"}',
                       cost_usd=0.008)
    bv = {"test_green": False, "trace_verdict": "FAILED", "weak_test_signal": False,
          "trace_missed": False, "check": "缺 fsync"}
    d = asyncio.run(delegate_step(backend, model="m", endpoint_id=None,
                                  task_goal="g", patch=_patch(), brain_verify=bv))
    assert d.cost_usd == 0.008
    assert d.to_dict()["cost_usd"] == 0.008


# ---------------- to_dict ----------------
def test_delegate_decision_to_dict():
    d = DelegateDecision(action="redelegate", next_subgoal="add fsync", target="same expert",
                         reason="r", confidence=0.8, parse_ok=True)
    out = d.to_dict()
    assert out["action"] == "redelegate"
    assert out["next_subgoal"] == "add fsync"
    assert out["confidence"] == 0.8


# ---------------- delegate_from_trajectory(run loop 用)----------------
def _apply_step():
    return {"tool": "apply_text_patch", "error": None,
            "args": {"path": "f.py", "replacements": [{"old_text": "a", "new_text": "b"}]}}


def test_delegate_from_trajectory_accept_no_llm():
    backend = _Backend(content='{"next_subgoal":"x"}')  # accept 不该调 LLM
    out = asyncio.run(bd.delegate_from_trajectory(
        backend, model="m", endpoint_id=None, goal="g", steps=[_apply_step()],
        test_green=True,
        brain_verify_dict={"trace_verdict": "SOLVED", "weak_test_signal": False, "trace_missed": False}))
    assert out["action"] == "accept"
    assert backend.calls == 0


def test_delegate_from_trajectory_uses_test_green_from_traj_not_bv():
    # 关键鲁棒性:test_green 用传入的 traj 值(False),即便 brain_verify_dict 缺 test_green
    # (trace 调用失败被 run_agent 兜成 error dict 的场景)
    backend = _Backend(content='{"next_subgoal":"补 os.fsync","target":"same expert","reason":"r","confidence":0.9}')
    out = asyncio.run(bd.delegate_from_trajectory(
        backend, model="m", endpoint_id=None, goal="g", steps=[_apply_step()],
        test_green=False,
        brain_verify_dict={"trace_verdict": "FAILED", "check": "缺 os.fsync"}))  # 故意无 test_green key
    assert out["action"] == "redelegate"
    assert "fsync" in out["next_subgoal"]


def test_delegate_from_trajectory_no_patch_skips_llm():
    backend = _Backend()
    out = asyncio.run(bd.delegate_from_trajectory(
        backend, model="m", endpoint_id=None, goal="g",
        steps=[{"tool": "read_text", "args": {"path": "f.py"}}],
        test_green=False, brain_verify_dict={}))
    assert out["action"] == "redelegate"  # test_green=False → policy 出 redelegate
    assert backend.calls == 0  # 无 patch → 不调 LLM formulate


def test_delegate_step_redelegate_empty_check_skips_llm():
    # trace 失败(brain_verify 无 check)→ action=redelegate 但不调 LLM(空 check 下只会 confabulate)
    backend = _Backend(content='{"next_subgoal":"编造的"}')  # 不该被调
    bv = {"test_green": False, "trace_verdict": None, "weak_test_signal": False,
          "trace_missed": False, "check": ""}  # 模拟 trace 失败的 error dict
    d = asyncio.run(delegate_step(backend, model="m", endpoint_id=None,
                                  task_goal="g", patch=_patch(), brain_verify=bv))
    assert d.action == "redelegate"  # policy 仍定 action
    assert d.next_subgoal == ""  # 无 check → 不 formulate
    assert backend.calls == 0  # 没调 LLM(防 confabulation)
    assert "无 check" in d.reason or "防编造" in d.reason


def test_delegate_from_trajectory_trace_failed_no_confabulation():
    # 完整失败路径:brain_verify trace 调用失败(error dict 无 check)+ test_green=False
    # → delegate 不 confabulate(policy 出 redelegate,但不调 LLM,next_subgoal 空)
    backend = _Backend(content='{"next_subgoal":"编造"}')
    out = asyncio.run(bd.delegate_from_trajectory(
        backend, model="m", endpoint_id=None, goal="g", steps=[_apply_step()],
        test_green=False,
        brain_verify_dict={"error": "brain_verify failed: timeout", "trace_verdict": None}))  # 无 check
    assert out["action"] == "redelegate"
    assert out["next_subgoal"] == ""
    assert backend.calls == 0


# ---------------- Trajectory.delegate 序列化 ----------------
def test_trajectory_delegate_field_serializes():
    from brainregion.sandbox.loop import Trajectory

    traj = Trajectory(task_id="t", arm="none")
    assert traj.delegate is None
    assert traj.to_dict()["delegate"] is None
    traj.delegate = {"action": "accept"}
    assert traj.to_dict()["delegate"] == {"action": "accept"}
