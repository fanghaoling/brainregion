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
    def __init__(self, content, error=None):
        self.content = content
        self.error = error


class _Backend:
    def __init__(self, content="", error=None):
        self._content = content
        self._error = error
        self.last_kwargs = None
        self.calls = 0

    async def complete(self, **kwargs):
        self.last_kwargs = kwargs
        self.calls += 1
        return _Resp(self._content, self._error)


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
    bv = {"test_green": False, "trace_verdict": "FAILED", "weak_test_signal": False, "trace_missed": False}
    d = asyncio.run(delegate_step(backend, model="m", endpoint_id=None,
                                  task_goal="g", patch=_patch(), brain_verify=bv))
    assert d.action == "redelegate"  # policy 仍生效
    assert d.next_subgoal == ""
    assert d.parse_ok is False


def test_delegate_step_sends_action_in_user():
    backend = _Backend(content='{"next_subgoal":"x"}')
    bv = {"test_green": False, "trace_verdict": "FAILED", "weak_test_signal": False, "trace_missed": False}
    asyncio.run(delegate_step(backend, model="m", endpoint_id=None,
                              task_goal="g", patch=_patch(), brain_verify=bv))
    # user 里写明已确定的 action
    assert "redelegate" in backend.last_kwargs["user"]


# ---------------- to_dict ----------------
def test_delegate_decision_to_dict():
    d = DelegateDecision(action="redelegate", next_subgoal="add fsync", target="same expert",
                         reason="r", confidence=0.8, parse_ok=True)
    out = d.to_dict()
    assert out["action"] == "redelegate"
    assert out["next_subgoal"] == "add fsync"
    assert out["confidence"] == 0.8
