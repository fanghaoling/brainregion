"""brain_delegate(§15.1 Delegate 步)单测。hermetic:mock backend + 纯 policy。"""
from __future__ import annotations

import asyncio

from brainregion.sandbox.brain_delegate import (
    DELEGATE_ACTIONS,
    DelegateDecision,
    RESOLVE_ACTIONS,
    Resolution,
    delegate_policy,
    delegate_step,
    orthogonal_check,
    resolve_escalate,
    resolve_escalate_from_trajectory,
)
from brainregion.sandbox import brain_delegate as bd
from brainregion.sandbox import brain_verify as bv


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


# ============================================================================
# escalate handler —— 正交复查(Delegate 子系统的 handler,§15.1)
# Diagnosis(正交 verdict/check)→ Action(resolve_escalate),零 LLM 决策。
# ============================================================================
def test_resolve_escalate_solved_to_accept():
    r = resolve_escalate(orthogonal_verdict="SOLVED", orthogonal_check="ok", trace_check="缺 fsync")
    assert r.action == "accept"
    assert r.accept_reason == "orthogonal_cleared"
    assert r.directive == ""


def test_resolve_escalate_failed_to_redelegate_with_directive():
    r = resolve_escalate(orthogonal_verdict="FAILED", orthogonal_check="缺 os.fsync", trace_check="缺 fsync")
    assert r.action == "redelegate"
    assert r.directive == "缺 os.fsync"  # directive = 正交 check 差距(grounded)
    assert r.accept_reason == ""


def test_resolve_escalate_none_to_weak_test_fallback():
    r = resolve_escalate(orthogonal_verdict=None, orthogonal_check="", trace_check="缺 fsync")
    assert r.action == "accept"
    assert r.accept_reason == "weak_test"


def test_resolve_escalate_actions_are_valid_vocab():
    for ov in ("SOLVED", "FAILED", None):
        r = resolve_escalate(orthogonal_verdict=ov, orthogonal_check="c", trace_check="t")
        assert r.action in RESOLVE_ACTIONS


def test_resolve_escalate_gap_consensus_same_check():
    # 正交 check 与 trace check 归一化相等 → gap_consensus=True(两 reviewer 找同一差距)
    r = resolve_escalate(orthogonal_verdict="FAILED", orthogonal_check="缺  Os.Fsync", trace_check="缺 os.fsync")
    assert r.gap_consensus is True


def test_resolve_escalate_gap_consensus_different_check():
    # 不同差距(归一化不等)→ gap_consensus=False(更值得 redelegate;GPT ④:比 check 非 仅 verdict)
    r = resolve_escalate(orthogonal_verdict="FAILED", orthogonal_check="缺 rollback", trace_check="缺 fsync")
    assert r.gap_consensus is False


# ---------------- orthogonal_check(复用 forced_trace 盲审)----------------
def test_orthogonal_check_uses_orthogonal_model_and_sys_trace():
    backend = _Backend(content='{"verdict":"FAILED","trace":"t","check":"缺 fsync"}')
    tr = asyncio.run(orthogonal_check(
        backend, orthogonal_model="glm-5.2", orthogonal_endpoint_id="zhipu",
        goal="原子写", test_req="r", patch=_patch()))
    assert tr.verdict == "FAILED"
    assert tr.check == "缺 fsync"
    assert backend.last_kwargs["model"] == "glm-5.2"            # orthogonal_model(非 main)
    assert backend.last_kwargs["endpoint_id"] == "zhipu"
    assert backend.last_kwargs["system"] == bv.SYS_TRACE        # 复用 SYS_TRACE(盲审,零新 prompt)


def test_orthogonal_check_passes_through_backend_error():
    backend = _Backend(content="", error="timeout")
    tr = asyncio.run(orthogonal_check(
        backend, orthogonal_model="glm-5.2", orthogonal_endpoint_id=None,
        goal="g", test_req="r", patch=_patch()))
    assert tr.verdict is None
    assert tr.error == "timeout"


def test_orthogonal_check_tracks_cost():
    backend = _Backend(content='{"verdict":"SOLVED","trace":"","check":""}', cost_usd=0.015)
    tr = asyncio.run(orthogonal_check(
        backend, orthogonal_model="glm-5.2", orthogonal_endpoint_id=None,
        goal="g", test_req="r", patch=_patch()))
    assert tr.cost_usd == 0.015


# ---------------- resolve_escalate_from_trajectory(loop 用)----------------
def test_resolve_escalate_from_trajectory_failed_to_redelegate():
    backend = _Backend(content='{"verdict":"FAILED","trace":"t","check":"缺 fsync"}')
    out = asyncio.run(resolve_escalate_from_trajectory(
        backend, orthogonal_model="glm-5.2", orthogonal_endpoint_id=None,
        goal="原子写", steps=[_apply_step()], brain_verify_dict={"check": "缺 fsync"}))
    assert out["action"] == "redelegate"
    assert out["directive"] == "缺 fsync"
    assert out["orthogonal_verdict"] == "FAILED"
    assert out["gap_consensus"] is True
    assert backend.calls == 1


def test_resolve_escalate_from_trajectory_solved_to_accept():
    backend = _Backend(content='{"verdict":"SOLVED","trace":"","check":"ok"}')
    out = asyncio.run(resolve_escalate_from_trajectory(
        backend, orthogonal_model="glm-5.2", orthogonal_endpoint_id=None,
        goal="g", steps=[_apply_step()], brain_verify_dict={"check": "缺 fsync"}))
    assert out["action"] == "accept"
    assert out["accept_reason"] == "orthogonal_cleared"


def test_resolve_escalate_from_trajectory_no_patch_fallback():
    backend = _Backend(content='{"verdict":"FAILED"}')  # 不该被调
    out = asyncio.run(resolve_escalate_from_trajectory(
        backend, orthogonal_model="glm-5.2", orthogonal_endpoint_id=None,
        goal="g", steps=[{"tool": "read_text", "args": {"path": "f.py"}}],
        brain_verify_dict={"check": "缺 fsync"}))
    assert out["action"] == "accept"
    assert out["accept_reason"] == "weak_test"
    assert backend.calls == 0
    assert out["error"]


def test_resolve_escalate_from_trajectory_backend_error_fallback():
    # 正交调用 raise → accept fallback(失败隔离,不抛)
    class _ErrBackend(_Backend):
        async def complete(self, **kwargs):
            raise RuntimeError("orthogonal boom")
    out = asyncio.run(resolve_escalate_from_trajectory(
        _ErrBackend(), orthogonal_model="glm-5.2", orthogonal_endpoint_id=None,
        goal="g", steps=[_apply_step()], brain_verify_dict={"check": "缺 fsync"}))
    assert out["action"] == "accept"
    assert out["accept_reason"] == "weak_test"
    assert "orthogonal_check failed" in (out["error"] or "")


# ---------------- Resolution.to_dict ----------------
def test_resolution_to_dict():
    r = Resolution(action="redelegate", directive="补 fsync", gap_consensus=True,
                   orthogonal_verdict="FAILED", cost_usd=0.01)
    out = r.to_dict()
    assert out["action"] == "redelegate"
    assert out["directive"] == "补 fsync"
    assert out["gap_consensus"] is True
    assert out["orthogonal_verdict"] == "FAILED"
    assert out["cost_usd"] == 0.01
