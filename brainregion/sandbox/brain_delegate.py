"""主脑 Delegate 步(§15.1 认知环):基于 brain_verify 的 grounded 信号决定下一步。

grounding-first(同 brain_verify,Step A/A.5/A.6 教训):**action 由信号确定性推出**(纯函数,
零 confabulation 空间),LLM 只在 redelegate/escalate 时 formulate **下一子目标** —— 且必须基于
trace 指出的具体差距(``trace.check``),不得编造 trace 没提的问题。

这是「主脑决定下一步该干什么」(§15.1 控制环 Delegate? 步)的落地,但下一步是 **grounded** 的:
主脑的「判断」被约束成「formulate grounded 的下一步」,不是自由决策(避免 Step A confabulation
在 Delegate 层重演)。消费 brain_verify 的输出( Merge 步),供未来外环(专家→verify→delegate→
再专家)或 run.json 复盘用。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .brain_verify import TraceResult, extract_final_patch, forced_trace, render_patch

# action 词表(可序列化进 run.json):
# accept      —— 客观测试过 + trace 认可,完成。
# redelegate  —— 客观测试败,用 trace 差距作为下一子目标重派专家。
# escalate    —— 测试过但 trace 怀疑(弱测试信号),升级(强化测试 / 正交复查)。
# give_up     —— 无客观测试结果(budget/parse/无补丁),无法 grounded 决策。
DELEGATE_ACTIONS = ("accept", "redelegate", "escalate", "give_up")


@dataclass
class DelegateDecision:
    """Delegate 步的输出(主脑的 grounded 下一步决策)。"""

    action: str  # DELEGATE_ACTIONS 之一
    next_subgoal: str = ""  # redelegate/escalate 时:给专家的下一子目标(grounded in trace.check)
    target: str = ""  # 重派给谁(如 "same expert" / "orthogonal reviewer" / "test-strengthener")
    reason: str = ""
    confidence: float | None = None
    parse_ok: bool = False
    cost_usd: float = 0.0
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "next_subgoal": self.next_subgoal,
            "target": self.target,
            "reason": self.reason,
            "confidence": self.confidence,
            "parse_ok": self.parse_ok,
            "cost_usd": self.cost_usd,
        }


def delegate_policy(
    *,
    test_green: bool | None,
    trace_verdict: str | None,
    weak_test_signal: bool,
    trace_missed: bool,  # noqa: ARG001  —— 保留入参对称(未来 policy 可用:trace 漏检影响置信度)
) -> str:
    """纯函数:grounded 信号 → action(确定性,无 LLM,零 confabulation 空间)。

    - test_green=True + weak_test_signal → escalate(测试可能弱,升级)
    - test_green=True + 无 weak → accept(客观测试过 + trace 认可)
    - test_green=False → redelegate(用 trace 差距重派)
    - test_green=None → give_up(无客观结果,无法 grounded 决策)
    """
    if test_green is True:
        return "escalate" if weak_test_signal else "accept"
    if test_green is False:
        return "redelegate"
    return "give_up"


# forced delegate 系统 prompt(grounding-first):action 已由信号确定,LLM 只 formulate 下一子目标,
# 且必须基于 trace 指出的具体差距(trace.check),严禁编造 trace 没提的问题。
SYS_DELEGATE = (
    "你是主脑(Control Plane)。专家刚完成一个子任务,brain_verify 给出了 grounded 验证信号。\n"
    "action 已由信号确定性推出(见下)。你的职责:**基于 trace 指出的具体差距**,formulate 下一步"
    "给专家的子目标(next_subgoal)。grounding-first 铁律:\n"
    "1. next_subgoal 只能针对 trace.check **明确指出的差距**,严禁编造 trace 没提的问题。\n"
    "2. next_subgoal 要具体、可执行(专家能据此直接改)。\n"
    "3. action=redelegate → next_subgoal = 让专家补 trace 指出的差距(如『加 os.fsync』)。\n"
    "   action=escalate(弱测试)→ next_subgoal = 强化测试覆盖该差距,或正交复查。\n"
    "输出严格 JSON:{next_subgoal, target, reason, confidence(0-1)}。target 指重派对象"
    "(如 same expert / orthogonal reviewer / test-strengthener)。"
)


def _build_user(task_goal: str, patch: dict[str, Any], bv: dict[str, Any], action: str) -> str:
    return (
        f"【任务总目标】\n{task_goal}\n\n"
        f"【专家的补丁】\n{render_patch(patch)}\n\n"
        "【brain_verify grounded 信号】\n"
        f"- 客观测试 test_green = {bv.get('test_green')}\n"
        f"- forced-trace 判定 trace_verdict = {bv.get('trace_verdict')}\n"
        f"- 弱测试信号 weak_test_signal = {bv.get('weak_test_signal')}\n"
        f"- trace 漏检 trace_missed = {bv.get('trace_missed')}\n"
        f"- trace 推理: {(bv.get('trace') or '')[:400]}\n"
        f"- trace 指出的差距(check): {(bv.get('check') or '')[:400]}\n\n"
        f"【已确定的 action(由信号确定性推出)】{action}\n\n"
        "基于 trace 指出的具体差距,formulate next_subgoal。"
    )


def _parse_json(content: str) -> dict[str, Any]:
    if not content:
        return {}
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return {}
    try:
        o = json.loads(m.group(0))
        return o if isinstance(o, dict) else {}
    except Exception:
        return {}


async def delegate_step(
    backend: Any,
    *,
    model: str,
    endpoint_id: str | None,
    task_goal: str,
    patch: dict[str, Any],
    brain_verify: dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> DelegateDecision:
    """主脑 Delegate 步:policy(确定性 action) + (redelegate/escalate 时)LLM formulate 下一子目标。

    ``brain_verify`` = BrainVerifyResult.to_dict() 的输出(Merge 步的 grounded 信号)。
    accept/give_up 不调 LLM(纯确定性);redelegate/escalate 调一次 LLM,formulate grounded 子目标。
    失败隔离:LLM 解析失败 → 返回 action 已定、next_subgoal 空的决策(不抛)。
    """
    bv = brain_verify or {}
    action = delegate_policy(
        test_green=bv.get("test_green"),
        trace_verdict=bv.get("trace_verdict"),
        weak_test_signal=bool(bv.get("weak_test_signal")),
        trace_missed=bool(bv.get("trace_missed")),
    )

    if action == "accept":
        return DelegateDecision(
            action="accept", reason="客观测试过 + forced-trace 认可", confidence=1.0, parse_ok=True,
        )
    if action == "give_up":
        return DelegateDecision(
            action="give_up", reason="无客观测试结果(budget/parse/无补丁)→ 无法 grounded 决策",
            confidence=None, parse_ok=True,
        )

    # redelegate / escalate 需要 trace.check 作 grounding。trace 调用失败(brain_verify 被 run_agent
    # 兜成 error dict,无 check)或 trace 没给出差距 → **不调 LLM**(空 check 下 LLM 只会 confabulate,
    # 违背「next_subgoal 必须 grounded in trace.check」承诺 —— Step A 反 confabulation 在失败路径的延续)。
    if not (bv.get("check") or "").strip():
        return DelegateDecision(
            action=action,
            reason="trace 无 check(trace 调用失败或未指出差距)→ 无法 ground next_subgoal,跳过 LLM 防编造",
            next_subgoal="", parse_ok=True,
        )

    # redelegate / escalate + 有 check → LLM formulate next_subgoal(grounded in trace.check)
    resp = await backend.complete(
        model=model, system=SYS_DELEGATE, user=_build_user(task_goal, patch, bv, action),
        endpoint_id=endpoint_id, thinking=False, temperature=temperature, max_tokens=max_tokens,
    )
    obj = _parse_json(resp.content or "")
    return DelegateDecision(
        action=action,
        next_subgoal=obj.get("next_subgoal", ""),
        target=obj.get("target", ""),
        reason=obj.get("reason", ""),
        confidence=obj.get("confidence"),
        parse_ok=bool(obj),
        cost_usd=float(getattr(resp, "cost_usd", 0.0) or 0.0),
        raw=(resp.content or "")[:200],
    )


async def delegate_from_trajectory(
    backend: Any,
    *,
    model: str,
    endpoint_id: str | None,
    goal: str,
    steps: list[dict[str, Any]],
    test_green: bool | None,
    brain_verify_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    """run loop 用:已有 brain_verify(Merge 步)+ trajectory steps → DelegateDecision.to_dict(进 run.json)。

    **test_green 用 traj.tests_green(verify_solution 跑出的可靠 ground truth)**,不用 brain_verify_dict
    里的 —— 后者在 trace 调用失败(被 run_agent 兜成 error dict)时会缺 test_green。trace 信号
    (trace_verdict/weak_test_signal/check)从 brain_verify_dict 取,失败时降级(空)。

    无 patch → 仍由 policy 出 action(多 give_up),但不调 LLM formulate(没补丁上下文)。
    吃原语,不 import Trajectory,防 delegate↔loop 循环依赖。
    """
    bv = brain_verify_dict or {}
    bv_signals = {
        "test_green": test_green,  # 可靠 ground truth
        "trace_verdict": bv.get("trace_verdict"),
        "weak_test_signal": bv.get("weak_test_signal", False),
        "trace_missed": bv.get("trace_missed", False),
        "trace": bv.get("trace", ""),
        "check": bv.get("check", ""),
    }
    patch = extract_final_patch({"steps": steps})
    if patch is None:
        action = delegate_policy(
            test_green=test_green, trace_verdict=bv_signals["trace_verdict"],
            weak_test_signal=bv_signals["weak_test_signal"], trace_missed=bv_signals["trace_missed"],
        )
        return DelegateDecision(
            action=action, reason="trajectory 无 apply_text_patch → 跳过 LLM formulate(无补丁上下文)",
            parse_ok=True,
        ).to_dict()
    decision = await delegate_step(
        backend, model=model, endpoint_id=endpoint_id,
        task_goal=goal, patch=patch, brain_verify=bv_signals,
    )
    return decision.to_dict()


# ============================================================================
# escalate handler —— 正交复查(Delegate 子系统的 handler,§15.1)
# ============================================================================
# escalate(test_green=True + weak_test_signal=True:客观测试过但 forced-trace 判 FAILED)的独立处理:
# 用**不同家族第二模型**盲审同 patch(forced_trace 换模型),作 tiebreaker 解「测试 SOLVED vs trace
# FAILED」冲突。Diagnosis(正交 verdict/check)→ Action(显式 resolve_escalate):正交 FAILED → redelegate
# (2 独立 FAILED 票 = 弱测试坐实);正交 SOLVED → accept(原 trace 过虑);未解析 → accept fallback(现状)。
# 死路(give_up)不新立 action —— 由 loop 的 no_progress 兜(连续同 check→停)。
# 复用 forced_trace(零新 prompt/解析);盲审 = 无原 trace 上下文(防锚定)。详见 plan / roadmap §15.8。
RESOLVE_ACTIONS = ("accept", "redelegate")  # escalate handler 行动词表(区别于 DELEGATE_ACTIONS)
ACCEPT_REASONS = ("normal", "weak_test", "orthogonal_cleared")


def _normalize_check(check: str) -> str:
    """归一化 check 供 gap_consensus / no_progress 比较。

    镜像 ``loop._normalize_check``;本地复刻防 delegate→loop 循环依赖(loop 懒 import delegate)。
    """
    return " ".join((check or "").split()).lower()[:200]


@dataclass
class Resolution:
    """escalate handler 输出:正交复查诊断 → 行动(Diagnosis/Action 分离,GPT)。"""

    action: str  # RESOLVE_ACTIONS 之一
    accept_reason: str = ""  # action=accept 时:normal / weak_test / orthogonal_cleared
    directive: str = ""  # action=redelegate 时:给专家的下一子目标(= 正交 check 差距,grounded)
    gap_consensus: bool = False  # 正交 check == 原 trace check(归一化):两 reviewer 找同一差距
    orthogonal_verdict: str | None = None  # 正交 forced-trace 判定(SOLVED/FAILED/None)
    cost_usd: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "accept_reason": self.accept_reason,
            "directive": self.directive,
            "gap_consensus": self.gap_consensus,
            "orthogonal_verdict": self.orthogonal_verdict,
            "cost_usd": self.cost_usd,
            "error": self.error,
        }


def resolve_escalate(
    *,
    orthogonal_verdict: str | None,
    orthogonal_check: str,
    trace_check: str,
) -> Resolution:
    """纯函数:正交诊断 → 行动(Diagnosis/Action 分离,零 LLM,零 confabulation)。

    - 正交 **SOLVED**(与原 trace 分歧,与客观测试一致)→ accept(``orthogonal_cleared``)。
    - 正交 **FAILED**(与原 trace 一致 = 2 独立 FAILED 票,弱测试坐实)→ redelegate
      (``directive`` = 正交 check 差距;ISS-016 类「缺 fsync」两票一致 → 重跑补 fsync)。
    - 正交 **None**(未解析/报错)→ accept fallback(``weak_test`` = 现状)。
    ``gap_consensus`` = 正交 check 与原 trace check 归一化相等(两 reviewer 找同一差距 vs 不同差距;
    GPT:不只看 verdict,要比 check —— 供 eval / 未来 within-round give_up 判据)。
    """
    gap_consensus = _normalize_check(orthogonal_check) == _normalize_check(trace_check)
    if orthogonal_verdict == "SOLVED":
        return Resolution(action="accept", accept_reason="orthogonal_cleared",
                          gap_consensus=gap_consensus, orthogonal_verdict=orthogonal_verdict)
    if orthogonal_verdict == "FAILED":
        return Resolution(action="redelegate", directive=str(orthogonal_check or ""),
                          gap_consensus=gap_consensus, orthogonal_verdict=orthogonal_verdict)
    return Resolution(action="accept", accept_reason="weak_test",
                      gap_consensus=gap_consensus, orthogonal_verdict=orthogonal_verdict)


async def orthogonal_check(
    backend: Any,
    *,
    orthogonal_model: str,
    orthogonal_endpoint_id: str | None,
    goal: str,
    test_req: str,
    patch: dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> TraceResult:
    """正交盲审:复用 ``forced_trace``(同 ``SYS_TRACE``、换 ``orthogonal_model``)。

    无原 trace 上下文(防锚定 —— 正交须独立判断,不看原 trace 说了什么)。
    """
    return await forced_trace(
        backend, model=orthogonal_model, endpoint_id=orthogonal_endpoint_id,
        goal=goal, test_req=test_req, patch=patch,
        temperature=temperature, max_tokens=max_tokens,
    )


async def resolve_escalate_from_trajectory(
    backend: Any,
    *,
    orthogonal_model: str,
    orthogonal_endpoint_id: str | None,
    goal: str,
    steps: list[dict[str, Any]],
    brain_verify_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    """run loop 用:escalate 时用正交第二模型盲审同 patch → ``resolve_escalate`` → ``Resolution.to_dict``(进 run.json)。

    **失败隔离**:orthogonal_check 报错 / 无 patch → accept fallback(``weak_test``,现状),不抛、不丢 run.json。
    ``test_req`` 用 ``goal``(与原 forced-trace 一致 —— run_agent 调 brain_verify_from_trajectory 时
    test_req 默认 goal)。吃原语(steps + brain_verify_dict),不 import Trajectory,防 delegate↔loop 循环依赖。
    """
    bv = brain_verify_dict or {}
    trace_check = str(bv.get("check", "") or "")
    patch = extract_final_patch({"steps": steps})
    if patch is None:  # 无 patch → 无法盲审 → accept fallback(不调 LLM)
        return Resolution(action="accept", accept_reason="weak_test",
                          error="trajectory 无 apply_text_patch → 跳过正交盲审").to_dict()
    try:
        tr = await orthogonal_check(
            backend, orthogonal_model=orthogonal_model, orthogonal_endpoint_id=orthogonal_endpoint_id,
            goal=goal, test_req=goal, patch=patch,
        )
    except Exception as exc:  # noqa: BLE001  sidecar:正交调用失败 → 兜,不崩主 run
        return Resolution(action="accept", accept_reason="weak_test",
                          error=f"orthogonal_check failed: {exc}").to_dict()
    res = resolve_escalate(orthogonal_verdict=tr.verdict, orthogonal_check=tr.check, trace_check=trace_check)
    res.cost_usd = float(getattr(tr, "cost_usd", 0.0) or 0.0)
    return res.to_dict()
