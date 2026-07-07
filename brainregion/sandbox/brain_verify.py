"""主脑 grounding-first 验证器(§15.8):专家(沙盒)补丁的 forced-trace + 客观测试 backstop。

经 Step A/A.5/A.6 实证(见 memory eval-harness-state §15 / roadmap §15.8),主脑验证的 grounding
层级 = **客观测试 > forced-trace > 正交 2-LLM > 单 plain > 脑补**。本模块落地「trace-first +
test-backstop」认知环验证核:

- :func:`forced_trace` —— LLM 一把,强制逐行 trace 补丁是否满足测试要求(廉价中间层;A.6 实测
  deepseek 0.67→1.0、glm 0.56→0.78)。把验证拆成可观察的 trace 步(对齐 §15.6 多轮脚手架)。
- :func:`composite_verify` —— trace 判定 ⊕ 客观 ``tests_green`` → agree 标志 + 弱测试信号
  (测试过但 trace 判 FAILED = 测试可能不够强,ISS-016 教训)+ trace 漏检(测试败但 trace 过)。

``test_green`` 由调用方提供(沙盒 :func:`verify_solution` 跑出来,或 run.json 里存的);本模块**不
自己跑 pytest**,保持单一职责 + 可单测。:func:`verify_with_brain` 是现成编排(verify_solution +
forced_trace + composite),供认知环 loop 直接调。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .task import SandboxTask, WorktreeTask
from .verify import verify_solution

# forced-trace 系统 prompt(A.6 验证通过的形):强制「从代码推变量 → 逐条对照断言 → 无法确定则
# FAILED」,治 plain-judge 的 grounding 失败(看到 except 子句就编下游 ok/status 对)。
SYS_TRACE = (
    "你是一名主程,审查一位专家的补丁。你将看到:任务目标、目标测试的具体要求、补丁(old_text → new_text)。\n"
    "你必须按以下 grounded trace 流程逐步核实,严禁跳步、严禁凭「看起来对」下结论:\n"
    "1. 触发路径:针对测试要求的那个触发场景(如『subprocess.run 抛 FileNotFoundError』或『evil regex (a+)+b』),"
    "逐行确定补丁后代码会走哪条分支、执行哪些语句。\n"
    "2. 变量追踪:在该分支上,**从补丁的实际代码**显式推出关键变量的值/调用"
    "(如 exit_code=? 、timed_out=? 、是否调用了 os.fsync 、status 字段取哪个字面量、是否 raise 了 ValueError)。"
    "必须从代码推出,不得假设『后续逻辑应该会…』—— 看不到的就不算满足。\n"
    "3. 逐条对照断言:把第2步推出的值,逐条对照测试要求的每一条(如 ok 必须 is False、status 必须 == 'launch_failed'、"
    "必须调用 os.fsync、必须 raise ValueError 含 'catastrophic backtracking')。任一条不符 → 判 FAILED。\n"
    "4. 判定:只有所有断言都**从代码可确定地满足**才 SOLVED;任一不符、或无法从代码确定 → FAILED。\n"
    "输出严格 JSON:\n"
    '{"trace": "第2步:从代码推出的变量值/调用,逐个列", "check": "第3步:每条断言 满足/不满足", '
    '"verdict": "SOLVED" 或 "FAILED", "confidence": 0.0-1.0, "reason": "一句话"}'
)


@dataclass
class TraceResult:
    """一次 forced-trace 调用的结果(失败隔离后,不向上抛)。"""

    verdict: str | None  # "SOLVED" | "FAILED" | None(未解析出)
    trace: str = ""
    check: str = ""
    confidence: float | None = None
    parse_ok: bool = False
    error: str | None = None
    raw: str = ""


@dataclass
class BrainVerifyResult:
    """trace-first + test-backstop 的复合判定。"""

    trace_verdict: str | None
    test_green: bool | None
    agree: bool | None  # trace 与测试一致(None = 至少一方缺)
    weak_test_signal: bool  # 测试过但 trace 判 FAILED(测试可能不够强)
    trace_missed: bool  # 测试败但 trace 判 SOLVED(trace 不可靠,以测试为准)
    trace: TraceResult
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str | None:
        """最终判定:以客观测试为准(test_green);无测试时退回 trace。"""
        if self.test_green is True:
            return "SOLVED"
        if self.test_green is False:
            return "FAILED"
        return self.trace_verdict


# ---------------- patch 抽取 / 渲染 ----------------
def extract_final_patch(trajectory: dict[str, Any]) -> dict[str, Any] | None:
    """从 trajectory 取最后一个**成功应用**的 apply_text_patch 步的 {path, replacements}。

    无 apply_text_patch 或全部 error → None。
    """
    steps = trajectory.get("steps") or []
    patch: dict[str, Any] | None = None
    for s in steps:
        if s.get("tool") != "apply_text_patch":
            continue
        args = s.get("args") or {}
        reps = args.get("replacements")
        if reps and not s.get("error"):
            patch = {"path": args.get("path", "?"), "replacements": reps}
    return patch


def render_patch(patch: dict[str, Any]) -> str:
    lines = [f"文件: {patch.get('path', '?')}"]
    for r in patch.get("replacements", []):
        lines.append("  old_text:")
        lines.append("    " + str(r.get("old_text", "")).replace("\n", "\n    "))
        lines.append("  new_text:")
        lines.append("    " + str(r.get("new_text", "")).replace("\n", "\n    "))
    return "\n".join(lines)


def _build_user(goal: str, test_req: str, patch: dict[str, Any]) -> str:
    return (
        f"【任务目标】\n{goal}\n\n"
        f"【目标测试的具体要求】\n{test_req}\n\n"
        f"【专家的补丁】\n{render_patch(patch)}\n\n"
        f"按 grounded trace 流程判断这个补丁是否满足测试要求。"
    )


def _parse_trace(content: str) -> tuple[str | None, dict[str, Any]]:
    if not content:
        return None, {}
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            if isinstance(o, dict):
                v = str(o.get("verdict", "")).strip().upper()
                if v in ("SOLVED", "FAILED"):
                    return v, o
        except Exception:
            pass
    m2 = re.search(r"\b(SOLVED|FAILED)\b", content.upper())
    if m2:
        return m2.group(1), {}
    return None, {}


# ---------------- 核心:forced-trace + composite ----------------
async def forced_trace(
    backend: Any,
    *,
    model: str,
    endpoint_id: str | None,
    goal: str,
    test_req: str,
    patch: dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> TraceResult:
    """主脑 forced-trace:对单个补丁跑 SYS_TRACE,返回解析后的 TraceResult。"""
    resp = await backend.complete(
        model=model,
        system=SYS_TRACE,
        user=_build_user(goal, test_req, patch),
        endpoint_id=endpoint_id,
        thinking=False,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    verdict, obj = _parse_trace(resp.content or "")
    return TraceResult(
        verdict=verdict,
        trace=(obj or {}).get("trace", ""),
        check=(obj or {}).get("check", ""),
        confidence=(obj or {}).get("confidence"),
        parse_ok=verdict is not None,
        error=getattr(resp, "error", None),
        raw=(resp.content or "")[:200],
    )


def composite_verify(trace: TraceResult, test_green: bool | None) -> BrainVerifyResult:
    """纯函数:trace 判定 ⊕ 客观 tests_green → 复合判定。

    - agree:trace 与测试结论一致(双方都有时;None=至少一方缺)。
    - weak_test_signal:测试过但 trace 判 FAILED → 测试可能不够强(ISS-016 教训:弱测试放行部分修复)。
    - trace_missed:测试败但 trace 判 SOLVED → trace 不可靠(模型相关,A.6 glm 弱),以测试为准。
    """
    tv = trace.verdict
    agree: bool | None = None
    if tv is not None and test_green is not None:
        agree = (tv == "SOLVED") == test_green
    weak = bool(test_green is True and tv == "FAILED")
    missed = bool(test_green is False and tv == "SOLVED")
    notes: list[str] = []
    if weak:
        notes.append("弱测试信号:客观测试过但主脑 trace 判 FAILED(测试可能不够强,见 ISS-016 教训)")
    if missed:
        notes.append("trace 漏检:客观测试败但 trace 判 SOLVED(trace 不可靠,须以测试为准)")
    if tv is None:
        notes.append("trace 输出未解析出 verdict(parse 失败或调用 error)")
    if test_green is None:
        notes.append("无客观测试结果(无测试/未跑)→ 最终判定退回 trace")
    return BrainVerifyResult(
        trace_verdict=tv, test_green=test_green, agree=agree,
        weak_test_signal=weak, trace_missed=missed, trace=trace, notes=notes,
    )


async def verify_with_brain(
    backend: Any,
    *,
    task: SandboxTask | WorktreeTask,
    run_dir: str | Path,
    trajectory: dict[str, Any],
    model: str,
    endpoint_id: str | None,
    test_req: str | None = None,
    python_exe: str | None = None,
) -> BrainVerifyResult:
    """认知环验证核编排:客观测试(verify_solution)+ forced-trace → composite。

    test_req 默认取 task.goal(production 用法建议传精确测试要求,如从测试文件抽取)。
    无 patch → 跳过 trace,仅返客观测试。注意:``verify_solution`` 同步跑 pytest,会阻塞事件循环
    (与 sandbox loop 末尾调用同模式);并发场景需自行包 executor。
    """
    goal = getattr(task, "goal", "")
    test_req = test_req or goal
    test = verify_solution(task, run_dir, python_exe=python_exe)
    test_green = test.get("tests_green")

    patch = extract_final_patch(trajectory)
    if patch is None:
        tr = TraceResult(verdict=None, error="no apply_text_patch in trajectory")
        res = composite_verify(tr, test_green)
        res.notes.insert(0, "trajectory 无 apply_text_patch → 跳过 trace,仅客观测试")
        return res

    tr = await forced_trace(
        backend, model=model, endpoint_id=endpoint_id,
        goal=goal, test_req=test_req, patch=patch,
    )
    return composite_verify(tr, test_green)
