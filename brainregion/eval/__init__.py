"""v5.5 评测 harness（bootstrap 尺子）。

规约见 docs/eval_harness.zh-CN.md。MVP：三变体（retrieve off/on/garbage）+ 盲评 + SQLite ledger +
sanity（含负对照），用于在量 region routing 前先把尺子验证准。
"""
from __future__ import annotations

from .delegation import (
    ARM_TRIGGERED_SINGLE_EXPERT,
    DELEGATION_ARMS,
    DelegationEvalTask,
    DelegationRun,
    ExpertEvalResult,
    ExpertActivation,
    MainEvalResult,
    build_delegation_plan,
    run_delegation_eval,
    summarize_delegation_records,
)
from .runner import DEFAULT_VARIANTS, build_engines, run_eval
from .schema import (
    BlindJudgement,
    EvalCaseRecord,
    EvalLedgerEntry,
    EvalTask,
    VariantSpec,
)

__all__ = [
    "ARM_TRIGGERED_SINGLE_EXPERT",
    "DELEGATION_ARMS",
    "DelegationEvalTask",
    "DelegationRun",
    "ExpertEvalResult",
    "ExpertActivation",
    "MainEvalResult",
    "build_delegation_plan",
    "run_delegation_eval",
    "summarize_delegation_records",
    "DEFAULT_VARIANTS",
    "build_engines",
    "run_eval",
    "BlindJudgement",
    "EvalCaseRecord",
    "EvalLedgerEntry",
    "EvalTask",
    "VariantSpec",
]
