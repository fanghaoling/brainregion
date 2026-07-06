"""客观 verifier:在 run_dir 跑 pytest → tests-green 定 solved;gold_diff 仅诊断记录。

solved 以 tests-green 为准(review opus-2:精确 diff/gold 太脆,等价修复会误判 mismatch、
系统性低估 solve_rate 且偏置 brainregion 臂)。gold_diff 作人类诊断串记入 trajectory。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from brainregion.workspace import workspace_run_check
from brainregion.workspace.files import scoped_workspace_root

from .task import SandboxTask

# 状态机(对齐 outcome/eval 习惯,可序列化进 ledger):
# solved = pytest 全绿;tests_fail = pytest 非零;budget_exceeded/parse_error/launch_failed 由 loop 决定。
SOLVE_STATUS = ("solved", "tests_fail", "budget_exceeded", "parse_error", "launch_failed", "unknown")


def verify_solution(task: SandboxTask, run_dir: str | Path) -> dict[str, Any]:
    """在 run_dir 跑 pytest(收容到 run_dir),返回 {solve_status, tests_green, pytest, gold_diff}。"""
    run_dir = str(run_dir)
    argv = [sys.executable, "-m", "pytest", *task.test_args]
    with scoped_workspace_root(run_dir):
        result = workspace_run_check(argv, cwd=run_dir, timeout_sec=60, max_output_chars=20_000)
    status = result.get("status", "unknown")
    tests_green = status == "passed"
    solve_status = "solved" if tests_green else ("tests_fail" if status == "failed" else status)
    if solve_status not in SOLVE_STATUS:
        solve_status = "unknown"
    return {
        "solve_status": solve_status,
        "tests_green": tests_green,
        "pytest": {
            "status": status,
            "exit_code": result.get("exit_code"),
            "stdout_chars": len(result.get("stdout", "")),
            "stderr_chars": len(result.get("stderr", "")),
            "stdout_tail": (result.get("stdout", "") or "")[-1500:],
            "launch_error": result.get("launch_error"),
        },
        "gold_diff": task.gold_diff,
    }
