"""沙盒:闭环 agent harness(§15 控制环 keystone 的 code-regime 验证场)。

固定主脑模型(deepseek/glm 等,非 Claude Code)经工作区工具(read/search/patch/run-check)
跑一个"让测试过"任务,observe→think→act 闭环。可插拔 BrainRegion 顾问臂(none/brainregion):
- ``none``:纯 loop(A/B 对照)。
- ``brainregion``:步首 wake_gate 路由 + 注入相关经验种子(给主脑)。

eval 模式 A/B 臂开/关,客观评测(tests green 为主)+ bootstrap CI gate(复用 eval/stats +
outcome OR-gate;cost 不当 primary,brainregion 臂因 wake/注入天然多一点开销)。

隔离:每 run 物化 fixture 到 tmp_dir,``scoped_workspace_root`` 把工作区工具收容到该目录
(ContextVar run-local,并发 eval 不串根)。路径收容复用 ``workspace.files._resolve_target``。
"""
from __future__ import annotations

from .isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from .loop import Trajectory, run_agent
from .option_runtime import ActivationRecord, CognitiveScheduler, OptionRegion, OptionResult
from .task import SandboxTask, WorktreeTask
from .verify import verify_solution
from .worktree import (
    WorktreeError,
    WorktreeHandle,
    bootstrap_worktree,
    capture_worktree_diff,
    create_worktree,
    detect_venv_python,
    is_git_repo,
    remove_worktree,
    worktree,
    write_worktree_run,
)

__all__ = [
    "SandboxTask",
    "WorktreeTask",
    "Trajectory",
    "materialize_fixture",
    "make_run_dir",
    "cleanup_run_dir",
    "run_agent",
    "OptionRegion",
    "OptionResult",
    "ActivationRecord",
    "CognitiveScheduler",
    "verify_solution",
    "WorktreeError",
    "WorktreeHandle",
    "bootstrap_worktree",
    "capture_worktree_diff",
    "create_worktree",
    "detect_venv_python",
    "is_git_repo",
    "remove_worktree",
    "worktree",
    "write_worktree_run",
]
