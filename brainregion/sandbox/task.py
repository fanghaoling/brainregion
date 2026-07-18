"""SandboxTask:一个「让测试过」任务的数据 schema。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SandboxTask:
    """agent 读码→定位 bug→打补丁→跑 pytest 转绿的任务。

    files/tests 是相对 tmp_dir 的 {path: content};agent 通过工作区工具在其中操作。
    gold_diff 仅人类诊断(solved 以 tests-green 为准,见 verify.py)。
    """

    id: str
    goal: str
    files: dict[str, str] = field(default_factory=dict)
    tests: dict[str, str] = field(default_factory=dict)
    test_args: list[str] = field(default_factory=lambda: ["-q"])
    gold_diff: str = ""
    gold_regions: list[str] = field(default_factory=list)
    seed_memory: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class WorktreeTask:
    """真实仓库任务:在 repo 的 worktree(独立检出 + 临时分支 off base_ref)里让 agent 修到 pytest 转绿。

    files/tests 不需要(来自仓库本身);goal 钉聚焦区(真仓库大,不钉则 agent 漫游);
    test_args 钉评测测试命令(常 ``["tests/test_x.py", "-q"]``)。
    bootstrap_commands 由 **harness**(非 agent;agent 受 allow-list 跑不了 uv/pip)跑配 env:
    ``None``=自动(``uv.lock`` 存在→``uv sync --extra dev``);``[]``=跳过;``[[...]]``=跑这些。
    gold_diff 仅 duck-compat(loop 里 ``traj.gold_diff=task.gold_diff``):真任务常无 gold diff,
    真「diff」价值在 run.json 抓的 git diff(见 worktree.capture_worktree_diff)。
    """

    id: str
    goal: str
    repo_path: str
    base_ref: str = "HEAD"
    test_args: list[str] = field(default_factory=lambda: ["-q"])
    bootstrap_commands: list[list[str]] | None = None
    expert_context_paths: list[str] = field(default_factory=list)
    seed_memory: list[dict[str, Any]] = field(default_factory=list)
    gold_diff: str = ""
    notes: str = ""
