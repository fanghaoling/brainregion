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
