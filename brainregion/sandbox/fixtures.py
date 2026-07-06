"""内置「让测试过」fixtures:小模块 + 失败测试 + bug。

每个都要求 agent 多步(read→定位→patch→跑 pytest 才转绿),否则一行为修不掉,保证 A/B 有信号。
gold_diff 仅作人类诊断(solved 以 tests-green 为准,见 verify.py)。
"""
from __future__ import annotations

from .task import SandboxTask

# --- fixture 1:off-by-one(区间端点)---
_F1_FILES = {
    "ranges.py": '''"""整数区间求和工具。"""


def sum_range(start: int, end: int) -> int:
    """返回 start..end(含两端)所有整数的和。

    例:sum_range(1, 5) == 1+2+3+4+5 == 15;sum_range(5, 5) == 5。
    """
    total = 0
    for i in range(start, end):
        total += i
    return total
''',
}
_F1_TESTS = {
    "test_ranges.py": '''from ranges import sum_range


def test_full_range():
    assert sum_range(1, 5) == 15


def test_single_element():
    assert sum_range(5, 5) == 5


def test_two_elements():
    assert sum_range(2, 3) == 5
''',
}

# --- fixture 2:错异常类型(负数应抛 ValueError)---
_F2_FILES = {
    "parser.py": '''"""把字符串解析为正整数;负数/非整数应抛 ValueError。"""


def parse_positive(text: str) -> int:
    """解析 text 为整数;若是负数,抛 ValueError。"""
    return int(text)
''',
}
_F2_TESTS = {
    "test_parser.py": '''import pytest

from parser import parse_positive


def test_positive():
    assert parse_positive("42") == 42


def test_negative_raises():
    with pytest.raises(ValueError):
        parse_positive("-1")


def test_non_integer_raises():
    with pytest.raises(ValueError):
        parse_positive("abc")
''',
}

# --- fixture 3:漏边界(空列表应返 0.0,不该 ZeroDivisionError)---
_F3_FILES = {
    "stats.py": '''"""简单统计工具。"""


def average(values: list[float]) -> float:
    """返回均值;空列表返回 0.0(不该崩)。"""
    return sum(values) / len(values)
''',
}
_F3_TESTS = {
    "test_stats.py": '''from stats import average


def test_normal():
    assert average([1.0, 2.0, 3.0]) == 2.0


def test_empty_returns_zero():
    assert average([]) == 0.0


def test_single():
    assert average([5.0]) == 5.0
''',
}

SANDBOX_FIXTURES: list[SandboxTask] = [
    SandboxTask(
        id="off_by_one",
        goal="ranges.py 的 sum_range 求和区间有 off-by-one bug(漏掉 end),让 test_ranges.py 转绿。",
        files=_F1_FILES,
        tests=_F1_TESTS,
        gold_diff="range(start, end) → range(start, end + 1)",
        gold_regions=["debugging"],
        notes="要求读懂 docstring 的『含两端』契约 + 改 range 端点 + 跑测试验证。",
    ),
    SandboxTask(
        id="wrong_exception",
        goal="parser.py 的 parse_positive 对负数应抛 ValueError(现直接返回),让 test_parser.py 转绿。",
        files=_F2_FILES,
        tests=_F2_TESTS,
        gold_diff="int(text) 后加 if value < 0: raise ValueError(...)",
        gold_regions=["debugging"],
        notes="负数校验缺失;int() 已抛 ValueError 给非整数(test_non_integer 现状已过)。",
    ),
    SandboxTask(
        id="empty_edge_case",
        goal="stats.py 的 average 对空列表应返回 0.0(现 ZeroDivisionError),让 test_stats.py 转绿。",
        files=_F3_FILES,
        tests=_F3_TESTS,
        gold_diff="空列表守卫:if not values: return 0.0",
        gold_regions=["debugging"],
        notes="边界处理缺失。",
    ),
]


def get_fixture(task_id: str) -> SandboxTask:
    for task in SANDBOX_FIXTURES:
        if task.id == task_id:
            return task
    raise KeyError(f"unknown sandbox fixture: {task_id}")


def list_fixture_ids() -> list[str]:
    return [t.id for t in SANDBOX_FIXTURES]
