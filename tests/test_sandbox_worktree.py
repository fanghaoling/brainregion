"""沙盒 worktree 模式测试:RAII 生命周期 + 隔离 + diff 抓取 + run artifact + run_agent 对 worktree。

用 **temp git repo** fixture(`git init` + commit buggy 模块+测试)—— 不碰真仓库,CI-safe。
模块级 skip:git 不在 PATH 则跳过。loop 用 mock backend(不调模型);verifier 真跑 pytest。
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git 不在 PATH")

from brainregion.providers.base import ModelResponse  # noqa: E402
from brainregion.sandbox.loop import run_agent  # noqa: E402
from brainregion.sandbox.task import WorktreeTask  # noqa: E402
from brainregion.sandbox.worktree import (  # noqa: E402
    WorktreeError,
    bootstrap_worktree,
    capture_worktree_diff,
    create_worktree,
    detect_venv_python,
    is_git_repo,
    remove_worktree,
    worktree,
    write_worktree_run,
)
from brainregion.workspace import apply_text_patch, read_text  # noqa: E402
from brainregion.workspace.files import scoped_workspace_root  # noqa: E402

_RANGES_BUGGY = (
    "def sum_range(start, end):\n"
    "    total = 0\n"
    "    for i in range(start, end):\n"
    "        total += i\n"
    "    return total\n"
)
_RANGES_TEST = (
    "from ranges import sum_range\n"
    "\n"
    "def test_single():\n"
    "    assert sum_range(5, 5) == 5\n"
)


def _git(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    p = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert p.returncode == 0, f"git {args} 失败: {p.stderr}"
    return p


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(["init"], str(r))
    _git(["config", "user.email", "t@t.test"], str(r))
    _git(["config", "user.name", "test"], str(r))
    (r / "ranges.py").write_text(_RANGES_BUGGY, encoding="utf-8")
    (r / "test_ranges.py").write_text(_RANGES_TEST, encoding="utf-8")
    _git(["add", "."], str(r))
    _git(["commit", "-m", "init"], str(r))
    return r


def _branch_exists(repo: Path, branch: str) -> bool:
    out = subprocess.run(
        ["git", "branch", "--list", branch], cwd=str(repo), capture_output=True, text=True,
    ).stdout
    return branch in out


# ---------- is_git_repo ----------

def test_is_git_repo_true_for_repo_false_for_plain_dir(repo: Path, tmp_path: Path):
    assert is_git_repo(repo) is True
    empty = tmp_path / "empty"
    empty.mkdir()
    assert is_git_repo(empty) is False


# ---------- create / remove ----------

def test_create_worktree_checks_out_base(repo: Path):
    h = create_worktree(repo)
    try:
        assert h.path != str(repo)
        assert h.branch.startswith("sandbox-")
        assert Path(h.path, "ranges.py").exists()  # base 检入
        assert Path(h.path, "test_ranges.py").exists()
    finally:
        remove_worktree(h)


def test_create_worktree_rejects_non_repo(tmp_path: Path):
    with pytest.raises(WorktreeError, match="not a git repo"):
        create_worktree(tmp_path / "nope")


def test_create_worktree_rejects_bad_base(repo: Path):
    with pytest.raises(WorktreeError, match="worktree add"):
        create_worktree(repo, base_ref="nonexistent-ref-xyz")


def test_remove_worktree_cleans_dir_and_branch(repo: Path):
    h = create_worktree(repo)
    p, branch = h.path, h.branch
    remove_worktree(h)
    assert not Path(p).exists()
    assert not _branch_exists(repo, branch)


def test_remove_worktree_idempotent(repo: Path):
    h = create_worktree(repo)
    remove_worktree(h)
    remove_worktree(h)  # 二次不崩


def test_two_worktrees_distinct(repo: Path):
    h1 = create_worktree(repo)
    h2 = create_worktree(repo)
    try:
        assert h1.branch != h2.branch
        assert h1.path != h2.path
    finally:
        remove_worktree(h1)
        remove_worktree(h2)


# ---------- RAII context manager ----------

def test_worktree_cm_cleans_on_exception(repo: Path):
    with pytest.raises(RuntimeError, match="boom"):
        with worktree(repo) as h:
            branch, p = h.branch, h.path
            raise RuntimeError("boom")
    assert not Path(p).exists()
    assert not _branch_exists(repo, branch)


def test_worktree_cm_autoremove_false_keeps(repo: Path):
    with worktree(repo, autoremove=False) as h:
        p, branch = h.path, h.branch
    assert Path(p).exists()  # 保留
    assert _branch_exists(repo, branch)
    remove_worktree(h)  # 手动清


# ---------- isolation ----------

def test_scoped_root_isolates_worktree_from_repo(repo: Path):
    """worktree 内 patch 不影响原仓库文件。"""
    with worktree(repo) as h:
        with scoped_workspace_root(h.path):
            sha = read_text("ranges.py")["sha256"]
            apply_text_patch(
                "ranges.py", expected_sha256=sha,
                replacements=[{"old_text": "for i in range(start, end):", "new_text": "for i in range(start, end + 1):"}],
                dry_run=False,
            )
    assert "range(start, end + 1)" not in (repo / "ranges.py").read_text(encoding="utf-8")


# ---------- diff capture ----------

def test_capture_diff_after_edit(repo: Path):
    with worktree(repo) as h:
        with scoped_workspace_root(h.path):
            sha = read_text("ranges.py")["sha256"]
            apply_text_patch(
                "ranges.py", expected_sha256=sha,
                replacements=[{"old_text": "for i in range(start, end):", "new_text": "for i in range(start, end + 1):"}],
                dry_run=False,
            )
        d = capture_worktree_diff(h)
        assert "range(start, end + 1)" in d["produced_diff"]
        assert d["diff_stat"]


def test_capture_diff_empty_when_unchanged(repo: Path):
    with worktree(repo) as h:
        d = capture_worktree_diff(h)
        assert d["produced_diff"] == ""


# ---------- bootstrap / detect_venv ----------

def test_bootstrap_none_auto_no_uvlock_is_noop(repo: Path):
    """无 uv.lock → 自动探测得 [](无命令)。"""
    with worktree(repo) as h:
        results = bootstrap_worktree(h, None)
        assert results == []


def test_detect_venv_python_none_when_absent(repo: Path):
    with worktree(repo) as h:
        assert detect_venv_python(h.path) is None


# ---------- run artifact ----------

def test_write_worktree_run_drops_json(tmp_path: Path):
    payload = {"run_id": "worktree-test", "mode": "worktree", "trajectory": {"task_id": "x"}}
    path = write_worktree_run(payload, out_dir=tmp_path)
    assert path.name == "worktree-test.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "worktree-test"


# ---------- run_agent against a worktree (mock backend) ----------

class _MockBackend:
    """按脚本返 tool-call(sha 来自 worktree 的 ranges.py);不调模型。"""

    def __init__(self, sha: str):
        self.sha = sha
        self.i = 0

    async def complete_messages(self, messages, **kw):
        seq = [
            json.dumps({"thought": "read", "tool": "read_text", "args": {"path": "ranges.py"}}),
            json.dumps({
                "thought": "fix", "tool": "apply_text_patch", "args": {
                    "path": "ranges.py", "expected_sha256": self.sha,
                    "replacements": [{"old_text": "for i in range(start, end):", "new_text": "for i in range(start, end + 1):"}],
                    "dry_run": False,
                },
            }),
            json.dumps({"thought": "test", "tool": "workspace_run_check",
                        "args": {"argv": [sys.executable, "-m", "pytest", "-q"]}}),
            json.dumps({"thought": "done", "done": True, "answer": "fixed off-by-one"}),
        ]
        content = seq[min(self.i, len(seq) - 1)]
        self.i += 1
        return ModelResponse(model=kw.get("model", "mock"), content=content, usage={}, cost_usd=0.0)

    async def complete(self, *, system, user, **kw):
        # brain_verify 的 forced-trace 用:补丁修了 off-by-one → SOLVED
        content = json.dumps({"trace": "range(start, end+1) 修了 off-by-one",
                              "check": "sum_range 现在含 end",
                              "verdict": "SOLVED", "confidence": 0.9})
        return ModelResponse(model=kw.get("model", "mock"), content=content, usage={}, cost_usd=0.0)


def test_run_agent_against_worktree_solves(repo: Path):
    task = WorktreeTask(
        id="wt-test",
        goal="ranges.py 的 sum_range 有 off-by-one bug,让 test_ranges.py 转绿。",
        repo_path=str(repo),
        test_args=["-q"],
    )
    with worktree(repo) as h:
        with scoped_workspace_root(h.path):
            sha = read_text("ranges.py")["sha256"]
        traj = asyncio.run(
            run_agent(_MockBackend(sha), "mock", task, run_dir=h.path, arm="none",
                      python_exe=sys.executable, brain_verify=True, brain_delegate=True)
        )
        assert traj.solve_status == "solved"
        assert traj.tests_green
        # §15.8 brain_verify 接 run loop:forced-trace 跑了 + 与客观测试 agree + 序列化进 to_dict
        assert traj.brain_verify is not None
        assert traj.brain_verify["trace_verdict"] == "SOLVED"
        assert traj.brain_verify["test_green"] is True
        assert traj.brain_verify["agree"] is True
        assert traj.to_dict()["brain_verify"]["final_verdict"] == "SOLVED"
        # §15.1 brain_delegate 接 run loop:solved → accept(确定性,不调 LLM)+ 序列化
        assert traj.delegate is not None
        assert traj.delegate["action"] == "accept"
        assert traj.to_dict()["delegate"]["action"] == "accept"


class _MockBackendRaiseTrace(_MockBackend):
    """同 _MockBackend(能解 task)但 forced-trace 的 complete 抛异常 —— 验 brain_verify 失败隔离。"""

    async def complete(self, *, system, user, **kw):
        raise RuntimeError("simulated trace backend explosion")


def test_run_agent_brain_verify_failure_isolated(repo: Path):
    """brain_verify 是 sidecar:其 LLM 调用抛异常时,run_agent 不得崩、不得丢主 run 的成果。"""
    task = WorktreeTask(
        id="wt-test",
        goal="ranges.py 的 sum_range 有 off-by-one bug,让 test_ranges.py 转绿。",
        repo_path=str(repo),
        test_args=["-q"],
    )
    with worktree(repo) as h:
        with scoped_workspace_root(h.path):
            sha = read_text("ranges.py")["sha256"]
        traj = asyncio.run(
            run_agent(_MockBackendRaiseTrace(sha), "mock", task, run_dir=h.path, arm="none",
                      python_exe=sys.executable, brain_verify=True)
        )
    # 主 run 不受 brain_verify 失败影响
    assert traj.solve_status == "solved"
    assert traj.tests_green
    # brain_verify 失败被显式隔离:run_agent 不抛,brain_verify 段记 error(不丢 run.json/diff)
    assert traj.brain_verify is not None
    assert traj.brain_verify.get("trace_verdict") is None
    assert str(traj.brain_verify.get("error", "")).startswith("brain_verify failed")
    assert traj.to_dict()["brain_verify"]["error"].startswith("brain_verify failed")
