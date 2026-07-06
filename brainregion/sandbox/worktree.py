"""Worktree 模式:真实仓库任务的隔离生命周期(RAII)+ env 配备 + run artifact。

worktree = 真实仓库的一个独立检出(系统 temp 内,新临时分支 off ``base_ref``)。agent 在其中
read/search/patch/run-check;sandbox 既有的 ``scoped_workspace_root`` 把工具收容到 worktree 路径(见
workspace/files.py)。MVP 不 commit/push —— 改动留 worktree,清理前抓 git diff 入 run.json(replay-dataset
种子),然后丢弃。

安全:① worktree 建在**仓库外**系统 temp(git 不拒嵌套、主工作树无 untracked 污染);② 临时分支
``sandbox-<ts>`` 唯一;③ RAII(``with worktree(...)``)保证清理(异常/KeyboardInterrupt/return 都不漏);
④ bootstrap 由 harness 跑(非 agent,后者受 allow-list 约束)。git 子进程镜像 ``git/store.py:_real_runner``。
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("brainregion.sandbox.worktree")

_DIFF_CAP = 64_000
_STAT_CAP = 2_000


class WorktreeError(RuntimeError):
    """worktree 生命周期错误(非 repo / bad base_ref / worktree add 失败)。"""


@dataclass
class WorktreeHandle:
    """一个 worktree 的句柄。path=工作目录;branch=临时分支;repo_path=源仓库。"""

    path: str
    branch: str
    repo_path: str
    created: bool = True


def _run_git(args: list[str], cwd: str) -> tuple[int, str, str]:
    """走 subprocess(git),UTF-8 解码。镜像 ``git/store.py:_real_runner``(Windows gbk 会在中文上崩)。"""
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout, proc.stderr


def is_git_repo(path: str | os.PathLike[str]) -> bool:
    """``path`` 是否是 git 工作树(``git rev-parse --is-inside-work-tree``)。git 不可用 → False。"""
    try:
        rc, out, _err = _run_git(["rev-parse", "--is-inside-work-tree"], str(path))
    except (FileNotFoundError, OSError):
        return False
    return rc == 0 and (out or "").strip() == "true"


def create_worktree(
    repo_path: str | os.PathLike[str],
    base_ref: str = "HEAD",
    branch: str | None = None,
) -> WorktreeHandle:
    """在 repo 的 base_ref 上建一个独立 worktree(新临时分支 + 系统 temp 内的检出目录)。

    临时分支 ``sandbox-<ts>`` 唯一;worktree 目录用 ``tempfile.mkdtemp``(仓库外 temp → git 不拒嵌套、
    无 untracked 污染)。失败 → rmtree 空目录 + 抛 ``WorktreeError``。
    """
    repo = Path(repo_path).expanduser().resolve(strict=False)
    if not is_git_repo(repo):
        raise WorktreeError(f"not a git repo: {repo}")
    if branch is None:
        branch = f"sandbox-{int(time.time() * 1000)}"
    worktree_path = tempfile.mkdtemp(prefix=f"br-worktree-{branch}-")
    rc, _out, err = _run_git(
        ["worktree", "add", "-b", branch, worktree_path, base_ref],
        str(repo),
    )
    if rc != 0:
        shutil.rmtree(worktree_path, ignore_errors=True)
        raise WorktreeError(
            f"git worktree add 失败 (base={base_ref}, branch={branch}): {(err or '').strip()[:300]}"
        )
    return WorktreeHandle(path=worktree_path, branch=branch, repo_path=str(repo))


def remove_worktree(handle: WorktreeHandle) -> dict[str, bool]:
    """删 worktree 目录 + 临时分支(尽力清理,rc 不抛);rmtree 兜底。幂等(重复调不崩)。"""
    rc1, _out, err1 = _run_git(["worktree", "remove", "--force", handle.path], handle.repo_path)
    if rc1 != 0:
        logger.warning("worktree remove rc=%s: %s", rc1, (err1 or "").strip()[:200])
    rc2, _out, err2 = _run_git(["branch", "-D", handle.branch], handle.repo_path)
    if rc2 != 0:
        logger.warning("branch -D %s rc=%s: %s", handle.branch, rc2, (err2 or "").strip()[:200])
    # 兜底:git 没删干净就 rmtree(只删自己建的 handle.path)
    shutil.rmtree(handle.path, ignore_errors=True)
    return {"worktree_removed": rc1 == 0, "branch_removed": rc2 == 0}


@contextlib.contextmanager
def worktree(
    repo_path: str | os.PathLike[str],
    base_ref: str = "HEAD",
    branch: str | None = None,
    *,
    autoremove: bool = True,
):
    """RAII 上下文管理器(GPT ①):``create`` → ``yield handle`` → ``finally remove``。

    异常/KeyboardInterrupt/多层 return 都不漏清;与项目既有 ``scoped_workspace_root`` 同一套路。
    ``autoremove=False`` 时(--keep 检视)跳过清理并 log 保留路径。
    """
    handle = create_worktree(repo_path, base_ref, branch)
    try:
        yield handle
    finally:
        if autoremove:
            remove_worktree(handle)
        else:
            logger.info("kept worktree (autoremove=False): %s branch=%s", handle.path, handle.branch)


def bootstrap_worktree(
    handle: WorktreeHandle,
    bootstrap_commands: list[list[str]] | None,
) -> list[dict[str, Any]]:
    """harness(非 agent)在 worktree 跑 bootstrap(env 配备)。

    ``bootstrap_commands``:``None``=自动探测(``uv.lock`` 存在→``[["uv","sync","--extra","dev"]]``,
    否则 ``[]``);``[]``=跳过;``[[...]]``=跑这些。非零 rc → 记 warning **不抛**(降级诚实 tests_fail,
    不崩)。返每命令 ``{cmd, returncode, stderr_tail}``。
    """
    commands = bootstrap_commands
    if commands is None:
        commands = [["uv", "sync", "--extra", "dev"]] if Path(handle.path, "uv.lock").exists() else []
    results: list[dict[str, Any]] = []
    for cmd in commands:
        logger.info("worktree bootstrap: %s (cwd=%s)", cmd, handle.path)
        try:
            proc = subprocess.run(
                cmd,
                cwd=handle.path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            rc, stderr_tail = proc.returncode, (proc.stderr or "")[-500:]
        except (FileNotFoundError, OSError) as e:
            rc, stderr_tail = -1, f"{type(e).__name__}: {e}"
        if rc != 0:
            logger.warning("worktree bootstrap 失败 %s rc=%s: %s", cmd, rc, stderr_tail.strip()[:200])
        results.append({"cmd": list(cmd), "returncode": rc, "stderr_tail": stderr_tail})
    return results


def detect_venv_python(worktree_path: str | os.PathLike[str]) -> str | None:
    """探测 worktree 内 bootstrap 建出的 venv python(Win ``.venv/Scripts/python.exe`` / POSIX ``.venv/bin/python``)。"""
    wt = Path(worktree_path)
    for cand in (wt / ".venv" / "Scripts" / "python.exe", wt / ".venv" / "bin" / "python"):
        if cand.exists():
            return str(cand)
    return None


def capture_worktree_diff(handle: WorktreeHandle) -> dict[str, Any]:
    """清理前抓 worktree 的 git diff(agent 未暂存产物)+ diff --stat。均 cap。**必须在 remove 前调**。"""
    _rc, diff, _err = _run_git(["diff"], handle.path)
    _rc2, stat, _err2 = _run_git(["diff", "--stat"], handle.path)
    diff = diff or ""
    return {
        "produced_diff": diff[:_DIFF_CAP],
        "produced_diff_truncated": len(diff) > _DIFF_CAP,
        "diff_stat": (stat or "")[:_STAT_CAP],
    }


def write_worktree_run(payload: dict[str, Any], out_dir: str | os.PathLike[str] | None = None) -> Path:
    """把 run artifact 落 JSON 到 ``.brain-region/sandbox/<run_id>.json``(默认)。replay-dataset 种子。"""
    out = Path(out_dir) if out_dir else Path(".brain-region") / "sandbox"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{payload['run_id']}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
