"""沙盒隔离:fixture 物化 + 路径校验。

每 run 物化 ``SandboxTask.files/tests`` 到一个 tmp_dir,``scoped_workspace_root`` 把工作区
工具收容到该目录(见 workspace/files.py)。spec 路径校验(review gpt-5):穿越发生在工具收容
**之前**(物化是 harness 直接写),故在此独立拒绝对/``..``/symlink-outside。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .task import SandboxTask


class FixturePathError(ValueError):
    """fixture spec 里某条路径会写出 tmp_dir。"""


def _safe_join(tmp_dir: Path, rel: str) -> Path:
    """把相对路径安全 join 到 tmp_dir;拒绝对路径 / ``..`` 穿越 / 解析后越界 / symlink 出界。"""
    if not rel:
        raise FixturePathError("empty fixture path")
    candidate = tmp_dir / rel
    # 先拒明显的绝对路径和 .. 片段(早失败、信息清),再 resolve 兜底(symlink/绕路)。
    if Path(rel).is_absolute():
        raise FixturePathError(f"absolute path not allowed in fixture: {rel}")
    parts = [p for p in Path(rel).parts if p not in (".",)]
    if any(part == ".." for part in parts):
        raise FixturePathError(f"parent traversal not allowed in fixture: {rel}")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(tmp_dir.resolve(strict=False))
    except ValueError as exc:
        raise FixturePathError(f"fixture path resolves outside tmp_dir: {rel}") from exc
    return resolved


def materialize_fixture(task: SandboxTask, tmp_dir: Path) -> None:
    """把 task.files + task.tests 写入 tmp_dir(路径校验后创建父目录 + 写文件)。"""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for rel, content in {**task.files, **task.tests}.items():
        target = _safe_join(tmp_dir, rel.replace("\\", "/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def make_run_dir(prefix: str = "brainregion-sandbox-") -> str:
    """per-run 临时目录(失败留检、成功由调用方清)。"""
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_run_dir(run_dir: str | os.PathLike[str]) -> None:
    """成功后清掉 run_dir(递归)。失败时调用方不调 → 留检。"""
    import shutil

    shutil.rmtree(run_dir, ignore_errors=True)
