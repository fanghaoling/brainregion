"""ISS-016 回归:apply_text_patch 须原子 + durable 写(temp + os.fsync + os.replace)。

非原子(直接 target.write_bytes 原地覆盖)→ crash 中途损坏文件;无 fsync → 掉电丢数据。
两条断言:① 不原地 write_bytes target(走 temp+os.replace 原子);② os.fsync temp 后才 replace(durable)。
"""
from __future__ import annotations

import os
from pathlib import Path

from brainregion.workspace import apply_text_patch, read_text
from brainregion.workspace.files import scoped_workspace_root


def test_apply_text_patch_does_not_overwrite_target_in_place(tmp_path, monkeypatch):
    """原子:不直接对 target write_bytes,应写 temp 再 os.replace。"""
    f = tmp_path / "target.txt"
    f.write_text("hello world\n", encoding="utf-8")
    written: list[str] = []
    real_write_bytes = Path.write_bytes

    def spy_write_bytes(self, data):
        written.append(str(self))
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", spy_write_bytes)
    with scoped_workspace_root(str(tmp_path)):
        sha = read_text("target.txt")["sha256"]
        apply_text_patch(
            "target.txt",
            expected_sha256=sha,
            replacements=[{"old_text": "hello", "new_text": "hi"}],
            dry_run=False,
        )
    target = os.path.normpath(str(f))
    direct = [os.path.normpath(p) for p in written if os.path.normpath(p) == target]
    assert not direct, (
        f"原地 write_bytes target(非原子,crash 中途损坏): {direct}; 应写 temp 再 os.replace。所有 write: {written}"
    )
    assert f.read_text(encoding="utf-8") == "hi world\n"


def test_apply_text_patch_fsyncs_before_replace(tmp_path, monkeypatch):
    """durable:os.replace 前 os.fsync temp 文件(防掉电丢数据)。"""
    f = tmp_path / "target.txt"
    f.write_text("hello world\n", encoding="utf-8")
    fsyncs: list[int] = []
    real_fsync = os.fsync

    def spy_fsync(fd):
        fsyncs.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr("brainregion.workspace.files.os.fsync", spy_fsync)
    with scoped_workspace_root(str(tmp_path)):
        sha = read_text("target.txt")["sha256"]
        apply_text_patch(
            "target.txt",
            expected_sha256=sha,
            replacements=[{"old_text": "hello", "new_text": "hi"}],
            dry_run=False,
        )
    assert fsyncs, "应 os.fsync temp 文件后再 os.replace(当前无 fsync = 不 durable)"
    assert f.read_text(encoding="utf-8") == "hi world\n"
