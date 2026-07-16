from __future__ import annotations

import sys

import pytest

from brainregion.workspace.commands import workspace_run_check
from brainregion.workspace.processes import ProcessTreeResult


@pytest.fixture()
def workspace_root(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_REGION_WORKSPACE_ROOTS", str(tmp_path))
    monkeypatch.setattr("brainregion.workspace.commands.emit_event", lambda *a, **k: {"ok": True})
    return tmp_path


def test_workspace_run_check_allows_python_pytest_version(workspace_root):
    result = workspace_run_check([sys.executable, "-m", "pytest", "--version"], timeout_sec=30)

    assert result["ok"] is True
    assert result["status"] == "passed"
    assert result["kind"] == "python -m pytest"
    assert result["exit_code"] == 0
    assert "pytest" in result["stdout"].lower()


def test_looks_like_python_recognizes_versioned_names():
    """CI(uv + Linux)sys.executable 常是 python3.10/3.11/3.12,旧集合不认 → _resolve_executable 误拒。"""
    from brainregion.workspace.commands import _looks_like_python

    for good in ["python", "python3", "python3.10", "python3.12", "python2.7", "py", "python.exe", "Python3.11"]:
        assert _looks_like_python(good), good
    for bad in ["pytest", "ruff", "python_evil", "notpython", "ruby"]:
        assert not _looks_like_python(bad), bad


def test_resolve_executable_preserves_venv_symlink(tmp_path, monkeypatch):
    """Linux .venv/bin/python 是 symlink → resolve 跟到基底 CPython(丢 venv 包);须保留 symlink 执行。"""
    import os as _os

    link = tmp_path.parent / "python3_venv_link"
    try:
        _os.symlink(sys.executable, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink 不可建(Windows 无权限或非 symlink venv 模式)")
    from brainregion.workspace.commands import _resolve_executable

    monkeypatch.setenv("BRAIN_REGION_WORKSPACE_ROOTS", str(tmp_path))
    # link 在 root 外;resolved basename = sys.executable 的 basename(python*)→ 放行;
    # 但执行路径必须保留 link(symlink),不能 resolve 到基底 CPython。
    cmd = _resolve_executable([str(link)], tmp_path, {"path": str(tmp_path)})
    assert cmd[0] == str(link), f"应保留 venv symlink 执行,got {cmd[0]}"


def test_workspace_run_check_supports_allowed_cwd(workspace_root):
    (workspace_root / "sub").mkdir()

    result = workspace_run_check([sys.executable, "-m", "pytest", "--version"], cwd="sub")

    assert result["ok"] is True
    assert result["cwd"].endswith("sub")


def test_workspace_run_check_reports_failed_check(workspace_root):
    result = workspace_run_check(
        [sys.executable, "-m", "pytest", "missing_test_file.py", "-q"],
        timeout_sec=30,
        max_output_chars=2000,
    )

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["exit_code"] != 0


def test_workspace_run_check_rejects_arbitrary_python_code(workspace_root):
    with pytest.raises(ValueError, match="command is not allowed"):
        workspace_run_check([sys.executable, "-c", "print('no')"])


def test_workspace_run_check_rejects_cwd_outside_root(workspace_root, tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)

    with pytest.raises(PermissionError, match="outside allowed workspace roots"):
        workspace_run_check([sys.executable, "-m", "pytest", "--version"], cwd=str(outside))


def test_workspace_run_check_rejects_non_python_executable_outside_root(workspace_root, tmp_path):
    outside = tmp_path.parent / "ruff.exe"
    outside.write_text("", encoding="utf-8")

    with pytest.raises(PermissionError, match="outside allowed workspace roots"):
        workspace_run_check([str(outside), "check"])


def test_workspace_run_check_caps_output_and_redacts_event_argv(workspace_root, monkeypatch):
    emitted = []
    monkeypatch.setattr("brainregion.workspace.commands.emit_event", lambda event_type, **fields: emitted.append((event_type, fields)))

    result = workspace_run_check([sys.executable, "-m", "pytest", "--version"], max_output_chars=5)

    assert result["stdout_truncated"] is True
    assert emitted[0][0] == "workspace.check_run"
    payload = emitted[0][1]["payload"]
    assert payload["kind"] == "python -m pytest"
    assert "argv" not in payload


def test_workspace_run_check_surfaces_missing_executable(workspace_root, monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("[Errno 2] No such file or directory: 'pytest'")

    monkeypatch.setattr("brainregion.workspace.commands.run_process_tree", boom)

    result = workspace_run_check([sys.executable, "-m", "pytest", "--version"])

    assert result["ok"] is False
    assert result["status"] == "launch_failed"
    assert result["exit_code"] is None
    assert "FileNotFoundError" in result["launch_error"]
    assert "FileNotFoundError" in result["stderr"]


def test_workspace_run_check_reports_process_tree_cleanup(workspace_root, monkeypatch):
    def timeout(*args, **kwargs):
        return ProcessTreeResult(
            returncode=None,
            stdout="partial output",
            stderr="",
            timed_out=True,
            cleanup_status="terminated",
        )

    monkeypatch.setattr("brainregion.workspace.commands.run_process_tree", timeout)

    result = workspace_run_check([sys.executable, "-m", "pytest", "-q"])

    assert result["status"] == "timeout"
    assert result["process_tree_cleanup"] == "terminated"
    assert result["cleanup_error"] is None
    assert result["stdout"] == "partial output"


def test_server_workspace_run_check_delegates(workspace_root):
    from brainregion import server

    result = server.workspace_run_check([sys.executable, "-m", "pytest", "--version"])

    assert result["ok"] is True
    assert result["kind"] == "python -m pytest"
