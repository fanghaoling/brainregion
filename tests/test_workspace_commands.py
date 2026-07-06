from __future__ import annotations

import sys

import pytest

from brainregion.workspace.commands import workspace_run_check


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

    monkeypatch.setattr("brainregion.workspace.commands.subprocess.run", boom)

    result = workspace_run_check([sys.executable, "-m", "pytest", "--version"])

    assert result["ok"] is False
    assert result["status"] == "launch_failed"
    assert result["exit_code"] is None
    assert "FileNotFoundError" in result["launch_error"]
    assert "FileNotFoundError" in result["stderr"]


def test_server_workspace_run_check_delegates(workspace_root):
    from brainregion import server

    result = server.workspace_run_check([sys.executable, "-m", "pytest", "--version"])

    assert result["ok"] is True
    assert result["kind"] == "python -m pytest"
