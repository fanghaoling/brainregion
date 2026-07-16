from __future__ import annotations

import os
import subprocess

from brainregion.workspace import processes


class _FakeProcess:
    pid = 1234
    returncode = 0
    _handle = 4321

    def communicate(self, timeout):
        return "out", "err"


def test_run_process_tree_starts_an_isolated_process_group(monkeypatch):
    captured = {}

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(processes.subprocess, "Popen", fake_popen)
    if os.name == "nt":
        monkeypatch.setattr(processes, "_create_windows_job", lambda process: (object(), 1))
        monkeypatch.setattr(processes, "_close_windows_job", lambda job: None)

    result = processes.run_process_tree(["pytest", "--version"], cwd=".", env={}, timeout=1)

    assert result.returncode == 0
    if os.name == "nt":
        assert captured["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert captured["start_new_session"] is True


def test_timeout_terminates_process_tree_and_preserves_output(monkeypatch):
    class TimeoutProcess(_FakeProcess):
        returncode = None

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["pytest"], timeout, output="partial", stderr="warning")
            return "complete", "warning"

    process = TimeoutProcess()
    monkeypatch.setattr(processes.subprocess, "Popen", lambda *args, **kwargs: process)
    if os.name == "nt":
        monkeypatch.setattr(processes, "_create_windows_job", lambda child: (object(), 1))
        monkeypatch.setattr(processes, "_close_windows_job", lambda job: None)
    monkeypatch.setattr(
        processes,
        "_terminate_process_tree",
        lambda child, job: ("terminated", None, None),
    )

    result = processes.run_process_tree(["pytest"], cwd=".", env={}, timeout=1)

    assert result.timed_out is True
    assert result.cleanup_status == "terminated"
    assert result.cleanup_error is None
    assert result.stdout == "complete"
    assert result.stderr == "warning"
