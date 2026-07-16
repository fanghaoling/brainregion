"""Bounded subprocess execution with process-tree cleanup."""
from __future__ import annotations

import ctypes
import os
import signal
import subprocess
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any, Sequence

PROCESS_CLEANUP_TIMEOUT_SEC = 5.0
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


@dataclass(frozen=True)
class ProcessTreeResult:
    returncode: int | None
    stdout: str | bytes | None
    stderr: str | bytes | None
    timed_out: bool = False
    cleanup_status: str = "not_needed"
    cleanup_error: str | None = None


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("per_process_user_time_limit", ctypes.c_int64),
        ("per_job_user_time_limit", ctypes.c_int64),
        ("limit_flags", wintypes.DWORD),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority_class", wintypes.DWORD),
        ("scheduling_class", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("read_operation_count", ctypes.c_uint64),
        ("write_operation_count", ctypes.c_uint64),
        ("other_operation_count", ctypes.c_uint64),
        ("read_transfer_count", ctypes.c_uint64),
        ("write_transfer_count", ctypes.c_uint64),
        ("other_transfer_count", ctypes.c_uint64),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("basic_limit_information", _JobObjectBasicLimitInformation),
        ("io_info", _IoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ]


def _windows_kernel32() -> Any:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _raise_windows_error(operation: str) -> None:
    error = ctypes.get_last_error()
    raise OSError(error, f"{operation} failed", None, error)


def _create_windows_job(process: subprocess.Popen[str]) -> tuple[Any, int]:
    kernel32 = _windows_kernel32()
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        _raise_windows_error("CreateJobObjectW")

    info = _JobObjectExtendedLimitInformation()
    info.basic_limit_information.limit_flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        handle,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        kernel32.CloseHandle(handle)
        _raise_windows_error("SetInformationJobObject")
    if not kernel32.AssignProcessToJobObject(handle, wintypes.HANDLE(int(process._handle))):
        kernel32.CloseHandle(handle)
        _raise_windows_error("AssignProcessToJobObject")
    return kernel32, handle


def _close_windows_job(job: tuple[Any, int]) -> None:
    kernel32, handle = job
    if not kernel32.CloseHandle(handle):
        _raise_windows_error("CloseHandle")


def _taskkill_process_tree(process: subprocess.Popen[str]) -> None:
    completed = subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=PROCESS_CLEANUP_TIMEOUT_SEC,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        raise OSError(f"taskkill failed with exit code {completed.returncode}: {detail}")


def _terminate_process_tree(
    process: subprocess.Popen[str],
    windows_job: tuple[Any, int] | None,
) -> tuple[str, str | None, tuple[Any, int] | None]:
    try:
        if os.name == "nt":
            if windows_job is not None:
                _close_windows_job(windows_job)
                return "terminated", None, None
            _taskkill_process_tree(process)
            return "terminated_fallback", None, None
        os.killpg(process.pid, signal.SIGKILL)
        return "terminated", None, None
    except (OSError, subprocess.SubprocessError) as exc:
        try:
            if process.poll() is None:
                process.kill()
        except OSError:
            pass
        return "failed", f"{type(exc).__name__}: {exc}", windows_job


def run_process_tree(
    command: Sequence[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: float,
) -> ProcessTreeResult:
    """Run one command in an isolated process tree and bound its lifetime."""
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        env=env,
        **popen_kwargs,
    )
    windows_job: tuple[Any, int] | None = None
    containment_error: str | None = None
    if os.name == "nt":
        try:
            windows_job = _create_windows_job(process)
        except OSError as exc:
            containment_error = f"{type(exc).__name__}: {exc}"

    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return ProcessTreeResult(process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        cleanup_status, cleanup_error, windows_job = _terminate_process_tree(process, windows_job)
        if containment_error and cleanup_error:
            cleanup_error = f"job containment unavailable ({containment_error}); {cleanup_error}"
        try:
            stdout, stderr = process.communicate(timeout=PROCESS_CLEANUP_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            stdout, stderr = exc.stdout, exc.stderr
            if cleanup_error is None:
                cleanup_error = "process pipes remained open after process-tree termination"
            cleanup_status = "failed"
        return ProcessTreeResult(
            None,
            stdout,
            stderr,
            timed_out=True,
            cleanup_status=cleanup_status,
            cleanup_error=cleanup_error,
        )
    finally:
        if windows_job is not None:
            try:
                _close_windows_job(windows_job)
            except OSError:
                pass
