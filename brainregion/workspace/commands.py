"""Safe workspace check command runner.

This module intentionally does not expose a general shell. It runs a small
allow-list of test/lint commands inside configured workspace roots and captures
bounded output for diagnosis.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from brainregion.runtime import emit_event

from .files import _allowed_roots, _is_relative_to, _resolve_target
from .processes import run_process_tree

DEFAULT_TIMEOUT_SEC = 60
HARD_TIMEOUT_SEC = 300
DEFAULT_MAX_OUTPUT_CHARS = 20_000
HARD_MAX_OUTPUT_CHARS = 200_000


def _normalize_argv(argv: list[str] | tuple[str, ...] | None) -> list[str]:
    if not argv:
        raise ValueError("argv must not be empty")
    normalized = [str(arg) for arg in argv if str(arg) != ""]
    if not normalized:
        raise ValueError("argv must not be empty")
    return normalized


def _basename(program: str) -> str:
    name = Path(program).name.casefold()
    return name[:-4] if name.endswith(".exe") else name


# python / python3 / python3.10 / python2.7(Linux/CI 的 sys.executable 常是版本化的 python3.x;
# 旧集合 {python,python3,py} 不认 python3.10 → _resolve_executable 误抛 PermissionError,CI 全红)
_VERSIONED_PYTHON = re.compile(r"^python\d*(\.\d+)?$")


def _looks_like_python(program: str) -> bool:
    name = _basename(program)
    return name == "py" or bool(_VERSIONED_PYTHON.match(name))


def _check_kind(argv: list[str]) -> str:
    program = _basename(argv[0])
    rest = argv[1:]
    if program == "pytest":
        return "pytest"
    if program == "ruff" and rest and rest[0] in {"check", "format"}:
        return f"ruff {rest[0]}"
    if _looks_like_python(argv[0]) and len(rest) >= 2 and rest[0] == "-m":
        module = rest[1]
        if module == "pytest":
            return "python -m pytest"
        if module == "ruff" and len(rest) >= 3 and rest[2] in {"check", "format"}:
            return f"python -m ruff {rest[2]}"
    if program == "uv" and len(rest) >= 2 and rest[0] == "run":
        runner = rest[1]
        if runner == "pytest":
            return "uv run pytest"
        if runner == "ruff" and len(rest) >= 3 and rest[2] in {"check", "format"}:
            return f"uv run ruff {rest[2]}"
    raise ValueError("command is not allowed; use pytest, ruff check/format, python -m pytest, or python -m ruff")


def _resolve_cwd(cwd: str) -> tuple[Path, dict[str, str]]:
    if cwd:
        target, root = _resolve_target(cwd)
        if not target.exists():
            raise FileNotFoundError(str(target))
        if not target.is_dir():
            raise NotADirectoryError(str(target))
        return target, root
    roots = _allowed_roots()
    if not roots:
        raise ValueError("no allowed workspace roots configured")
    root_path = Path(roots[0]["path"])
    return root_path, roots[0]


def _resolve_executable(argv: list[str], cwd: Path, root: dict[str, str]) -> list[str]:
    program = argv[0]
    if not any(sep in program for sep in ("/", "\\")) and not Path(program).is_absolute():
        return argv
    original = Path(program)
    if not original.is_absolute():
        original = cwd / program
    original = original.expanduser()  # 不 resolve —— 保留 venv symlink(见下)
    resolved = original.resolve(strict=False)  # resolve —— 仅用于 root 收容 + python 识别校验
    root_path = Path(root["path"])
    if _is_relative_to(resolved, root_path):
        if not resolved.exists():
            raise FileNotFoundError(str(resolved))
        return [str(resolved), *argv[1:]]
    if _looks_like_python(str(resolved)):
        # 用**原始路径**执行(非 resolved):Linux 上 .venv/bin/python 是 symlink → resolve 会跟到
        # 基底 CPython(无 venv site-packages,`python -m pytest` 会 "No module named pytest",CI 全红)。
        # 保留 symlink 让 venv(pyvenv.cfg)生效。Windows python.exe 非 symlink → original==resolved 无影响。
        # 安全:_looks_like_python 判的是 resolved basename(target 是不是 python),target 非则早 PermissionError。
        if not original.exists():
            raise FileNotFoundError(str(original))
        return [str(original), *argv[1:]]
    raise PermissionError("executable path is outside allowed workspace roots")


def _cap_output(text: str | bytes | None, max_chars: int) -> tuple[str, bool]:
    if text is None:
        return "", False
    value = text.decode("utf-8", errors="replace") if isinstance(text, bytes) else str(text)
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


def _relative_cwd(cwd: Path, root: dict[str, str]) -> str:
    root_path = Path(root["path"])
    if not _is_relative_to(cwd, root_path):
        return "."
    value = str(cwd.relative_to(root_path))
    return value or "."


def workspace_run_check(
    argv: list[str] | tuple[str, ...],
    *,
    cwd: str = "",
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> dict[str, Any]:
    """Run an allowed check command inside an allowed workspace root."""
    normalized_argv = _normalize_argv(argv)
    kind = _check_kind(normalized_argv)
    cwd_path, root = _resolve_cwd(cwd)
    command = _resolve_executable(normalized_argv, cwd_path, root)
    timeout_sec = max(1, min(int(timeout_sec or DEFAULT_TIMEOUT_SEC), HARD_TIMEOUT_SEC))
    max_output_chars = max(1, min(int(max_output_chars or DEFAULT_MAX_OUTPUT_CHARS), HARD_MAX_OUTPUT_CHARS))

    started = time.perf_counter()
    timed_out = False
    launch_error: str | None = None
    process_tree_cleanup = "not_needed"
    cleanup_error: str | None = None
    exit_code: int | None
    stdout = ""
    stderr = ""
    stdout_truncated = False
    stderr_truncated = False
    try:
        completed = run_process_tree(
            command,
            cwd=str(cwd_path),
            timeout=timeout_sec,
            env=os.environ.copy(),
        )
        exit_code = completed.returncode
        timed_out = completed.timed_out
        process_tree_cleanup = completed.cleanup_status
        cleanup_error = completed.cleanup_error
        stdout, stdout_truncated = _cap_output(completed.stdout, max_output_chars)
        stderr, stderr_truncated = _cap_output(completed.stderr, max_output_chars)
    except (subprocess.SubprocessError, OSError) as exc:
        # Missing/unrunnable executable or other launch failure: return a clean
        # failed status instead of bubbling up and crashing the MCP call.
        exit_code = None
        launch_error = f"{type(exc).__name__}: {exc}"
        stderr, stderr_truncated = _cap_output(launch_error, max_output_chars)

    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    if timed_out:
        status = "timeout"
    elif launch_error:
        status = "launch_failed"
    elif exit_code == 0:
        status = "passed"
    else:
        status = "failed"
    event_payload = {
        "kind": kind,
        "cwd": _relative_cwd(cwd_path, root),
        "status": status,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "timed_out": timed_out,
        "process_tree_cleanup": process_tree_cleanup,
        "cleanup_error": cleanup_error,
        "launch_error": launch_error,
        "stdout_chars": len(stdout),
        "stderr_chars": len(stderr),
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
    emit_event("workspace.check_run", payload=event_payload)
    return {
        "ok": status == "passed",
        "status": status,
        "kind": kind,
        "argv": normalized_argv,
        "cwd": str(cwd_path),
        "root": root,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "timeout_sec": timeout_sec,
        "process_tree_cleanup": process_tree_cleanup,
        "cleanup_error": cleanup_error,
        "launch_error": launch_error,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
