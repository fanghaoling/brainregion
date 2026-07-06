"""Safe read-only workspace file tools.

These helpers are intentionally conservative: they only read inside configured
workspace roots, deny common secret/generated dependency paths, and cap returned
text. Write/edit tools should build on the same root and hash checks later.
"""
from __future__ import annotations

import hashlib
import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from brainregion import defaults
from brainregion.runtime import emit_event

DEFAULT_MAX_BYTES = 64_000
HARD_MAX_BYTES = 512_000
MAX_READ_FILE_BYTES = 2_000_000
MAX_INSPECT_SAMPLE = 8192
DEFAULT_SEARCH_MAX_RESULTS = 50
HARD_SEARCH_MAX_RESULTS = 500
DEFAULT_SEARCH_MAX_FILE_BYTES = 1_000_000
MAX_CONTEXT_LINES = 3
MAX_MATCH_TEXT_CHARS = 320

_DENIED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
_DENIED_SUFFIXES = {
    ".db",
    ".db-shm",
    ".db-wal",
    ".sqlite",
    ".sqlite3",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".der",
}


def _as_root_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [part for part in str(value).split(os.pathsep) if part.strip()]


def _canonical(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _configured_roots() -> list[tuple[Path, str]]:
    env_roots = _as_root_list(os.environ.get("BRAIN_REGION_WORKSPACE_ROOTS"))
    if env_roots:
        return [(Path(raw), "env:BRAIN_REGION_WORKSPACE_ROOTS") for raw in env_roots]

    try:
        config_roots = _as_root_list(defaults.apply().get("workspace_roots"))
    except Exception:
        config_roots = []
    if config_roots:
        return [(Path(raw), "config:workspace_roots") for raw in config_roots]

    fallback: list[tuple[Path, str]] = []
    if os.environ.get("UNITY_PROJECT_ROOT"):
        fallback.append((Path(os.environ["UNITY_PROJECT_ROOT"]), "env:UNITY_PROJECT_ROOT"))
    fallback.append((Path.cwd(), "cwd"))
    return fallback


def _allowed_roots() -> list[dict[str, str]]:
    roots: list[tuple[Path, str]] = []
    for raw, source in _configured_roots():
        path = _canonical(raw)
        if path.exists() and path.is_dir():
            roots.append((path, source))
    deduped = _dedupe_paths([path for path, _source in roots])
    by_path = {os.path.normcase(str(path)): source for path, source in roots}
    return [{"path": str(path), "source": by_path[os.path.normcase(str(path))]} for path in deduped]


def list_allowed_roots() -> dict[str, Any]:
    """Return the currently configured workspace roots."""
    roots = _allowed_roots()
    return {
        "ok": True,
        "roots": roots,
        "count": len(roots),
        "env_var": "BRAIN_REGION_WORKSPACE_ROOTS",
    }


def _deny_reason(path: Path) -> str | None:
    parts = {part.casefold() for part in path.parts}
    denied_dirs = sorted(parts & _DENIED_DIRS)
    if denied_dirs:
        return f"denied directory segment: {denied_dirs[0]}"

    name = path.name.casefold()
    if name == ".env" or name.startswith(".env."):
        return "denied sensitive env file"
    for suffix in _DENIED_SUFFIXES:
        if name.endswith(suffix):
            return f"denied sensitive/generated suffix: {suffix}"
    return None


def _resolve_target(path: str | os.PathLike[str]) -> tuple[Path, dict[str, str]]:
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("path must not be empty")

    roots = _allowed_roots()
    if not roots:
        raise ValueError("no allowed workspace roots configured")

    first_root = Path(roots[0]["path"])
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = first_root / candidate
    target = _canonical(candidate)

    for root in roots:
        root_path = Path(root["path"])
        if _is_relative_to(target, root_path):
            reason = _deny_reason(target)
            if reason:
                raise PermissionError(reason)
            return target, root
    raise PermissionError("path is outside allowed workspace roots")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_binary(sample: bytes) -> bool:
    return b"\0" in sample


def _text_type(path: Path) -> tuple[bool, str]:
    try:
        with path.open("rb") as f:
            sample = f.read(MAX_INSPECT_SAMPLE)
    except OSError:
        return False, "unreadable"
    if _looks_binary(sample):
        return False, "binary"
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False, "non_utf8"
    return True, "utf-8"


def inspect_file(path: str) -> dict[str, Any]:
    """Inspect a file path without returning file contents."""
    target, root = _resolve_target(path)
    root_path = Path(root["path"])
    exists = target.exists()
    result: dict[str, Any] = {
        "ok": True,
        "path": str(target),
        "relative_path": str(target.relative_to(root_path)) if _is_relative_to(target, root_path) else str(target),
        "root": root,
        "exists": exists,
        "is_file": target.is_file() if exists else False,
        "is_dir": target.is_dir() if exists else False,
    }
    if not exists:
        return result
    stat = target.stat()
    result["size_bytes"] = stat.st_size
    result["suffix"] = target.suffix
    if target.is_file():
        is_text, encoding = _text_type(target)
        result["is_text"] = is_text
        result["encoding"] = encoding
        result["sha256"] = _sha256_file(target)
    return result


def _line_window(lines: list[str], start_line: int, end_line: int | None) -> tuple[list[str], int, int]:
    total = len(lines)
    start = max(1, int(start_line or 1))
    end = total if end_line is None else max(start, int(end_line))
    return lines[start - 1:end], start, min(end, total)


def _cap_text(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    capped = encoded[:max_bytes]
    return capped.decode("utf-8", errors="ignore"), True


def _as_patterns(value: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        items = [value]
    else:
        items = [str(v) for v in value]
    return [item.replace("\\", "/") for item in items if item.strip()]


def _path_matches(path: Path, patterns: list[str]) -> bool:
    if not patterns:
        return False
    text = path.as_posix()
    name = path.name
    return any(fnmatch.fnmatchcase(text, pattern) or fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _trim_line(text: str) -> tuple[str, bool]:
    stripped = text.rstrip("\r\n")
    if len(stripped) <= MAX_MATCH_TEXT_CHARS:
        return stripped, False
    return stripped[:MAX_MATCH_TEXT_CHARS], True


def _compile_matcher(query: str, *, regex: bool, case_sensitive: bool):
    if not query:
        raise ValueError("query must not be empty")
    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags)
        except re.error as exc:
            raise ValueError(f"invalid regex query: {exc}") from exc
        return lambda line: bool(pattern.search(line))
    needle = query if case_sensitive else query.casefold()
    return lambda line: needle in (line if case_sensitive else line.casefold())


def _iter_candidate_files(
    roots: list[dict[str, str]],
    *,
    include_globs: list[str],
    exclude_globs: list[str],
    max_file_bytes: int,
    skipped: dict[str, int],
):
    for root in roots:
        root_path = Path(root["path"])
        for current, dirs, files in os.walk(root_path):
            current_path = Path(current)
            kept_dirs: list[str] = []
            for dirname in dirs:
                directory = current_path / dirname
                rel_dir = directory.relative_to(root_path)
                if _deny_reason(directory):
                    skipped["denied"] += 1
                    continue
                if _path_matches(rel_dir, exclude_globs):
                    skipped["excluded"] += 1
                    continue
                kept_dirs.append(dirname)
            dirs[:] = kept_dirs

            for filename in files:
                path = current_path / filename
                rel_path = path.relative_to(root_path)
                if _deny_reason(path):
                    skipped["denied"] += 1
                    continue
                if include_globs and not _path_matches(rel_path, include_globs):
                    skipped["not_included"] += 1
                    continue
                if _path_matches(rel_path, exclude_globs):
                    skipped["excluded"] += 1
                    continue
                try:
                    size_bytes = path.stat().st_size
                except OSError:
                    skipped["unreadable"] += 1
                    continue
                if size_bytes > max_file_bytes:
                    skipped["too_large"] += 1
                    continue
                yield path, root, rel_path, size_bytes


def _context_block(lines: list[str], line_index: int, context_lines: int) -> list[dict[str, Any]]:
    if context_lines <= 0:
        return []
    start = max(0, line_index - context_lines)
    end = min(len(lines), line_index + context_lines + 1)
    context: list[dict[str, Any]] = []
    for idx in range(start, end):
        text, truncated = _trim_line(lines[idx])
        context.append({"line": idx + 1, "text": text, "truncated": truncated})
    return context


def search_text(
    query: str,
    *,
    root: str = "",
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    case_sensitive: bool = False,
    regex: bool = False,
    max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
    context_lines: int = 0,
    max_file_bytes: int = DEFAULT_SEARCH_MAX_FILE_BYTES,
) -> dict[str, Any]:
    """Search UTF-8 text files inside allowed workspace roots."""
    matcher = _compile_matcher(query, regex=regex, case_sensitive=case_sensitive)
    include = _as_patterns(include_globs)
    exclude = _as_patterns(exclude_globs)
    max_results = max(1, min(int(max_results or DEFAULT_SEARCH_MAX_RESULTS), HARD_SEARCH_MAX_RESULTS))
    context_lines = max(0, min(int(context_lines or 0), MAX_CONTEXT_LINES))
    max_file_bytes = max(1, min(int(max_file_bytes or DEFAULT_SEARCH_MAX_FILE_BYTES), MAX_READ_FILE_BYTES))

    roots = _allowed_roots()
    if root:
        target, root_info = _resolve_target(root)
        if not target.exists():
            raise FileNotFoundError(str(target))
        if not target.is_dir():
            raise NotADirectoryError(str(target))
        roots = [{**root_info, "path": str(target)}]
    if not roots:
        raise ValueError("no allowed workspace roots configured")

    skipped: dict[str, int] = {
        "denied": 0,
        "excluded": 0,
        "not_included": 0,
        "too_large": 0,
        "binary": 0,
        "non_utf8": 0,
        "unreadable": 0,
    }
    matches: list[dict[str, Any]] = []
    scanned_files = 0
    matched_files: set[str] = set()
    truncated = False

    for path, root_info, rel_path, size_bytes in _iter_candidate_files(
        roots,
        include_globs=include,
        exclude_globs=exclude,
        max_file_bytes=max_file_bytes,
        skipped=skipped,
    ):
        try:
            raw = path.read_bytes()
        except OSError:
            skipped["unreadable"] += 1
            continue
        if _looks_binary(raw[:MAX_INSPECT_SAMPLE]):
            skipped["binary"] += 1
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped["non_utf8"] += 1
            continue

        scanned_files += 1
        lines = text.splitlines(keepends=True)
        for line_index, line in enumerate(lines):
            if not matcher(line):
                continue
            preview, preview_truncated = _trim_line(line)
            relative_path = rel_path.as_posix()
            matches.append(
                {
                    "path": str(path),
                    "relative_path": relative_path,
                    "root": root_info,
                    "line": line_index + 1,
                    "text": preview,
                    "text_truncated": preview_truncated,
                    "context": _context_block(lines, line_index, context_lines),
                    "size_bytes": size_bytes,
                }
            )
            matched_files.add(relative_path)
            if len(matches) >= max_results:
                truncated = True
                break
        if truncated:
            break

    emit_event(
        "workspace.text_search",
        payload={
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "regex": regex,
            "case_sensitive": case_sensitive,
            "matches": len(matches),
            "matched_files": len(matched_files),
            "scanned_files": scanned_files,
            "truncated": truncated,
        },
    )
    return {
        "ok": True,
        "query": query,
        "regex": regex,
        "case_sensitive": case_sensitive,
        "include_globs": include,
        "exclude_globs": exclude,
        "roots": roots,
        "scanned_files": scanned_files,
        "matched_files": len(matched_files),
        "count": len(matches),
        "max_results": max_results,
        "truncated": truncated,
        "skipped": skipped,
        "matches": matches,
    }


def read_text(
    path: str,
    *,
    start_line: int = 1,
    end_line: int | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Read UTF-8 text from an allowed workspace file."""
    target, root = _resolve_target(path)
    if not target.exists():
        raise FileNotFoundError(str(target))
    if not target.is_file():
        raise IsADirectoryError(str(target))
    size_bytes = target.stat().st_size
    if size_bytes > MAX_READ_FILE_BYTES:
        raise ValueError(f"file is too large to read safely: {size_bytes} bytes")

    max_bytes = max(1, min(int(max_bytes or DEFAULT_MAX_BYTES), HARD_MAX_BYTES))
    raw = target.read_bytes()
    if _looks_binary(raw[:MAX_INSPECT_SAMPLE]):
        raise ValueError("binary files are not supported")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8: {exc}") from exc

    root_path = Path(root["path"])
    lines = text.splitlines(keepends=True)
    selected, actual_start, actual_end = _line_window(lines, start_line, end_line)
    selected_text, truncated = _cap_text("".join(selected), max_bytes)
    result = {
        "ok": True,
        "path": str(target),
        "relative_path": str(target.relative_to(root_path)),
        "root": root,
        "encoding": "utf-8",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "total_lines": len(lines),
        "start_line": actual_start,
        "end_line": actual_end,
        "max_bytes": max_bytes,
        "truncated": truncated,
        "text": selected_text,
    }
    emit_event(
        "workspace.file_read",
        payload={
            "path": result["relative_path"],
            "size_bytes": result["size_bytes"],
            "start_line": actual_start,
            "end_line": actual_end,
            "truncated": truncated,
        },
    )
    return result
