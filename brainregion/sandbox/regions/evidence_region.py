"""Grounded, model-free evidence collection for code sandbox tasks."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from brainregion.core.context import ContextBlock

from ..task import SandboxTask, WorktreeTask

_PATH_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+")
_TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".json",
        ".md",
        ".py",
        ".rs",
        ".toml",
        ".ts",
        ".tsx",
        ".yaml",
        ".yml",
    }
)


@dataclass(frozen=True)
class EvidenceRequest:
    path: str
    max_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"action": "read_text", "path": self.path, "max_bytes": self.max_bytes}


class EvidenceRegion:
    """Select explicit task paths and turn host reads into grounded evidence blocks."""

    name = "evidence"
    access_mode = "grounded"
    uses_model = False

    def __init__(self, *, max_files: int = 4, max_bytes_per_file: int = 12000) -> None:
        if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files <= 0:
            raise ValueError("max_files must be a positive integer")
        if (
            isinstance(max_bytes_per_file, bool)
            or not isinstance(max_bytes_per_file, int)
            or max_bytes_per_file <= 0
        ):
            raise ValueError("max_bytes_per_file must be a positive integer")
        self.max_files = max_files
        self.max_bytes_per_file = max_bytes_per_file
        self._results: list[tuple[EvidenceRequest, dict[str, Any]]] = []
        self._errors: list[tuple[EvidenceRequest, str]] = []

    def requests(self, task: SandboxTask | WorktreeTask) -> tuple[EvidenceRequest, ...]:
        candidates = [*_PATH_RE.findall(str(task.goal or ""))]
        candidates.extend(
            value
            for value in task.test_args
            if isinstance(value, str) and value and not value.startswith("-")
        )
        paths: list[str] = []
        for candidate in candidates:
            normalized = _normalize_relative_text_path(candidate)
            if normalized and normalized not in paths:
                paths.append(normalized)
            if len(paths) >= self.max_files:
                break
        return tuple(
            EvidenceRequest(path=path, max_bytes=self.max_bytes_per_file)
            for path in paths
        )

    def observe(
        self,
        request: EvidenceRequest,
        *,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        if result is not None and error:
            raise ValueError("result and error are mutually exclusive")
        if result is not None:
            self._results.append((request, dict(result)))
        else:
            self._errors.append((request, str(error or "read_failed")[:500]))

    def blocks(self) -> tuple[ContextBlock, ...]:
        blocks: list[ContextBlock] = []
        for request, result in self._results:
            relative_path = str(result.get("relative_path") or request.path)
            snapshot = {
                "path": relative_path,
                "sha256": str(result.get("sha256") or ""),
                "line_range": [result.get("start_line"), result.get("end_line")],
                "total_lines": result.get("total_lines"),
                "truncated": bool(result.get("truncated", False)),
                "text": str(result.get("text") or ""),
            }
            blocks.append(
                ContextBlock(
                    source="evidence_region",
                    title=f"Source snapshot: {relative_path}",
                    content=json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")),
                    framing="data",
                    metadata={
                        "kind": "source_snapshot",
                        "path": relative_path,
                        "sha": snapshot["sha256"],
                        "region": self.name,
                    },
                )
            )
        return tuple(blocks)

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": "explicit_task_paths_v1",
            "access_mode": self.access_mode,
            "uses_model": self.uses_model,
            "reads_succeeded": len(self._results),
            "reads_failed": len(self._errors),
            "paths_read": [request.path for request, _result in self._results],
            "failed_paths": [request.path for request, _error in self._errors],
            "blocks_published": len(self._results),
            "confidence": 1.0 if self._results else 0.0,
            "last_decision": "evidence_published" if self._results else "no_explicit_paths",
        }


def _normalize_relative_text_path(value: str) -> str:
    candidate = str(value or "").strip("`'\"()[]{}<>,;:").replace("\\", "/")
    while candidate.startswith("./"):
        candidate = candidate[2:]
    if not candidate or candidate.startswith("/"):
        return ""
    path = PurePosixPath(candidate)
    if path.is_absolute() or ".." in path.parts or path.suffix.casefold() not in _TEXT_SUFFIXES:
        return ""
    return path.as_posix()


__all__ = ["EvidenceRegion", "EvidenceRequest"]
