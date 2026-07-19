"""Grounded, model-free evidence collection for code sandbox tasks."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from brainregion.core.context import ContextBlock
from brainregion.core.intent import IntentAssignment

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
    action: Literal["read_text", "search_text"]
    path: str = ""
    query: str = ""
    max_bytes: int = 12000
    max_results: int = 20

    def to_dict(self) -> dict[str, Any]:
        if self.action == "read_text":
            return {"action": self.action, "path": self.path, "max_bytes": self.max_bytes}
        return {
            "action": self.action,
            "query": self.query,
            "max_results": self.max_results,
        }


class EvidenceRegion:
    """Select explicit task paths and turn host reads into grounded evidence blocks."""

    name = "evidence"
    access_mode = "grounded"
    uses_model = False

    def __init__(
        self,
        *,
        max_files: int = 4,
        max_bytes_per_file: int = 12000,
        max_searches: int = 3,
        max_results_per_search: int = 20,
        max_follow_ups: int = 3,
    ) -> None:
        if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files <= 0:
            raise ValueError("max_files must be a positive integer")
        if (
            isinstance(max_bytes_per_file, bool)
            or not isinstance(max_bytes_per_file, int)
            or max_bytes_per_file <= 0
        ):
            raise ValueError("max_bytes_per_file must be a positive integer")
        if isinstance(max_searches, bool) or not isinstance(max_searches, int) or max_searches <= 0:
            raise ValueError("max_searches must be a positive integer")
        if (
            isinstance(max_results_per_search, bool)
            or not isinstance(max_results_per_search, int)
            or max_results_per_search <= 0
        ):
            raise ValueError("max_results_per_search must be a positive integer")
        if (
            isinstance(max_follow_ups, bool)
            or not isinstance(max_follow_ups, int)
            or max_follow_ups <= 0
        ):
            raise ValueError("max_follow_ups must be a positive integer")
        self.max_files = max_files
        self.max_bytes_per_file = max_bytes_per_file
        self.max_searches = max_searches
        self.max_results_per_search = max_results_per_search
        self.max_follow_ups = max_follow_ups
        self._results: list[tuple[EvidenceRequest, dict[str, Any]]] = []
        self._errors: list[tuple[EvidenceRequest, str]] = []
        self._follow_up_requests: list[EvidenceRequest] = []

    def _allowed_actions(
        self,
        task: SandboxTask | WorktreeTask,
        assignment: IntentAssignment | None,
    ) -> frozenset[str]:
        if assignment is None:
            return frozenset({"read_text"})
        if assignment.task_id != task.id:
            raise ValueError("evidence assignment task_id must match sandbox task")
        if assignment.capability != "code_evidence" or assignment.region != self.name:
            raise ValueError("evidence region requires a code_evidence assignment")
        allowed_actions = frozenset(assignment.allowed_actions)
        unsupported = allowed_actions - {"read_text", "search_text"}
        if unsupported:
            raise ValueError(f"unsupported evidence action(s): {sorted(unsupported)}")
        return allowed_actions

    def requests(
        self,
        task: SandboxTask | WorktreeTask,
        assignment: IntentAssignment | None = None,
    ) -> tuple[EvidenceRequest, ...]:
        allowed_actions = self._allowed_actions(task, assignment)
        candidates: list[str] = []
        search_queries: tuple[str, ...] = ()
        if assignment is not None:
            candidates.extend(assignment.resource_hints)
            search_queries = assignment.search_queries[: self.max_searches]
        candidates.extend(_PATH_RE.findall(str(task.goal or "")))
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
        requests: list[EvidenceRequest] = []
        if "read_text" in allowed_actions:
            requests.extend(
                EvidenceRequest(
                    action="read_text",
                    path=path,
                    max_bytes=self.max_bytes_per_file,
                )
                for path in paths
            )
        if "search_text" in allowed_actions:
            requests.extend(
                EvidenceRequest(
                    action="search_text",
                    query=query[:500],
                    max_results=self.max_results_per_search,
                )
                for query in search_queries
                if query.strip()
            )
        return tuple(requests)

    def follow_up_request(
        self,
        task: SandboxTask | WorktreeTask,
        assignment: IntentAssignment,
        payload: dict[str, Any],
    ) -> EvidenceRequest:
        """Validate one main-brain evidence request without transferring read/search ownership."""
        if not isinstance(payload, dict):
            raise ValueError("evidence request args must be an object")
        allowed_actions = self._allowed_actions(task, assignment)
        action = str(payload.get("action") or "").strip().casefold()
        if action not in {"read_text", "search_text"}:
            raise ValueError("evidence request action must be read_text or search_text")
        if action not in allowed_actions:
            raise ValueError(f"evidence action {action!r} is not owned by the assignment")
        allowed_fields = {"action", "path"} if action == "read_text" else {"action", "query"}
        unknown = set(payload) - allowed_fields
        if unknown:
            raise ValueError(f"evidence request unknown field(s): {sorted(unknown)}")
        if len(self._follow_up_requests) >= self.max_follow_ups:
            raise ValueError("evidence follow-up limit reached")

        observed = [request for request, _ in (*self._results, *self._errors)]
        considered = observed + [
            request for request in self._follow_up_requests if request not in observed
        ]
        if action == "read_text":
            path = _normalize_relative_text_path(str(payload.get("path") or ""))
            if not path:
                raise ValueError("evidence read path must be a relative text file")
            if sum(request.action == "read_text" for request in considered) >= self.max_files:
                raise ValueError("evidence file limit reached")
            request = EvidenceRequest(
                action="read_text",
                path=path,
                max_bytes=self.max_bytes_per_file,
            )
            if any(item.action == "read_text" and item.path == path for item in considered):
                raise ValueError("evidence path was already requested")
        else:
            query = str(payload.get("query") or "").strip()
            if not query or len(query) > 500:
                raise ValueError("evidence search query must contain 1..500 characters")
            if sum(request.action == "search_text" for request in considered) >= self.max_searches:
                raise ValueError("evidence search limit reached")
            request = EvidenceRequest(
                action="search_text",
                query=query,
                max_results=self.max_results_per_search,
            )
            if any(item.action == "search_text" and item.query == query for item in considered):
                raise ValueError("evidence query was already requested")

        self._follow_up_requests.append(request)
        return request

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
            self._errors.append((request, str(error or f"{request.action}_failed")[:500]))

    def blocks(self) -> tuple[ContextBlock, ...]:
        blocks: list[ContextBlock] = []
        for request, result in self._results:
            if request.action == "search_text":
                matches = [
                    {
                        "path": str(match.get("relative_path") or ""),
                        "line": match.get("line"),
                        "text": str(match.get("text") or ""),
                        "context": list(match.get("context") or ()),
                    }
                    for match in result.get("matches", ())
                    if isinstance(match, dict)
                ]
                query_sha = hashlib.sha256(request.query.encode("utf-8")).hexdigest()
                raw_matched_files = result.get("matched_files")
                matched_file_count = (
                    int(raw_matched_files)
                    if isinstance(raw_matched_files, int)
                    else len(raw_matched_files or ())
                )
                snapshot = {
                    "query": request.query,
                    "matches": matches,
                    "matched_file_count": matched_file_count,
                    "matched_paths": list(
                        dict.fromkeys(match["path"] for match in matches if match["path"])
                    ),
                    "scanned_files": int(result.get("scanned_files") or 0),
                    "truncated": bool(result.get("truncated", False)),
                }
                blocks.append(
                    ContextBlock(
                        source="evidence_region",
                        title=f"Search evidence: {request.query[:80]}",
                        content=json.dumps(
                            snapshot,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        framing="data",
                        metadata={
                            "kind": "search_results",
                            "query_sha256": query_sha,
                            "region": self.name,
                        },
                    )
                )
                continue
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
        successful_reads = [
            request for request, _result in self._results if request.action == "read_text"
        ]
        successful_searches = [
            request for request, _result in self._results if request.action == "search_text"
        ]
        failed_reads = [
            request for request, _error in self._errors if request.action == "read_text"
        ]
        failed_searches = [
            request for request, _error in self._errors if request.action == "search_text"
        ]
        return {
            "policy": "intent_scoped_evidence_v2",
            "access_mode": self.access_mode,
            "uses_model": self.uses_model,
            "reads_succeeded": len(successful_reads),
            "reads_failed": len(failed_reads),
            "searches_succeeded": len(successful_searches),
            "searches_failed": len(failed_searches),
            "paths_read": [request.path for request in successful_reads],
            "queries_searched": [request.query for request in successful_searches],
            "failed_paths": [request.path for request in failed_reads],
            "failed_queries": [request.query for request in failed_searches],
            "blocks_published": len(self._results),
            "follow_up_requests": len(self._follow_up_requests),
            "follow_up_remaining": self.max_follow_ups - len(self._follow_up_requests),
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
