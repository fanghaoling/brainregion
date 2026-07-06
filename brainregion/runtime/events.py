"""Small runtime event log used by the debug dashboard.

This is intentionally dependency-free. It provides:
- an in-memory ring buffer for live subscribers,
- JSONL persistence for recent history,
- a blocking wait API that the SSE endpoint can use.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEFAULT_MAX_EVENTS = 2000


def runtime_events_path(root: str | os.PathLike[str] | None = None) -> Path:
    project_root = Path(root or os.environ.get("UNITY_PROJECT_ROOT", "."))
    return project_root / ".brain-region" / "runtime" / "events.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_sequence(previous: int = 0) -> int:
    return max(previous + 1, time.time_ns())


class RuntimeEventStore:
    def __init__(self, *, path: Path | None = None, max_events: int = _DEFAULT_MAX_EVENTS):
        self._path = path
        self._max_events = max_events
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._sequence = 0
        self._condition = threading.Condition()

    @property
    def path(self) -> Path:
        return self._path or runtime_events_path()

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        with self._condition:
            self._sequence = _event_sequence(self._sequence)
            event = {
                "id": fields.pop("id", uuid.uuid4().hex),
                "sequence": self._sequence,
                "timestamp": fields.pop("timestamp", _now_iso()),
                "type": event_type,
            }
            event.update({k: v for k, v in fields.items() if v is not None})
            self._events.append(event)
            try:
                self._append_jsonl(event)
            except Exception:
                pass
            self._condition.notify_all()
            return dict(event)

    def list(self, *, after_sequence: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        limit = max(1, limit)
        file_events = self._read_jsonl_events(after_sequence=after_sequence, limit=limit)
        with self._condition:
            memory_events = [dict(e) for e in self._events if int(e.get("sequence", 0)) > after_sequence]
        return self._merge_events(memory_events, file_events, limit=limit)

    def wait(self, *, after_sequence: int = 0, timeout: float = 15.0, limit: int = 200) -> list[dict[str, Any]]:
        deadline = time.monotonic() + max(0.1, timeout)
        while True:
            events = self.list(after_sequence=after_sequence, limit=limit)
            if events:
                return events
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            with self._condition:
                events = [dict(e) for e in self._events if int(e.get("sequence", 0)) > after_sequence]
                if events:
                    return events[-max(1, limit):]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._condition.wait(timeout=min(0.5, remaining))

    def clear_memory(self) -> None:
        with self._condition:
            self._events.clear()
            self._sequence = 0
            self._condition.notify_all()

    def _append_jsonl(self, event: dict[str, Any]) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str))
            f.write("\n")

    def _read_jsonl_events(self, *, after_sequence: int, limit: int) -> list[dict[str, Any]]:
        path = self.path
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if int(event.get("sequence", 0) or 0) > after_sequence:
                        events.append(event)
        except OSError:
            return []
        return events[-max(1, limit):]

    @staticmethod
    def _merge_events(
        memory_events: list[dict[str, Any]],
        file_events: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for event in [*file_events, *memory_events]:
            key = str(event.get("id") or event.get("sequence"))
            by_id[key] = dict(event)
        return sorted(by_id.values(), key=lambda e: int(e.get("sequence", 0) or 0))[-max(1, limit):]


_DEFAULT_STORE = RuntimeEventStore()


def emit_event(event_type: str, **fields: Any) -> dict[str, Any]:
    return _DEFAULT_STORE.emit(event_type, **fields)


def list_events(*, after_sequence: int = 0, limit: int = 200) -> list[dict[str, Any]]:
    return _DEFAULT_STORE.list(after_sequence=after_sequence, limit=limit)


def wait_events(*, after_sequence: int = 0, timeout: float = 15.0, limit: int = 200) -> list[dict[str, Any]]:
    return _DEFAULT_STORE.wait(after_sequence=after_sequence, timeout=timeout, limit=limit)
