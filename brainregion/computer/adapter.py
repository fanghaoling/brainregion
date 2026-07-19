"""Host adapter boundary for computer interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .contracts import ActionIntent, SceneObservation


@dataclass(frozen=True)
class AdapterExecution:
    succeeded: bool
    reason: str = ""


@runtime_checkable
class ComputerUseAdapter(Protocol):
    """Mechanism-only adapter; policy and completion belong to the runtime."""

    app_id: str

    def observe(self, *, session_id: str) -> SceneObservation: ...

    def execute(self, intent: ActionIntent) -> AdapterExecution: ...
