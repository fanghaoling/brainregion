"""Controlled computer-use contracts and host runtime."""

from .adapter import AdapterExecution, ComputerUseAdapter
from .contracts import ActionIntent, ActionReceipt, FrameRef, SceneObservation, UIElement
from .mock import MockComputerUseAdapter
from .session import ComputerUseSession

__all__ = [
    "ActionIntent",
    "ActionReceipt",
    "AdapterExecution",
    "ComputerUseAdapter",
    "ComputerUseSession",
    "FrameRef",
    "MockComputerUseAdapter",
    "SceneObservation",
    "UIElement",
]
