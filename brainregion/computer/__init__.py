"""Controlled computer-use contracts, locator, perception layer and host runtime."""

from .adapter import AdapterExecution, ComputerUseAdapter
from .contracts import (
    ActionIntent,
    ActionReceipt,
    FrameRef,
    Panel,
    SceneObservation,
    UIElement,
)
from .locator import (
    ElementDescriptor,
    Locator,
    PanelAnchor,
    ResolvedTarget,
    ResolutionStep,
    WithinPanel,
)
from .mock import MockComputerUseAdapter
from .perception import PerceptionRegion, ResolutionResult
from .session import ComputerUseSession
from .targeting import TargetingController
from .unity_mock import UnityEditorMockAdapter

__all__ = [
    "ActionIntent",
    "ActionReceipt",
    "AdapterExecution",
    "ComputerUseAdapter",
    "ComputerUseSession",
    "ElementDescriptor",
    "FrameRef",
    "Locator",
    "MockComputerUseAdapter",
    "Panel",
    "PanelAnchor",
    "PerceptionRegion",
    "ResolvedTarget",
    "ResolutionResult",
    "ResolutionStep",
    "SceneObservation",
    "TargetingController",
    "UIElement",
    "UnityEditorMockAdapter",
    "WithinPanel",
]
