"""Deterministic contracts between main-brain intent and region execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .task_coordination import TaskSpec

IntentRisk = Literal["low", "medium", "high"]
IntentAutonomy = Literal["read_only", "workspace_write", "requires_approval"]

_RISKS = frozenset({"low", "medium", "high"})
_AUTONOMY = frozenset({"read_only", "workspace_write", "requires_approval"})
_READ_ONLY_ACTIONS = frozenset({"inspect_file", "read_text", "search_text"})


def _required_text(value: Any, name: str, *, max_length: int = 4000) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    if len(text) > max_length:
        raise ValueError(f"{name} cannot exceed {max_length} characters")
    return text


def _identifier(value: Any, name: str) -> str:
    text = _required_text(value, name, max_length=200)
    if any(char.isspace() for char in text):
        raise ValueError(f"{name} cannot contain whitespace")
    return text


def _string_tuple(
    value: Any,
    name: str,
    *,
    max_items: int = 64,
    max_item_length: int = 2000,
) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{name} must be an array")
    output: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if len(text) > max_item_length:
            raise ValueError(f"{name} entries cannot exceed {max_item_length} characters")
        if text and text not in output:
            output.append(text)
    if len(output) > max_items:
        raise ValueError(f"{name} cannot contain more than {max_items} items")
    return tuple(output)


@dataclass(frozen=True)
class CognitiveIntent:
    """A bounded statement of what the main brain wants, without tool calls."""

    intent_id: str
    objective: str
    required_capabilities: tuple[str, ...]
    success_criteria: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    resource_hints: tuple[str, ...] = ()
    search_queries: tuple[str, ...] = ()
    risk: IntentRisk = "low"
    autonomy: IntentAutonomy = "read_only"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CognitiveIntent":
        if not isinstance(data, dict):
            raise ValueError("intent must be an object")
        unknown = set(data) - {
            "intent_id",
            "objective",
            "required_capabilities",
            "success_criteria",
            "constraints",
            "resource_hints",
            "search_queries",
            "risk",
            "autonomy",
        }
        if unknown:
            raise ValueError(f"intent unknown field(s): {sorted(unknown)}")
        capabilities = tuple(
            capability.casefold()
            for capability in _string_tuple(
                data.get("required_capabilities"),
                "required_capabilities",
                max_items=16,
                max_item_length=200,
            )
        )
        if not capabilities:
            raise ValueError("required_capabilities cannot be empty")
        risk = str(data.get("risk") or "low").strip().casefold()
        if risk not in _RISKS:
            raise ValueError(f"risk must be one of {sorted(_RISKS)}")
        autonomy = str(data.get("autonomy") or "read_only").strip().casefold()
        if autonomy not in _AUTONOMY:
            raise ValueError(f"autonomy must be one of {sorted(_AUTONOMY)}")
        return cls(
            intent_id=_identifier(data.get("intent_id"), "intent_id"),
            objective=_required_text(data.get("objective"), "objective"),
            required_capabilities=capabilities,
            success_criteria=_string_tuple(data.get("success_criteria"), "success_criteria"),
            constraints=_string_tuple(data.get("constraints"), "constraints"),
            resource_hints=_string_tuple(
                data.get("resource_hints"),
                "resource_hints",
                max_item_length=1000,
            ),
            search_queries=_string_tuple(
                data.get("search_queries"),
                "search_queries",
                max_items=16,
                max_item_length=500,
            ),
            risk=risk,  # type: ignore[arg-type]
            autonomy=autonomy,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "objective": self.objective,
            "required_capabilities": list(self.required_capabilities),
            "success_criteria": list(self.success_criteria),
            "constraints": list(self.constraints),
            "resource_hints": list(self.resource_hints),
            "search_queries": list(self.search_queries),
            "risk": self.risk,
            "autonomy": self.autonomy,
        }


@dataclass(frozen=True)
class CapabilityRoute:
    capability: str
    region: str
    allowed_actions: tuple[str, ...]
    output_contract: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability", _identifier(self.capability, "capability").casefold())
        object.__setattr__(self, "region", _identifier(self.region, "region").casefold())
        actions = tuple(
            action.casefold()
            for action in _string_tuple(
                self.allowed_actions,
                "allowed_actions",
                max_items=32,
                max_item_length=100,
            )
        )
        if not actions:
            raise ValueError("allowed_actions cannot be empty")
        object.__setattr__(self, "allowed_actions", actions)
        object.__setattr__(
            self,
            "output_contract",
            _identifier(self.output_contract, "output_contract").casefold(),
        )


@dataclass(frozen=True)
class IntentAssignment:
    assignment_id: str
    task_id: str
    capability: str
    region: str
    objective: str
    success_criteria: tuple[str, ...]
    constraints: tuple[str, ...]
    resource_hints: tuple[str, ...]
    search_queries: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    output_contract: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "task_id": self.task_id,
            "capability": self.capability,
            "region": self.region,
            "objective": self.objective,
            "success_criteria": list(self.success_criteria),
            "constraints": list(self.constraints),
            "resource_hints": list(self.resource_hints),
            "search_queries": list(self.search_queries),
            "allowed_actions": list(self.allowed_actions),
            "output_contract": self.output_contract,
        }


@dataclass(frozen=True)
class CompiledIntent:
    intent: CognitiveIntent
    task: TaskSpec
    assignments: tuple[IntentAssignment, ...]

    def assignment_for(self, capability: str) -> IntentAssignment | None:
        normalized = str(capability or "").strip().casefold()
        return next(
            (assignment for assignment in self.assignments if assignment.capability == normalized),
            None,
        )

    def action_owners(self) -> dict[str, str]:
        owners: dict[str, str] = {}
        for assignment in self.assignments:
            for action in assignment.allowed_actions:
                owner = owners.get(action)
                if owner is not None and owner != assignment.region:
                    raise ValueError(f"action {action!r} has conflicting owners: {owner!r}, {assignment.region!r}")
                owners[action] = assignment.region
        return owners

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.to_dict(),
            "task": self.task.to_dict(),
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "action_owners": self.action_owners(),
            "compiler": "deterministic_intent_compiler_v1",
            "models_called": False,
            "contains_context_content": False,
            "contains_reasoning": False,
        }


DEFAULT_CAPABILITY_ROUTES = (
    CapabilityRoute(
        capability="code_evidence",
        region="evidence",
        allowed_actions=("read_text", "search_text"),
        output_contract="evidence_packet",
    ),
)


class IntentCompiler:
    """Compile semantic capabilities into deterministic region ownership."""

    def __init__(self, routes: tuple[CapabilityRoute, ...] = DEFAULT_CAPABILITY_ROUTES) -> None:
        if not routes:
            raise ValueError("routes cannot be empty")
        self._routes: dict[str, CapabilityRoute] = {}
        for route in routes:
            if not isinstance(route, CapabilityRoute):
                raise ValueError("routes must contain CapabilityRoute values")
            if route.capability in self._routes:
                raise ValueError(f"duplicate capability route: {route.capability}")
            self._routes[route.capability] = route

    def compile(self, intent: CognitiveIntent | dict[str, Any]) -> CompiledIntent:
        if isinstance(intent, dict):
            intent = CognitiveIntent.from_dict(intent)
        if not isinstance(intent, CognitiveIntent):
            raise ValueError("intent must be CognitiveIntent or an object")
        unknown = [capability for capability in intent.required_capabilities if capability not in self._routes]
        if unknown:
            raise ValueError(f"unroutable capability(s): {unknown}")

        assignments: list[IntentAssignment] = []
        for capability in intent.required_capabilities:
            route = self._routes[capability]
            if intent.autonomy == "read_only":
                disallowed = set(route.allowed_actions) - _READ_ONLY_ACTIONS
                if disallowed:
                    raise ValueError(f"read_only intent cannot own write action(s): {sorted(disallowed)}")
            assignment_id = f"{intent.intent_id}:{capability}"
            if len(assignment_id) > 200:
                raise ValueError("compiled assignment_id cannot exceed 200 characters")
            assignments.append(
                IntentAssignment(
                    assignment_id=assignment_id,
                    task_id=intent.intent_id,
                    capability=capability,
                    region=route.region,
                    objective=intent.objective,
                    success_criteria=intent.success_criteria,
                    constraints=intent.constraints,
                    resource_hints=intent.resource_hints,
                    search_queries=intent.search_queries,
                    allowed_actions=route.allowed_actions,
                    output_contract=route.output_contract,
                )
            )

        task = TaskSpec.from_dict(
            {
                "task_id": intent.intent_id,
                "goal": intent.objective,
                "success_criteria": list(intent.success_criteria),
                "constraints": list(intent.constraints),
            }
        )
        compiled = CompiledIntent(intent=intent, task=task, assignments=tuple(assignments))
        compiled.action_owners()
        return compiled


__all__ = [
    "CapabilityRoute",
    "CognitiveIntent",
    "CompiledIntent",
    "DEFAULT_CAPABILITY_ROUTES",
    "IntentAssignment",
    "IntentAutonomy",
    "IntentCompiler",
    "IntentRisk",
]
