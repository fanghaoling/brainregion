"""Deterministic task-phase and difficulty telemetry for the sandbox runtime.

The controller is a control-plane observer. It does not select models, inject
prompts, or inspect private model reasoning. Later routing policies can consume
its structured output after the phase labels have been calibrated on real runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class CognitivePhase(str, Enum):
    UNDERSTAND = "understand"
    PLAN = "plan"
    EXECUTE = "execute"
    RECOVER = "recover"
    VERIFY = "verify"
    SYNTHESIZE = "synthesize"


class ComputeTier(str, Enum):
    DETERMINISTIC = "deterministic"
    ECONOMY = "economy"
    STANDARD = "standard"
    STRONG = "strong"


@dataclass(frozen=True)
class DifficultyVector:
    """Explainable difficulty dimensions where larger values mean harder."""

    scope: float
    ambiguity: float
    novelty: float
    risk: float
    irreversibility: float
    verification_gap: float
    stagnation: float = 0.0
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "scope",
            "ambiguity",
            "novelty",
            "risk",
            "irreversibility",
            "verification_gap",
            "stagnation",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"difficulty {name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"difficulty {name} must be between 0 and 1")

    @property
    def score(self) -> float:
        base = (
            self.scope * 0.22
            + self.ambiguity * 0.18
            + self.novelty * 0.16
            + self.risk * 0.18
            + self.irreversibility * 0.10
            + self.verification_gap * 0.16
        )
        return min(1.0, base * 0.75 + self.stagnation * 0.25)

    def with_stagnation(self, stagnation: float) -> "DifficultyVector":
        return replace(self, stagnation=max(0.0, min(1.0, float(stagnation))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": round(self.scope, 3),
            "ambiguity": round(self.ambiguity, 3),
            "novelty": round(self.novelty, 3),
            "risk": round(self.risk, 3),
            "irreversibility": round(self.irreversibility, 3),
            "verification_gap": round(self.verification_gap, 3),
            "stagnation": round(self.stagnation, 3),
            "score": round(self.score, 3),
            "reasons": list(self.reasons),
        }


def assess_task_difficulty(task: Any) -> DifficultyVector:
    """Build a metadata-only baseline; runtime signals are added separately."""

    is_worktree = bool(getattr(task, "repo_path", ""))
    files = getattr(task, "files", {}) or {}
    tests = getattr(task, "tests", {}) or {}
    test_args = getattr(task, "test_args", []) or []
    seed_memory = getattr(task, "seed_memory", []) or []
    artifact_count = len(files) + len(tests)

    if is_worktree:
        scope = 0.55
        risk = 0.45
        irreversibility = 0.30
    else:
        scope = min(0.75, 0.15 + artifact_count * 0.07)
        risk = 0.15
        irreversibility = 0.10

    has_objective_verification = bool(tests or test_args)
    ambiguity = 0.20 if has_objective_verification else 0.60
    verification_gap = 0.20 if has_objective_verification else 0.70
    novelty = 0.30 if seed_memory else 0.50

    reasons: list[str] = ["worktree" if is_worktree else "isolated_sandbox"]
    if artifact_count >= 6:
        reasons.append("multi_artifact")
    if not has_objective_verification:
        reasons.append("weak_verification_contract")
    if seed_memory:
        reasons.append("relevant_seed_memory_available")
    else:
        reasons.append("no_task_memory_signal")

    return DifficultyVector(
        scope=scope,
        ambiguity=ambiguity,
        novelty=novelty,
        risk=risk,
        irreversibility=irreversibility,
        verification_gap=verification_gap,
        reasons=tuple(reasons),
    )


def recommended_tier(phase: CognitivePhase, difficulty: DifficultyVector) -> ComputeTier:
    """Telemetry-only routing recommendation; no model is changed here."""

    if phase is CognitivePhase.VERIFY:
        if difficulty.verification_gap < 0.50 and difficulty.stagnation < 0.50:
            return ComputeTier.DETERMINISTIC
        return ComputeTier.STANDARD
    if phase is CognitivePhase.RECOVER:
        if difficulty.stagnation >= 0.50 or difficulty.score >= 0.45:
            return ComputeTier.STRONG
        return ComputeTier.STANDARD
    if phase is CognitivePhase.PLAN:
        if difficulty.score >= 0.65 or difficulty.risk >= 0.75:
            return ComputeTier.STRONG
        return ComputeTier.STANDARD
    if phase is CognitivePhase.EXECUTE:
        if difficulty.risk >= 0.75 or difficulty.irreversibility >= 0.75:
            return ComputeTier.STANDARD
        return ComputeTier.ECONOMY
    if phase is CognitivePhase.UNDERSTAND:
        if difficulty.ambiguity >= 0.75 or difficulty.risk >= 0.80:
            return ComputeTier.STRONG
        return ComputeTier.STANDARD
    return ComputeTier.STANDARD if difficulty.score >= 0.65 else ComputeTier.ECONOMY


@dataclass(frozen=True)
class PhaseTransition:
    index: int
    step: int
    previous: CognitivePhase
    current: CognitivePhase
    reason: str
    difficulty: DifficultyVector
    recommended_tier: ComputeTier

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "step": self.step,
            "from": self.previous.value,
            "to": self.current.value,
            "reason": self.reason,
            "difficulty": self.difficulty.to_dict(),
            "recommended_tier": self.recommended_tier.value,
        }


_INFORMATION_OPERATIONS = frozenset(
    {
        "list_allowed_roots",
        "inspect_file",
        "read_text",
        "search_text",
        "observe",
        "recall_map",
        "recall_topo",
        "recall_path",
        "region:evidence",
    }
)
_EXECUTION_OPERATIONS = frozenset(
    {"apply_text_patch", "act", "delegate_navigation", "region:option"}
)
_VERIFICATION_OPERATIONS = frozenset({"workspace_run_check", "region:verification"})


@dataclass
class PhaseController:
    """Rule-based phase observer driven only by task and tool facts."""

    base_difficulty: DifficultyVector
    phase: CognitivePhase = CognitivePhase.UNDERSTAND
    transitions: list[PhaseTransition] = field(default_factory=list)
    operations: int = 0
    steps_since_effect: int = 0
    repeated_targets: int = 0
    consecutive_errors: int = 0
    failed_verification: bool = False

    @classmethod
    def for_task(cls, task: Any) -> "PhaseController":
        return cls(base_difficulty=assess_task_difficulty(task))

    @property
    def stagnation(self) -> float:
        no_effect = max(0.0, min(1.0, (self.steps_since_effect - 2) / 4))
        repeated = min(1.0, self.repeated_targets / 2)
        errors = min(1.0, self.consecutive_errors / 2)
        failed_verification = 1.0 if self.failed_verification else 0.0
        return max(no_effect, repeated, errors, failed_verification)

    @property
    def difficulty(self) -> DifficultyVector:
        return self.base_difficulty.with_stagnation(self.stagnation)

    @property
    def tier(self) -> ComputeTier:
        return recommended_tier(self.phase, self.difficulty)

    def before_operation(self, *, step: int, operation: str) -> PhaseTransition | None:
        if operation == "plan":
            return self._transition(step, CognitivePhase.PLAN, "plan_requested")
        if operation in _VERIFICATION_OPERATIONS:
            return self._transition(step, CognitivePhase.VERIFY, "verification_requested")
        if operation in _EXECUTION_OPERATIONS:
            return self._transition(step, CognitivePhase.EXECUTE, "effectful_action_selected")
        if self.phase is CognitivePhase.PLAN and operation in _INFORMATION_OPERATIONS:
            return self._transition(step, CognitivePhase.EXECUTE, "plan_execution_started")
        return None

    def after_operation(
        self,
        *,
        step: int,
        operation: str,
        error: bool = False,
        workspace_effect: bool = False,
        verification_passed: bool | None = None,
        target_is_new: bool = True,
    ) -> PhaseTransition | None:
        self.operations += 1
        productive_information = (
            operation in _INFORMATION_OPERATIONS and target_is_new and not error
        )
        if productive_information:
            self.steps_since_effect = 0
        else:
            self.steps_since_effect += 1
        if operation in _INFORMATION_OPERATIONS:
            if target_is_new:
                self.repeated_targets = max(0, self.repeated_targets - 1)
            else:
                self.repeated_targets += 1

        if error:
            self.consecutive_errors += 1
            return self._transition(step, CognitivePhase.RECOVER, "operation_error")
        self.consecutive_errors = 0

        if verification_passed is False:
            self.failed_verification = True
            return self._transition(step, CognitivePhase.RECOVER, "verification_failed")
        if verification_passed is True:
            self.failed_verification = False
            self.steps_since_effect = 0
            self.repeated_targets = 0
            return self._transition(step, CognitivePhase.SYNTHESIZE, "verification_passed")
        if workspace_effect:
            self.failed_verification = False
            self.steps_since_effect = 0
            self.repeated_targets = 0
            return self._transition(
                step,
                CognitivePhase.VERIFY,
                "workspace_effect_requires_verification",
            )
        if self.phase is CognitivePhase.UNDERSTAND and operation in _INFORMATION_OPERATIONS:
            return self._transition(step, CognitivePhase.PLAN, "initial_evidence_collected")
        if self.phase is CognitivePhase.RECOVER and operation in _INFORMATION_OPERATIONS:
            return self._transition(step, CognitivePhase.PLAN, "recovery_evidence_collected")
        return None

    def observe_model_failure(self, *, step: int, reason: str) -> PhaseTransition | None:
        self.operations += 1
        self.steps_since_effect += 1
        self.consecutive_errors += 1
        return self._transition(step, CognitivePhase.RECOVER, reason)

    def observe_completion(self, *, step: int) -> PhaseTransition | None:
        return self._transition(step, CognitivePhase.SYNTHESIZE, "model_completed")

    def observe_final_verification(
        self,
        *,
        step: int,
        passed: bool,
    ) -> PhaseTransition | None:
        self.failed_verification = not passed
        if passed:
            self.steps_since_effect = 0
            return self._transition(step, CognitivePhase.SYNTHESIZE, "final_verification_passed")
        return self._transition(step, CognitivePhase.RECOVER, "final_verification_failed")

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "policy": "deterministic_phase_observer_v1",
            "phase": self.phase.value,
            "recommended_tier": self.tier.value,
            "difficulty": self.difficulty.to_dict(),
            "operations": self.operations,
            "steps_since_effect": self.steps_since_effect,
            "transition_count": len(self.transitions),
            "transitions": [transition.to_dict() for transition in self.transitions],
            "changes_model_routing": False,
            "contains_reasoning": False,
        }

    def _transition(
        self,
        step: int,
        phase: CognitivePhase,
        reason: str,
    ) -> PhaseTransition | None:
        if phase is self.phase:
            return None
        previous = self.phase
        self.phase = phase
        transition = PhaseTransition(
            index=len(self.transitions),
            step=max(-1, int(step)),
            previous=previous,
            current=phase,
            reason=str(reason or "phase_changed")[:100],
            difficulty=self.difficulty,
            recommended_tier=recommended_tier(phase, self.difficulty),
        )
        self.transitions.append(transition)
        return transition


__all__ = [
    "CognitivePhase",
    "ComputeTier",
    "DifficultyVector",
    "PhaseController",
    "PhaseTransition",
    "assess_task_difficulty",
    "recommended_tier",
]
