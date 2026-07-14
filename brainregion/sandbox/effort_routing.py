"""Same-model effort recommendations derived from deterministic task phases."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .phase_control import CognitivePhase, ComputeTier, DifficultyVector, recommended_tier


@dataclass(frozen=True)
class EffortControls:
    thinking: bool
    effort: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"thinking": self.thinking, "effort": self.effort}


def controls_for_tier(tier: ComputeTier) -> EffortControls:
    """Map a compute recommendation to provider-neutral same-model controls."""

    if tier in {ComputeTier.DETERMINISTIC, ComputeTier.ECONOMY}:
        return EffortControls(thinking=False, effort=None)
    if tier is ComputeTier.STANDARD:
        return EffortControls(thinking=True, effort="medium")
    return EffortControls(thinking=True, effort="high")


@dataclass(frozen=True)
class EffortRoutingDecision:
    index: int
    step: int
    phase: CognitivePhase
    difficulty_score: float
    recommended_tier: ComputeTier
    recommended: EffortControls
    mode: Literal["shadow", "active"]
    configured_thinking: bool | None
    configured_effort: str | None
    actual_thinking: bool | None
    actual_effort: str | None
    would_change: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        effective = {
            "thinking": self.actual_thinking,
            "effort": self.actual_effort,
        }
        return {
            "index": self.index,
            "step": self.step,
            "phase": self.phase.value,
            "difficulty_score": round(self.difficulty_score, 3),
            "recommended_tier": self.recommended_tier.value,
            "recommended": self.recommended.to_dict(),
            "mode": self.mode,
            "configured": {
                "thinking": self.configured_thinking,
                "effort": self.configured_effort,
            },
            # ``actual`` is retained for artifact compatibility. Both fields
            # describe the backend request, not provider-side execution proof.
            "effective": effective,
            "actual": effective,
            "control_scope": "backend_request",
            "would_change": self.would_change,
            "recommendation_applied": self.mode == "active",
            "controls_changed": self.mode == "active" and self.would_change,
            "reason": self.reason,
        }


@dataclass
class PhaseEffortShadow:
    """Observe or explicitly apply phase effort decisions without switching models."""

    mode: Literal["shadow", "active"] = "shadow"
    decisions: list[EffortRoutingDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in {"shadow", "active"}:
            raise ValueError("effort routing mode must be 'shadow' or 'active'")

    def observe(
        self,
        *,
        step: int,
        phase: CognitivePhase,
        difficulty: DifficultyVector,
        actual_thinking: bool | None,
        actual_effort: str | None,
    ) -> EffortRoutingDecision:
        tier = recommended_tier(phase, difficulty)
        controls = controls_for_tier(tier)
        active = self.mode == "active"
        decision = EffortRoutingDecision(
            index=len(self.decisions),
            step=max(0, int(step)),
            phase=phase,
            difficulty_score=difficulty.score,
            recommended_tier=tier,
            recommended=controls,
            mode=self.mode,
            configured_thinking=actual_thinking,
            configured_effort=actual_effort,
            actual_thinking=controls.thinking if active else actual_thinking,
            actual_effort=controls.effort if active else actual_effort,
            would_change=(
                actual_thinking is not controls.thinking
                or actual_effort != controls.effort
            ),
            reason=f"phase_{phase.value}_tier_{tier.value}",
        )
        self.decisions.append(decision)
        return decision

    def snapshot(self) -> dict[str, Any]:
        by_phase: dict[str, dict[str, Any]] = {}
        for decision in self.decisions:
            phase = decision.phase.value
            row = by_phase.setdefault(
                phase,
                {
                    "calls": 0,
                    "would_change_calls": 0,
                    "recommended_tiers": {},
                },
            )
            row["calls"] += 1
            row["would_change_calls"] += int(decision.would_change)
            tier = decision.recommended_tier.value
            row["recommended_tiers"][tier] = row["recommended_tiers"].get(tier, 0) + 1

        return {
            "enabled": True,
            "mode": self.mode,
            "policy": f"same_model_phase_effort_{self.mode}_v1",
            "decision_count": len(self.decisions),
            "would_change_calls": sum(int(item.would_change) for item in self.decisions),
            "applied_change_calls": sum(
                int(self.mode == "active" and item.would_change) for item in self.decisions
            ),
            "agreement_calls": sum(int(not item.would_change) for item in self.decisions),
            "recommended_thinking_calls": sum(
                int(item.recommended.thinking) for item in self.decisions
            ),
            "actual_thinking_calls": sum(
                int(item.actual_thinking is True) for item in self.decisions
            ),
            "by_phase": by_phase,
            "decisions": [item.to_dict() for item in self.decisions],
            "changes_model_routing": False,
            "changes_inference_controls": self.mode == "active",
            "control_scope": "backend_request",
            "provider_execution_telemetry": "not_collected",
            "contains_reasoning": False,
            "contains_content": False,
        }


def disabled_effort_shadow_metrics() -> dict[str, Any]:
    return {
        "enabled": False,
        "mode": "off",
        "policy": "same_model_phase_effort_shadow_v1",
        "decision_count": 0,
        "would_change_calls": 0,
        "applied_change_calls": 0,
        "agreement_calls": 0,
        "recommended_thinking_calls": 0,
        "actual_thinking_calls": 0,
        "by_phase": {},
        "decisions": [],
        "changes_model_routing": False,
        "changes_inference_controls": False,
        "control_scope": "backend_request",
        "provider_execution_telemetry": "not_collected",
        "contains_reasoning": False,
        "contains_content": False,
    }


__all__ = [
    "EffortControls",
    "EffortRoutingDecision",
    "PhaseEffortShadow",
    "controls_for_tier",
    "disabled_effort_shadow_metrics",
]
