"""框架核心：项目无关的 Pipeline/Stage/Engine/Document/Report 抽象。"""
from __future__ import annotations

from .activation import (
    ActivationContract,
    ActivationPlan,
    ActivationSignal,
    ContextRequest,
    RegionWakeDecision,
    evaluate_activation,
    plan_activation,
)
from .document import DocumentType, ReviewDocument
from .engine import ReviewEngine
from .pipeline import Pipeline, PipelineContext, Stage
from .planner import PlanReport, PlanRequest, PlannerEngine
from .regions import RegionDefinition, route_regions
from .report import CanonicalFinding, Finding, ReviewReport

__all__ = [
    "ActivationContract",
    "ActivationPlan",
    "ActivationSignal",
    "ContextRequest",
    "RegionWakeDecision",
    "evaluate_activation",
    "plan_activation",
    "DocumentType",
    "ReviewDocument",
    "ReviewEngine",
    "PlanRequest",
    "PlanReport",
    "PlannerEngine",
    "RegionDefinition",
    "route_regions",
    "Pipeline",
    "PipelineContext",
    "Stage",
    "Finding",
    "CanonicalFinding",
    "ReviewReport",
]
