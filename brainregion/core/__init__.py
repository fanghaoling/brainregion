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
from .assignment_expert import (
    AssignmentExpertResult,
    AssignmentExpertRunner,
    AssignmentExpertState,
)
from .context_export import (
    ContextExportDecision,
    bypass_context_export,
    context_export_mode,
    endpoint_context_trust,
    evaluate_context_export,
)
from .context_loader import (
    ActivatedContext,
    ContextLoadRecord,
    estimate_context_tokens,
    load_activation_context,
)
from .context_pressure import (
    ContextPressureBand,
    ContextPressureObserver,
    ContextPressureSample,
    disabled_context_pressure_metrics,
)
from .cognitive_workspace import CognitiveWorkspace, WorkspaceDelivery, WorkspaceView
from .region_reporting import (
    EscalationPolicy,
    RegionContextReceipt,
    RegionCoordinationBoard,
    RegionReport,
)
from .region_expert import RegionExpertEngine, RegionExpertResult
from .document import DocumentType, ReviewDocument
from .engine import ReviewEngine
from .task_coordination import (
    EvidenceWakeRequest,
    ExpertAssignment,
    MemoryRequest,
    TaskCoordinationBoard,
    TaskSpec,
)
from .pipeline import Pipeline, PipelineContext, Stage
from .planner import PlanReport, PlanRequest, PlannerEngine
from .regions import RegionDefinition, route_regions
from .report import CanonicalFinding, Finding, ReviewReport

__all__ = [
    "ActivationContract",
    "ActivationPlan",
    "ActivationSignal",
    "AssignmentExpertResult",
    "AssignmentExpertRunner",
    "AssignmentExpertState",
    "ContextRequest",
    "ContextExportDecision",
    "bypass_context_export",
    "context_export_mode",
    "endpoint_context_trust",
    "evaluate_context_export",
    "RegionWakeDecision",
    "evaluate_activation",
    "plan_activation",
    "ActivatedContext",
    "ContextLoadRecord",
    "ContextPressureBand",
    "ContextPressureObserver",
    "ContextPressureSample",
    "disabled_context_pressure_metrics",
    "CognitiveWorkspace",
    "WorkspaceDelivery",
    "EscalationPolicy",
    "RegionContextReceipt",
    "RegionCoordinationBoard",
    "RegionReport",
    "RegionExpertEngine",
    "RegionExpertResult",
    "EvidenceWakeRequest",
    "ExpertAssignment",
    "MemoryRequest",
    "TaskCoordinationBoard",
    "TaskSpec",
    "WorkspaceView",
    "estimate_context_tokens",
    "load_activation_context",
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
