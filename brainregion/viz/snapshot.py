"""BrainSnapshot:脑状态的 render-ready、可序列化投影(可视化 Phase 1)。

把 Inspector `inspect()` 的 dict 投影成一组小 dataclass,供 Renderer 消费。
**单一真相源**:只读 inspect() 的输出,不重读 store、不二次聚合(同 Inspector「读已存 summary
不重算」教训)。snapshot 是**可长期落盘 artifact**——`to_dict()`/`from_dict()` 集中序列化,
`schema_version` 让旧 snapshot.json 在字段演进后仍可被 renderer/version 分支处理(对齐
`eval.store.SUMMARY_SCHEMA_VERSION` 纪律)。

region-centric:hero 是 `RegionSnapshot` 列表(每 region 的 total/recallable/woke)。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from brainregion import __version__
from brainregion.inspector import inspect as _inspect

SNAPSHOT_SCHEMA_VERSION = 3

# KPI 染色状态(HTML renderer 据此上色)
_OK = "ok"
_WARN = "warn"
_BAD = "bad"
_NEUTRAL = "neutral"

_SUMMARY_MAX = 120  # manifest summary 截断上限(snapshot 作 artifact 控体积;全量正文在 store)


@dataclass(frozen=True)
class ManifestExperience:
    """一条记忆的 diff-ready 清单项(Phase 2 Brain Diff 用)。

    manifest = 全量记忆清单(无 details/source/age,那些留 preview);要的是可比的
    id + 语义字段(region/status/summary/triggers)。created_at 仅展示/排序,**不参与 diff**。
    构造归一(GPT round2 ②③):triggers → sorted(set)(集合语义,顺序无关)、summary → ≤120 字。
    """

    id: str
    region: str
    status: str
    summary: str
    triggers: tuple[str, ...] = ()
    created_at: str = ""

    def __post_init__(self):
        # frozen → 用 object.__setattr__ 归一(list/tuple 入参都变 sorted tuple)
        object.__setattr__(self, "triggers", tuple(sorted(set(self.triggers or ()))))
        object.__setattr__(self, "summary", (self.summary or "")[:_SUMMARY_MAX])

    def semantic_equals(self, other: "ManifestExperience") -> bool:
        """diff 语义相等:只比 region/status/summary/triggers(均已归一,直接比)。

        id(配对时同)、created_at、未来 metadata(last_seen/confidence)忽略——
        防「加可变 metadata → 全 changed」(GPT round2 ①)。"""
        return (self.region == other.region and self.status == other.status
                and self.summary == other.summary and self.triggers == other.triggers)

    def to_dict(self) -> dict:
        return {"id": self.id, "region": self.region, "status": self.status,
                "summary": self.summary, "triggers": list(self.triggers), "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestExperience":
        return cls(id=d.get("id", ""), region=d.get("region", ""), status=d.get("status", ""),
                   summary=d.get("summary", ""), triggers=d.get("triggers", ()),
                   created_at=d.get("created_at", ""))


@dataclass(frozen=True)
class Kpi:
    """一个 headline 指标(label/value/status/hint)。status 供 renderer 染色。"""

    label: str
    value: str
    status: str = _NEUTRAL
    hint: str = ""

    def to_dict(self) -> dict:
        return {"label": self.label, "value": self.value, "status": self.status, "hint": self.hint}

    @classmethod
    def from_dict(cls, d: dict) -> "Kpi":
        return cls(
            label=d.get("label", ""),
            value=d.get("value", ""),
            status=d.get("status", _NEUTRAL),
            hint=d.get("hint", ""),
        )


@dataclass(frozen=True)
class RegionSnapshot:
    """一个脑区的状态:总记忆数 / 可召回数 / 是否在当前查询被唤醒。

    woke ∈ {"yes","no","unknown"}(unknown=无激活查询);inactive = total-recallable(渲染派生)。
    """

    region: str
    total: int
    recallable: int
    woke: str = "unknown"
    score: int = 0
    confidence: float = 0.0
    phase: str = "unknown"
    suggested_actions: int = 0
    action_tools: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"region": self.region, "total": self.total,
                "recallable": self.recallable, "woke": self.woke,
                "score": self.score, "confidence": self.confidence,
                "phase": self.phase, "suggested_actions": self.suggested_actions,
                "action_tools": list(self.action_tools)}

    @classmethod
    def from_dict(cls, d: dict) -> "RegionSnapshot":
        return cls(region=d.get("region", ""), total=int(d.get("total", 0)),
                   recallable=int(d.get("recallable", 0)), woke=d.get("woke", "unknown"),
                   score=int(d.get("score", 0) or 0),
                   confidence=float(d.get("confidence", 0.0) or 0.0),
                   phase=d.get("phase", "unknown"),
                   suggested_actions=int(d.get("suggested_actions", 0) or 0),
                   action_tools=tuple(d.get("action_tools") or ()))


@dataclass(frozen=True)
class BrainSnapshot:
    """脑整体状态快照。activation/memory/runs/calibration 是 Inspector 输出的不透明 dict(原样塞回)。"""

    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    generated_at: str = ""
    brainregion_version: str = ""
    has_query: bool = False
    query_label: str = ""                       # 截断 problem/goal,diff 页标签(GPT r1③)
    kpis: list[Kpi] = field(default_factory=list)
    regions: list[RegionSnapshot] = field(default_factory=list)
    manifest: list[ManifestExperience] = field(default_factory=list)  # 全量记忆清单(Phase 2 diff 用)
    activation: dict | None = None
    memory: dict = field(default_factory=dict)
    runs: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BrainSnapshot":
        ver = int(d.get("schema_version", 1))
        if ver > SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"snapshot schema_version {ver} > supported {SNAPSHOT_SCHEMA_VERSION}; "
                f"升级 brainregion 或用旧版本渲染"
            )
        # v1 重建(kpis/regions 递归;opaque dict 原样)。v2 增 query_label/manifest(v1 缺 → 默认空,不报错)。
        return cls(
            schema_version=ver,
            generated_at=d.get("generated_at", ""),
            brainregion_version=d.get("brainregion_version", ""),
            has_query=bool(d.get("has_query", False)),
            query_label=d.get("query_label", ""),
            kpis=[Kpi.from_dict(k) for k in d.get("kpis", [])],
            regions=[RegionSnapshot.from_dict(r) for r in d.get("regions", [])],
            manifest=[ManifestExperience.from_dict(m) for m in d.get("manifest", [])],
            activation=d.get("activation"),
            memory=d.get("memory", {}) or {},
            runs=d.get("runs", {}) or {},
            calibration=d.get("calibration", {}) or {},
        )


def build_snapshot(
    *,
    goal: str = "",
    problem: str = "",
    context: str = "",
    files: dict | None = None,
    gold_regions: list[str] | None = None,
    run_id: str | None = None,
    region: str | None = None,
    judge_id: str | None = None,
    history_limit: int = 20,
    memory_preview_k: int = 5,
    top_k: int = 3,
    escalate_confidence: float = 0.5,
    shadow_wake_threshold: float | None = None,
) -> BrainSnapshot:
    """投影 Inspector 输出 → BrainSnapshot。

    恒取 memory/run/calibration;仅当 problem 或 goal 非空才取 activation(无查询的空 wake 无意义)。
    """
    has_query = bool(problem or goal)
    # inspect(view=...) 返回 {section: {...}} 包一层；这里拆出各 section 的 dict。
    memory = _inspect(view="memory", region=region, memory_preview_k=memory_preview_k,
                      memory_manifest=True)["memory"]
    runs = _inspect(view="run", run_id=run_id, history_limit=history_limit)["run"]
    calibration = _inspect(view="calibration", judge_id=judge_id)["calibration"]
    activation = None
    if has_query:
        activation = _inspect(
            view="activation", goal=goal, problem=problem, context=context, files=files or {},
            gold_regions=gold_regions, escalate_confidence=escalate_confidence,
            shadow_wake_threshold=shadow_wake_threshold, top_k=top_k,
        )["activation"]

    regions = _build_regions(memory, activation)
    kpis = _build_kpis(memory, regions, activation, runs)
    manifest = [ManifestExperience.from_dict(m) for m in (memory.get("manifest") or [])]
    return BrainSnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        brainregion_version=__version__,
        has_query=has_query,
        query_label=(problem or goal)[:80],
        kpis=kpis,
        regions=regions,
        manifest=manifest,
        activation=activation,
        memory=memory,
        runs=runs,
        calibration=calibration,
    )


def _build_regions(memory: dict, activation: dict | None) -> list[RegionSnapshot]:
    """从 by_region(total)+ by_region_recallable + activation.woken 派生 region snapshots(按 total 降序)。"""
    by_region = memory.get("by_region") or {}
    by_region_recallable = memory.get("by_region_recallable") or {}
    woken = set((activation or {}).get("woken") or [])
    matrix = {
        row.get("id"): row
        for row in ((activation or {}).get("region_matrix") or [])
        if row.get("id")
    }
    has_query = activation is not None
    region_ids = list(dict.fromkeys([*matrix.keys(), *by_region.keys()]))
    out = []
    for r in region_ids:
        total = int(by_region.get(r, 0))
        row = matrix.get(r) or {}
        phase = row.get("phase") or ("woken" if r in woken else ("quiet" if has_query else "unknown"))
        out.append(RegionSnapshot(
            region=r,
            total=total,
            recallable=int(by_region_recallable.get(r, 0)),
            woke=("yes" if r in woken else ("no" if has_query else "unknown")),
            score=int(row.get("score") or 0),
            confidence=float(row.get("confidence") or 0.0),
            phase=phase,
            suggested_actions=int(row.get("suggested_actions") or 0),
            action_tools=tuple(row.get("action_tools") or ()),
        ))
    if has_query:
        out.sort(key=lambda x: (-x.confidence, -x.score, x.region))
    else:
        out.sort(key=lambda x: x.total, reverse=True)
    return out


def _build_kpis(memory: dict, regions: list[RegionSnapshot],
                activation: dict | None, runs: dict) -> list[Kpi]:
    health = memory.get("health") or {}
    total = int(memory.get("total", 0))
    recallable = int(health.get("recallable", 0))
    non_recallable = total - recallable

    # KPI 1: Memory recallable/total
    if total == 0:
        mem_status, mem_hint = _NEUTRAL, "暂无记忆"
    else:
        ratio = non_recallable / total
        mem_status = _OK if ratio < 0.2 else (_WARN if ratio < 0.5 else _BAD)
        mem_hint = f"{non_recallable} 条失效（已覆盖/错误/过期）"
    mem_kpi = Kpi(label="记忆", value=f"{recallable} / {total} 可召回",
                  status=mem_status, hint=mem_hint)

    # KPI 2: Regions total + woke
    n_regions = len(regions)
    if activation is not None:
        woke_n = len([r for r in regions if r.woke == "yes"])
        reg_hint = f"{woke_n} 唤醒"
    else:
        reg_hint = "无激活查询"
    reg_kpi = Kpi(label="脑区", value=f"共 {n_regions} 个", status=_NEUTRAL, hint=reg_hint)

    # KPI 3: Last Run gate decision
    decision = _last_run_decision(runs)
    run_kpi = Kpi(label="最近 Run", value=decision or "无 Run",
                  status=_decision_status(decision), hint="")
    return [mem_kpi, reg_kpi, run_kpi]


def _last_run_decision(runs: dict) -> str | None:
    """取最近 run 的 gate decision。run_id 给定 → 该 run 的 gate;否则 history[0].status。"""
    if not runs:
        return None
    if "gate" in runs:  # run_id 路径(inspect_run 单 run)
        gate = runs.get("gate") or {}
        dec = gate.get("decision")
        if dec:
            return str(dec)
        # 无 gate:看 summary sanity
        summary = runs.get("summary") or {}
        sanity = summary.get("sanity") or {}
        return "FAIL" if sanity.get("errors") else "OK"
    history = runs.get("history") or []
    if not history:
        return None
    return history[0].get("status")


def _decision_status(decision: str | None) -> str:
    d = (decision or "").upper()
    if d == "GO":
        return _OK
    if "NO_GO" in d or d == "FAIL":
        return _BAD
    if "INCONCLUSIVE" in d:
        return _WARN
    return _NEUTRAL
