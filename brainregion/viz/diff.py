"""BrainDiff:两 BrainSnapshot 的对比(可视化 Phase 2)。

回答「脑状态前后变了什么」。核心是 **experience 级 diff**(哪些记忆新增/移除/语义变了),
基于 snapshot 的 memory manifest(全量 ManifestExperience 清单)。纯逻辑——预算所有计数到
``diff.summary``,**renderer 零业务逻辑**(GPT round2 ④)。

A = before,B = after,Δ = B − A。store 是 append-only → removed 罕见(真删才出现),主信号是
added + changed(status/trigger/summary/region 变)。changed 用 ManifestExperience.semantic_equals 判
(忽略 id/created_at/未来 metadata,GPT round2 ①),change_kind 据差异字段定。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .snapshot import BrainSnapshot, Kpi, ManifestExperience

DIFF_SCHEMA_VERSION = 1

# change_kind 枚举
ADDED = "ADDED"
REMOVED = "REMOVED"
STATUS_CHANGED = "STATUS_CHANGED"
SUMMARY_CHANGED = "SUMMARY_CHANGED"
TRIGGER_CHANGED = "TRIGGER_CHANGED"
REGION_CHANGED = "REGION_CHANGED"
MULTI = "MULTI"


@dataclass(frozen=True)
class ExperienceChange:
    """一条记忆的变化。added→before=None;removed→after=None;changed→两侧 + change_kind。"""

    before: ManifestExperience | None
    after: ManifestExperience | None
    change_kind: str = ""


@dataclass(frozen=True)
class BrainDiff:
    """两 snapshot 的对比结果。summary 预算计数(renderer 直读);memory/regions/runs/calibration 是明细。"""

    schema_version: int = DIFF_SCHEMA_VERSION
    a_meta: dict = field(default_factory=dict)
    b_meta: dict = field(default_factory=dict)
    kpis_a: list[Kpi] = field(default_factory=list)
    kpis_b: list[Kpi] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    regions: dict = field(default_factory=dict)
    runs: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_diff(a: BrainSnapshot, b: BrainSnapshot, *, label_a: str = "A", label_b: str = "B") -> BrainDiff:
    """对比 a(before) vs b(after)。不调 Inspector/DB(纯 snapshot 数据)。"""
    notes: list[str] = []

    # ── experience 级(manifest,GPT r1② before/after + r2① semantic_equals + r2⑤ change_kind)──
    a_by_id = {e.id: e for e in a.manifest}
    b_by_id = {e.id: e for e in b.manifest}
    added: list[ExperienceChange] = []
    removed: list[ExperienceChange] = []
    changed: list[ExperienceChange] = []
    if not a.manifest or not b.manifest:
        # 一侧 v1(无 manifest)→ experience 级空,只做聚合
        if a.manifest != b.manifest:
            side = "A" if not a.manifest else "B"
            notes.append(f"{side} 是 v1 snapshot（无 manifest），experience 级 diff 不可用，仅聚合级")
    else:
        for eid, b_exp in b_by_id.items():
            if eid not in a_by_id:
                added.append(ExperienceChange(before=None, after=b_exp, change_kind=ADDED))
        for eid, a_exp in a_by_id.items():
            b_exp = b_by_id.get(eid)
            if b_exp is None:
                removed.append(ExperienceChange(before=a_exp, after=None, change_kind=REMOVED))
            elif not a_exp.semantic_equals(b_exp):
                changed.append(ExperienceChange(before=a_exp, after=b_exp,
                                                change_kind=_change_kind(a_exp, b_exp)))

    # ── 聚合:memory total/by_status ──
    bs_a = ((a.memory.get("health") or {}).get("by_status") or {})
    bs_b = ((b.memory.get("health") or {}).get("by_status") or {})
    memory_diff = {
        "total_a": (a.memory.get("total") or 0),
        "total_b": (b.memory.get("total") or 0),
        "by_status_a": dict(bs_a), "by_status_b": dict(bs_b),
        "added": added, "removed": removed, "changed": changed,
    }

    # ── regions(从 manifest 按 region 聚合)──
    ra = _region_counts(a.manifest)
    rb = _region_counts(b.manifest)
    regions_diff = {
        "added": sorted(r for r in rb if r not in ra),
        "removed": sorted(r for r in ra if r not in rb),
        "deltas": sorted(
            ({"region": r, "a": ra.get(r, 0), "b": rb.get(r, 0)}
             for r in set(ra) | set(rb) if ra.get(r, 0) != rb.get(r, 0)),
            key=lambda d: d["region"],
        ),
    }

    # ── runs(history 模式:run_id 集合差)──
    hist_a = _history_rows(a)
    hist_b = _history_rows(b)
    new_runs: list[dict] = []
    if hist_a is None or hist_b is None:
        notes.append("focused-run timeline diff defer（任一 snapshot 是单 run 详情模式，仅 history 模式可比）")
    else:
        a_ids = {r.get("run_id") for r in hist_a}
        new_runs = [{"run_id": r.get("run_id"), "status": r.get("status")}
                    for r in hist_b if r.get("run_id") not in a_ids]

    # ── calibration ──
    cal_a = a.calibration or {}
    cal_b = b.calibration or {}
    blocked_a = bool(cal_a.get("am_i_blocked"))
    blocked_b = bool(cal_b.get("am_i_blocked"))
    np_a_ids = {r.get("judge_id") for r in (cal_a.get("not_passed") or [])}
    new_not_passed = [r for r in (cal_b.get("not_passed") or [])
                      if r.get("judge_id") not in np_a_ids]

    # ── gate(复用「最近 Run」KPI 的 value = gate decision 或「无 Run」)──
    gate_a = _kpi_value(a, "最近 Run")
    gate_b = _kpi_value(b, "最近 Run")

    # ── 预算 summary(GPT r2④,renderer 直读)──
    summary = {
        "added": len(added), "removed": len(removed), "changed": len(changed),
        "regions_added": len(regions_diff["added"]), "regions_removed": len(regions_diff["removed"]),
        "new_runs": len(new_runs), "gate_a": gate_a, "gate_b": gate_b,
        "blocked_a": blocked_a, "blocked_b": blocked_b,
        "total_a": memory_diff["total_a"], "total_b": memory_diff["total_b"],
    }

    return BrainDiff(
        schema_version=DIFF_SCHEMA_VERSION,
        a_meta=_meta(a, label_a), b_meta=_meta(b, label_b),
        kpis_a=list(a.kpis), kpis_b=list(b.kpis),
        summary=summary,
        memory=memory_diff,
        regions=regions_diff,
        runs={"new": new_runs},
        calibration={"blocked_a": blocked_a, "blocked_b": blocked_b, "new_not_passed": new_not_passed},
        notes=notes,
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def _change_kind(a: ManifestExperience, b: ManifestExperience) -> str:
    """据语义字段差异定 change_kind(GPT r2⑤)。多字段 → MULTI。"""
    diffs = []
    if a.region != b.region:
        diffs.append(REGION_CHANGED)
    if a.status != b.status:
        diffs.append(STATUS_CHANGED)
    if a.summary != b.summary:
        diffs.append(SUMMARY_CHANGED)
    if a.triggers != b.triggers:
        diffs.append(TRIGGER_CHANGED)
    if len(diffs) > 1:
        return MULTI
    return diffs[0] if diffs else ""


def _region_counts(manifest) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in manifest:
        counts[e.region] = counts.get(e.region, 0) + 1
    return counts


def _history_rows(snap: BrainSnapshot) -> list[dict] | None:
    """history 模式 → 返回 rows;focused-run(单 run 详情)→ None。"""
    runs = snap.runs or {}
    if "history" in runs:
        return runs["history"] or []
    return None


def _kpi_value(snap: BrainSnapshot, label: str) -> str:
    for k in snap.kpis:
        if k.label == label:
            return k.value
    return ""


def _meta(snap: BrainSnapshot, label: str) -> dict:
    return {"label": label, "generated_at": snap.generated_at,
            "brainregion_version": snap.brainregion_version,
            "query_label": snap.query_label, "schema_version": snap.schema_version}
