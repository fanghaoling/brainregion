"""可视化 Phase 2(Brain Diff)测试:ManifestExperience + manifest 补强 + schema 2 +
build_diff(experience 级/聚合/summary 预算)+ DiffHtmlRenderer(零业务逻辑/XSS)+ CLI --diff 不变量。

UNITY_PROJECT_ROOT=tmp 隔离 DB。diff 单元用直接构造 BrainSnapshot(不依赖 DB)。
"""
from __future__ import annotations

import json

import pytest

from brainregion.eval import store as eval_store
from brainregion.memory import store as memory_store
from brainregion.viz import (
    BrainSnapshot,
    ManifestExperience,
    SNAPSHOT_SCHEMA_VERSION,
    build_diff,
    build_snapshot,
    render_diff,
)
from brainregion.viz.diff import (
    ADDED, MULTI, REMOVED, STATUS_CHANGED, TRIGGER_CHANGED,
)


@pytest.fixture
def root(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    return tmp_path


def _me(eid, region="r", status="active", summary="s", triggers=("k",), created_at=""):
    return ManifestExperience(id=eid, region=region, status=status, summary=summary,
                              triggers=triggers, created_at=created_at)


def _snap(manifest, *, memory=None, runs=None, calibration=None, kpis=None, query_label=""):
    """构造最小 BrainSnapshot 供 diff 测试(不依赖 DB)。"""
    total = len(manifest)
    return BrainSnapshot(
        generated_at="2026-07-02T10:00:00+00:00", brainregion_version="0.1.0",
        has_query=bool(query_label), query_label=query_label,
        kpis=kpis or [], manifest=list(manifest),
        memory=memory or {"total": total, "health": {"by_status": {"active": total}}},
        runs=runs or {}, calibration=calibration or {},
    )


# ── ManifestExperience 单元(GPT round2 ①②③)──────────────────────────────────

def test_manifest_normalizes_triggers_sorted_dedup():
    m = ManifestExperience(id="x", region="r", status="active", summary="s", triggers=["b", "a", "a", "c"])
    assert m.triggers == ("a", "b", "c")  # sorted + 去重


def test_manifest_truncates_summary():
    m = ManifestExperience(id="x", region="r", status="active", summary="字" * 200, triggers=())
    assert len(m.summary) == 120


def test_semantic_equals_ignores_created_at_and_id():
    a = _me("1", summary="s", triggers=("a", "b"), created_at="2026-01-01")
    b = _me("999", summary="s", triggers=("a", "b"), created_at="2026-12-31")  # id/created_at 不同
    assert a.semantic_equals(b)
    c = _me("1", status="superseded", summary="s", triggers=("a", "b"))
    assert not a.semantic_equals(c)  # status 变 → 不等


def test_triggers_order_independent_for_equality():
    a = _me("1", triggers=("a", "b"))
    b = _me("1", triggers=["b", "a"])  # 入参反序 → 构造归一后同
    assert a.triggers == b.triggers == ("a", "b")
    assert a.semantic_equals(b)


# ── inspect_memory manifest 补强 + schema 2 ───────────────────────────────────

def test_inspect_memory_manifest_opt_in(root):
    from brainregion.inspector import memory as mem_view
    memory_store.record_experience(summary="m", triggers=["a", "b"], region="unity_ecs")
    lean = mem_view.inspect_memory()
    assert "manifest" not in lean  # 默认精简(live inspect 不爆)
    full = mem_view.inspect_memory(manifest=True)
    man = full["manifest"]
    assert len(man) == 1
    e = man[0]
    assert e["id"] and e["region"] == "unity_ecs" and set(e.keys()) == \
        {"id", "region", "status", "summary", "triggers", "created_at"}
    assert e["triggers"] == ["a", "b"]


def test_build_snapshot_has_manifest_query_label_schema2(root):
    memory_store.record_experience(summary="x", triggers=["k"], region="unity_ecs")
    snap = build_snapshot(problem="unity 寻路 task")
    assert snap.schema_version == 2 == SNAPSHOT_SCHEMA_VERSION
    assert len(snap.manifest) == 1 and snap.manifest[0].region == "unity_ecs"
    assert snap.query_label == "unity 寻路 task"


def test_schema_roundtrip_v2_and_v1_compat():
    snap = _snap([_me("1")], query_label="q")
    d = snap.to_dict()
    assert d["schema_version"] == 2 and d["query_label"] == "q" and len(d["manifest"]) == 1
    # v2 往返
    snap2 = BrainSnapshot.from_dict(d)
    assert len(snap2.manifest) == 1 and snap2.query_label == "q"
    # v1 缺 manifest/query_label → 默认空,不报错
    v1 = BrainSnapshot.from_dict({"schema_version": 1, "kpis": [], "regions": []})
    assert v1.manifest == [] and v1.query_label == ""
    # 过新 → 拒
    with pytest.raises(ValueError, match="schema_version"):
        BrainSnapshot.from_dict({"schema_version": 999})


# ── build_diff:experience 级(GPT r1② before/after + r2①⑤)────────────────────

def test_diff_added_removed_changed():
    a = _snap([_me("1", summary="keep"), _me("2", summary="gone"), _me("3", status="active")])
    b = _snap([_me("1", summary="keep"), _me("3", status="superseded"), _me("4", summary="new")])
    d = build_diff(a, b)
    added = [c.after.id for c in d.memory["added"]]
    removed = [c.before.id for c in d.memory["removed"]]
    changed = {c.before.id: c.change_kind for c in d.memory["changed"]}
    assert added == ["4"] and all(c.before is None and c.change_kind == ADDED for c in d.memory["added"])
    assert removed == ["2"] and all(c.after is None and c.change_kind == REMOVED for c in d.memory["removed"])
    assert changed == {"3": STATUS_CHANGED}


def test_diff_change_kinds_status_trigger_multi():
    base = [_me("1", status="active", triggers=("a",), summary="s1"),
            _me("2", status="active", triggers=("x",), summary="s2"),
            _me("3", status="active", triggers=("p",), summary="s3")]
    after = [_me("1", status="superseded", triggers=("a",), summary="s1"),   # status→STATUS_CHANGED
             _me("2", status="active", triggers=("y",), summary="s2"),       # trigger→TRIGGER_CHANGED
             _me("3", status="pending", triggers=("q",), summary="s3*")]     # 多→MULTI
    d = build_diff(_snap(base), _snap(after))
    kinds = {c.before.id: c.change_kind for c in d.memory["changed"]}
    assert kinds == {"1": STATUS_CHANGED, "2": TRIGGER_CHANGED, "3": MULTI}


def test_diff_triggers_reordered_not_changed():
    a = _snap([_me("1", triggers=("a", "b"))])
    b = _snap([_me("1", triggers=["b", "a"])])  # 同集合,反序
    d = build_diff(a, b)
    assert d.memory["changed"] == []  # 归一后等 → 不算 changed


# ── build_diff:聚合 + summary 预算(GPT r2④)──────────────────────────────────

def test_diff_summary_precomputed_matches_lists():
    a = _snap([_me("1"), _me("2", region="r2")],
              memory={"total": 2, "health": {"by_status": {"active": 2}}},
              runs={"history": [{"run_id": "r1", "status": "GO"}]})
    b = _snap([_me("1"), _me("2", region="r2"), _me("3", region="r3")],
              memory={"total": 3, "health": {"by_status": {"active": 3}}},
              runs={"history": [{"run_id": "r1", "status": "GO"}, {"run_id": "r2", "status": "NO_GO"}]})
    d = build_diff(a, b)
    s = d.summary
    assert s["added"] == 1 and s["removed"] == 0 and s["changed"] == 0
    assert s["regions_added"] == 1 and s["new_runs"] == 1
    assert s["total_a"] == 2 and s["total_b"] == 3
    # gate 来自「最近 Run」KPI;此 fixture 无 kpis → 两边空(gate diff 在 kpis 测试里覆盖)


def test_diff_kpis_paired_by_label():
    from brainregion.viz import Kpi
    a = _snap([], kpis=[Kpi("记忆", "1 / 1 可召回", "ok"), Kpi("最近 Run", "GO", "ok")])
    b = _snap([], kpis=[Kpi("记忆", "1 / 2 可召回", "warn"), Kpi("最近 Run", "NO_GO", "bad")])
    d = build_diff(a, b)
    assert d.summary["gate_a"] == "GO" and d.summary["gate_b"] == "NO_GO"
    assert [k.label for k in d.kpis_a] == [k.label for k in d.kpis_b]


def test_diff_runs_focused_run_note():
    a = _snap([], runs={"history": [{"run_id": "r1", "status": "GO"}]})
    b = _snap([], runs={"gate": {"decision": "GO"}, "run": {"run_id": "r1"}, "timeline": []})  # focused-run
    d = build_diff(a, b)
    assert d.runs["new"] == []
    assert any("focused-run" in n for n in d.notes)


def test_diff_calibration_blocked_flip_and_new_not_passed():
    a = _snap([], calibration={"am_i_blocked": False, "not_passed": []})
    b = _snap([], calibration={"am_i_blocked": True,
               "not_passed": [{"judge_id": "j9", "judge_model": "m"}]})
    d = build_diff(a, b)
    assert d.calibration["blocked_a"] is False and d.calibration["blocked_b"] is True
    assert d.calibration["new_not_passed"][0]["judge_id"] == "j9"


# ── v1 降级 ────────────────────────────────────────────────────────────────────

def test_diff_v1_degrades_to_aggregate():
    v1 = BrainSnapshot.from_dict({"schema_version": 1, "kpis": [], "regions": [],
                                  "memory": {"total": 0, "health": {"by_status": {}}}})  # 无 manifest
    v2 = _snap([_me("1")])
    d = build_diff(v1, v2)
    assert d.memory["added"] == [] and d.memory["changed"] == []  # experience 级空
    assert any("v1" in n for n in d.notes)
    assert d.summary["total_b"] == 1  # 聚合级仍工作


# ── DiffHtmlRenderer:零业务逻辑 + 顺序 + XSS + 自包含 ─────────────────────────

def test_diff_html_reads_summary_and_orders_sections():
    a = _snap([_me("1", status="active"), _me("2", summary="gone")])
    b = _snap([_me("1", status="superseded"), _me("3", summary="new")])
    d = build_diff(a, b)
    h = render_diff(d)
    assert h.startswith("<!DOCTYPE html>")
    assert "Executive Summary" in h
    # 顺序:变化(Changed)在 新增(Added)之前,新增在 移除(Removed)之前
    pos_chg, pos_add, pos_rem = (h.find("变化</h2>"), h.find("新增</h2>"), h.find("移除</h2>"))
    assert 0 < pos_chg < pos_add < pos_rem


def test_diff_html_xss_and_selfcontained():
    a = _snap([_me("1", summary="<script>alert(1)</script>", triggers=("<img>",))])
    b = _snap([_me("1", summary="<script>alert(2)</script>", triggers=("<svg>",))])  # SUMMARY_CHANGED + TRIGGER
    h = render_diff(build_diff(a, b))
    assert "&lt;script&gt;" in h and "<script>alert" not in h  # summary 转义
    assert "<script" not in h and "src=" not in h and "http" not in h  # 自包含/零 JS


# ── CLI --diff 不变量(不调 Inspector/DB)────────────────────────────────────────

def test_cli_diff_does_not_touch_inspector_or_store(root, monkeypatch, tmp_path, capsys):
    from brainregion.cli import build_parser, run_snapshot
    # 先正常 build 两份 snapshot 落盘
    memory_store.record_experience(summary="base", triggers=["k"], region="unity_ecs")
    a_file = tmp_path / "a.json"
    b_file = tmp_path / "b.json"
    out_file = tmp_path / "diff.html"
    build_snapshot().to_dict  # 触发 import
    a = build_snapshot()
    memory_store.record_experience(summary="added later", triggers=["k"], region="blender")
    b = build_snapshot()
    a_file.write_text(json.dumps(a.to_dict()), encoding="utf-8")
    b_file.write_text(json.dumps(b.to_dict()), encoding="utf-8")

    # monkeypatch:Inspector/store 全 raise——--diff 不得碰
    def _boom(*ar, **kw):
        raise AssertionError("--diff must not call Inspector or store")
    monkeypatch.setattr("brainregion.viz.snapshot._inspect", _boom)
    monkeypatch.setattr(memory_store, "list_experiences", _boom)
    monkeypatch.setattr(eval_store, "fetch_run", _boom)

    args = build_parser().parse_args(["snapshot", "--diff", str(a_file), str(b_file),
                                      "--out", str(out_file), "--label-a", "before", "--label-b", "after"])
    run_snapshot(args)
    text = out_file.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>") and "Executive Summary" in text
    assert "before" in text and "after" in text
