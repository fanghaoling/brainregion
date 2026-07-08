"""可视化 Phase 1 测试：build_snapshot + 序列化双向 + --from 不变量 + HtmlRenderer(XSS/自包含)+ CLI。

UNITY_PROJECT_ROOT=tmp 隔离 eval.db + brain_region_reviews.db（复用 test_inspector 的 seed 套路）。
"""
from __future__ import annotations

import json

import pytest

from brainregion.eval import store as eval_store
from brainregion.eval.schema import EvalCaseRecord, EvalLedgerEntry
from brainregion.memory import governance, store as memory_store
from brainregion.viz import (
    BrainSnapshot,
    HtmlRenderer,
    Kpi,
    RegionSnapshot,
    SNAPSHOT_SCHEMA_VERSION,
    build_snapshot,
    render,
    render_html,
)


# ── fixtures / seed ──────────────────────────────────────────────────────────

@pytest.fixture
def root(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    return tmp_path


def _seed_run(run_id="r1", decision="GO", date="2026-07-02T00:00:00+00:00", n_tasks=1):
    summary = {"per_variant": {}, "gate": {"decision": decision}}
    eval_store.record_run(EvalLedgerEntry(
        run_id=run_id, date=date, git_sha="abc", variants=["retrieve_on"],
        judge_models=["j"], rubric_hash="rh", n_tasks=n_tasks, summary=summary,
    ))


def _fake_wake(woken):
    """构造 wake_gate 的返回形状（inspect_activation._summarize_activation 消费）。"""
    return {
        "activated_regions": {"woken": woken, "retrieved": [], "escalated": woken,
                              "shadow": [], "reasons": {}, "confidence": {}},
        "wake_metrics": {"hit": woken, "missed": [], "false_wake": [],
                         "metrics_status": "unscored"},
        "suggested_actions": [],
        "trace": {"shadow_promoted": 0, "sentinel_hits": []},
    }


# ── build_snapshot：无查询 ─────────────────────────────────────────────────────

def test_build_snapshot_no_query_skips_activation(root, monkeypatch):
    """无 problem/goal → activation=None，且 wake_gate 绝不被调。"""
    def _boom(**k):
        raise AssertionError("wake_gate must not be called without a query")
    monkeypatch.setattr("brainregion.inspector.activation.wake_gate", _boom)

    memory_store.record_experience(summary="m1", triggers=["k"], region="unity_ecs")
    snap = build_snapshot()
    assert snap.activation is None
    assert snap.has_query is False
    assert len(snap.kpis) == 3
    assert {k.label for k in snap.kpis} == {"记忆", "脑区", "最近 Run"}


def test_build_snapshot_regions_from_by_region(root):
    memory_store.record_experience(summary="a", triggers=["k"], region="unity_ecs")
    memory_store.record_experience(summary="b", triggers=["k"], region="unity_ecs")
    memory_store.record_experience(summary="c", triggers=["k"], region="blender",
                                   status=governance.SUPERSEDED)  # 不可召回
    snap = build_snapshot()
    by_region = {r.region: r for r in snap.regions}
    assert by_region["unity_ecs"].total == 2
    assert by_region["unity_ecs"].recallable == 2
    assert by_region["blender"].total == 1
    assert by_region["blender"].recallable == 0  # superseded → 不可召回
    assert all(r.woke == "unknown" for r in snap.regions)  # 无查询


# ── build_snapshot：有查询 ────────────────────────────────────────────────────

def test_build_snapshot_with_query_marks_woke(root, monkeypatch):
    memory_store.record_experience(summary="u", triggers=["k"], region="unity_ecs")
    memory_store.record_experience(summary="b", triggers=["k"], region="blender")
    monkeypatch.setattr("brainregion.inspector.activation.wake_gate",
                        lambda **k: _fake_wake(["unity_ecs"]))
    snap = build_snapshot(problem="some unity task")
    assert snap.activation is not None
    assert snap.has_query is True
    by_region = {r.region: r for r in snap.regions}
    assert by_region["unity_ecs"].woke == "yes"
    assert by_region["blender"].woke == "no"


# ── KPI 正确性 ─────────────────────────────────────────────────────────────────

def test_kpi_memory_and_last_run(root):
    for i in range(3):
        memory_store.record_experience(summary=f"active-{i}", triggers=["k"], region="r")
    for i in range(2):
        memory_store.record_experience(summary=f"bad-{i}", triggers=["k"], region="r",
                                       status=governance.WRONG)
    _seed_run("r1", decision="GO")
    snap = build_snapshot()
    kpi_by_label = {k.label: k for k in snap.kpis}
    assert kpi_by_label["记忆"].value == "3 / 5 可召回"
    # 2/5 = 0.4 inactive → warn（< 0.5 但 ≥ 0.2）
    assert kpi_by_label["记忆"].status == "warn"
    assert kpi_by_label["最近 Run"].value == "GO"
    assert kpi_by_label["最近 Run"].status == "ok"


def test_kpi_last_run_no_runs_neutral(root):
    snap = build_snapshot()
    kpi_by_label = {k.label: k for k in snap.kpis}
    assert kpi_by_label["最近 Run"].value == "无 Run"
    assert kpi_by_label["最近 Run"].status == "neutral"


# ── 序列化双向 + schema_version ────────────────────────────────────────────────

def test_roundtrip_to_dict_from_dict(root):
    memory_store.record_experience(summary="x", triggers=["k"], region="unity_ecs")
    _seed_run("r1", decision="NO_GO")
    snap = build_snapshot(problem="q")  # 带查询，activation 非 None
    d = snap.to_dict()
    assert d["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert d["brainregion_version"]  # 非空
    assert d["activation"] is not None

    snap2 = BrainSnapshot.from_dict(d)
    assert snap2.schema_version == SNAPSHOT_SCHEMA_VERSION
    assert [k.label for k in snap2.kpis] == [k.label for k in snap.kpis]
    assert {r.region for r in snap2.regions} == {r.region for r in snap.regions}
    assert snap2.activation is not None
    # 渲染从 from_dict 重建的 snapshot 也能出 HTML
    assert render_html(snap2).startswith("<!DOCTYPE html>")


def test_from_dict_rejects_newer_schema():
    bad = {"schema_version": 999, "kpis": [], "regions": []}
    with pytest.raises(ValueError, match="schema_version"):
        BrainSnapshot.from_dict(bad)


def test_dataclass_to_dict_from_dict_units():
    k = Kpi(label="L", value="V", status="ok", hint="h")
    assert Kpi.from_dict(k.to_dict()) == k
    r = RegionSnapshot(region="r", total=3, recallable=2, woke="yes")
    assert RegionSnapshot.from_dict(r.to_dict()) == r


# ── --from 硬不变量：渲染不碰 Inspector/DB ─────────────────────────────────────

def test_from_dict_render_does_not_touch_inspector_or_store(root, monkeypatch):
    """save→render 确定性：from_dict + render 绝不调 inspect/store。"""
    memory_store.record_experience(summary="x", triggers=["k"], region="r")
    saved = build_snapshot().to_dict()

    def _boom(*a, **k):
        raise AssertionError("from_dict/render must not call Inspector or store")
    monkeypatch.setattr("brainregion.viz.snapshot._inspect", _boom)
    monkeypatch.setattr(memory_store, "list_experiences", _boom)
    monkeypatch.setattr(eval_store, "fetch_run", _boom)

    # 对照：build_snapshot 现在会炸（证明 monkeypatch 真的在守）
    with pytest.raises(AssertionError):
        build_snapshot()
    # 真测：from_dict + render 不炸
    snap = BrainSnapshot.from_dict(saved)
    html_out = render_html(snap)
    assert "<!DOCTYPE html>" in html_out


# ── HtmlRenderer：内容 / XSS / 自包含 / timeline ───────────────────────────────

def test_html_contains_kpis_regions_headers(root):
    memory_store.record_experience(summary="hello", triggers=["k"], region="unity_ecs")
    html_out = render_html(build_snapshot())
    assert "脑状态快照" in html_out
    assert "记忆" in html_out and "脑区" in html_out and "最近 Run" in html_out
    assert "unity_ecs" in html_out


def test_html_xss_escapes_memory_summary(root):
    """memory summary 带 <script> → 输出转义，无裸 <script>alert。"""
    memory_store.record_experience(summary="<script>alert(1)</script>", triggers=["k"], region="r")
    html_out = render_html(build_snapshot())
    assert "&lt;script&gt;" in html_out
    assert "<script>alert" not in html_out


def test_html_selfcontained_no_external_no_js(root):
    html_out = render_html(build_snapshot())
    assert "<style>" in html_out
    assert "<script" not in html_out       # 零 JS
    assert "src=" not in html_out           # 无外部资源引用
    assert "http://" not in html_out and "https://" not in html_out


def test_html_timeline_when_run_id(root):
    """--run 给定 → run detail 段含 timeline 阶段符号。"""
    _seed_run("r1", decision="GO")
    eval_store.record_case(EvalCaseRecord(
        run_id="r1", task_id="t1", variant="retrieve_on", report_summary={},
        retrieved_case_ids=["c1"], cost={}, latency_ms=0.0, outputs_json='{"x":1}', error="",
    ))
    html_out = render_html(build_snapshot(run_id="r1"))
    assert "Run 详情" in html_out
    assert "检索" in html_out   # 阶段列头（中文）


# ── Renderer 协议 + 分发 ───────────────────────────────────────────────────────

def test_render_dispatch_unknown_format(root):
    snap = build_snapshot()
    assert isinstance(render(snap, "html"), str)
    with pytest.raises(ValueError, match="render format"):
        render(snap, "bogus")


def test_htmlrenderer_satisfies_protocol():
    from brainregion.viz.render import Renderer
    assert isinstance(HtmlRenderer(), Renderer)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def test_cli_snapshot_json_stdout(root, monkeypatch, capsys):
    from brainregion.cli import build_parser, run_snapshot
    memory_store.record_experience(summary="fact", triggers=["k"], region="unity_ecs")
    args = build_parser().parse_args(["snapshot", "--json"])
    run_snapshot(args)
    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert any(r["region"] == "unity_ecs" for r in out["regions"])


def test_cli_snapshot_out_writes_html(root, tmp_path):
    from brainregion.cli import build_parser, run_snapshot
    out_file = tmp_path / "snap.html"
    args = build_parser().parse_args(["snapshot", "--out", str(out_file)])
    run_snapshot(args)
    text = out_file.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "<style>" in text


def test_cli_snapshot_save_then_from_renders(root, tmp_path):
    """--save 落盘 → --from 读回渲染（不调 Inspector/DB）。"""
    from brainregion.cli import build_parser, run_snapshot
    memory_store.record_experience(summary="persisted", triggers=["k"], region="unity_ecs")
    save_file = tmp_path / "snap.json"
    out_file = tmp_path / "from.html"
    throwaway_html = tmp_path / "throwaway.html"  # --save 默认仍写 HTML,导向 tmp 不污染 cwd

    args = build_parser().parse_args(["snapshot", "--save", str(save_file), "--out", str(throwaway_html)])
    run_snapshot(args)
    assert save_file.exists()
    data = json.loads(save_file.read_text(encoding="utf-8"))
    assert data["schema_version"] == SNAPSHOT_SCHEMA_VERSION

    args2 = build_parser().parse_args(["snapshot", "--from", str(save_file), "--out", str(out_file)])
    run_snapshot(args2)
    assert out_file.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


# ── Inspector memory 补强回归 ──────────────────────────────────────────────────

def test_inspect_memory_returns_by_region_recallable(root):
    from brainregion.inspector import memory as mem_view
    memory_store.record_experience(summary="a", triggers=["k"], region="unity_ecs")
    memory_store.record_experience(summary="b", triggers=["k"], region="unity_ecs",
                                   status=governance.WRONG)
    res = mem_view.inspect_memory()
    assert res["by_region_recallable"]["unity_ecs"] == 1  # 只有 active 那条
    assert res["by_region"]["unity_ecs"] == 2


def test_env_scene_html_builds():
    """场景查看页(Phase C+):build_env_scene_html 自包含 + 含网格/SSE/中文。"""
    from brainregion.viz.env_scene import build_env_scene_html
    html = build_env_scene_html()
    assert "<html" in html and 'id="grid"' in html        # 网格渲染容器
    assert "EventSource" in html                           # live SSE
    assert "场景" in html                                  # 中文(viz-output-chinese)
    assert "env.step" in html                              # 过滤 env 事件


def test_env_scene_route_served():
    """/scene 路由 serve 场景页(集成:起 debug_server + urllib 取 /scene)。"""
    import threading
    import time
    import urllib.request

    from brainregion.viz.debug_server import DebugDashboardOptions, serve_debug_dashboard
    opts = DebugDashboardOptions(port=8801, goal="t")
    t = threading.Thread(target=serve_debug_dashboard, args=(opts,), daemon=True)
    t.start()
    time.sleep(1.0)
    try:
        scene = urllib.request.urlopen("http://127.0.0.1:8801/scene", timeout=5).read().decode("utf-8")
        dash = urllib.request.urlopen("http://127.0.0.1:8801/", timeout=5).read().decode("utf-8")
    finally:
        pass  # daemon thread 随进程退
    assert "GridWorld 场景查看" in scene and 'id="grid"' in scene
    assert "BrainRegion 调试面板" in dash  # 原 dashboard 路由不受影响


def test_runs_index_lists_replay_htmls(tmp_path):
    """list_env_runs 扫盘抽 meta + build_runs_index_html 出场次表。"""
    from brainregion.viz.runs_index import build_runs_index_html, list_env_runs
    from brainregion.sandbox.envs import write_replay_html
    # 造两场 replay
    write_replay_html(tmp_path / "env-111.html", ["@"], {"model": "m1", "size": 4, "solved": True,
                         "total_reward": 1.0, "n_steps": 5, "termination": "done", "visibility_radius": None})
    write_replay_html(tmp_path / "env-222.html", ["@.?"], {"model": "m2", "size": 6, "solved": False,
                         "total_reward": 0.0, "n_steps": 30, "termination": "max_steps", "visibility_radius": 2})
    runs = list_env_runs(tmp_path)
    assert len(runs) == 2
    ids = {r["run_id"] for r in runs}
    assert ids == {"env-111", "env-222"}
    # meta 抽出
    m1 = next(r["meta"] for r in runs if r["run_id"] == "env-111")
    assert m1["model"] == "m1" and m1["solved"] is True
    html = build_runs_index_html(runs)
    assert "过去场次" in html and "/replay/env-111" in html and "/replay/env-222" in html
    assert "✅ 解出" in html and "❌ 未解" in html
