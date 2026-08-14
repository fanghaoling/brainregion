"""Inspector 测试：facade + 5 态 timeline + read 路径 + provenance + 只读硬约束。

全部用 UNITY_PROJECT_ROOT=tmp 隔离 eval.db + brain_region_reviews.db（不污染真实库）。
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from brainregion import inspector
from brainregion.eval import store as eval_store
from brainregion.eval.schema import (
    BlindJudgement,
    CalibrationRecord,
    EvalCaseRecord,
    EvalLedgerEntry,
)
from brainregion.inspector import activation, calibration, memory as mem_view, run as run_view
from brainregion.memory import store as memory_store


# ── fixtures / seed helpers ───────────────────────────────────────────────────

@pytest.fixture
def eval_root(monkeypatch, tmp_path):
    """隔离 eval.db + brain_region_reviews.db 到 tmp。"""
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    return tmp_path


def _seed_run(run_id="r1", summary=None, date="2026-07-02T00:00:00+00:00", n_tasks=1):
    entry = EvalLedgerEntry(
        run_id=run_id, date=date, git_sha="abc123",
        variants=["retrieve_off", "retrieve_on"], judge_models=["fake-judge"],
        rubric_hash="rh", n_tasks=n_tasks, summary=summary or {"per_variant": {}},
    )
    eval_store.record_run(entry)
    return entry


def _seed_case(run_id, task_id, variant, report_summary=None, retrieved=None,
               outputs_json="", error=""):
    rec = EvalCaseRecord(
        run_id=run_id, task_id=task_id, variant=variant,
        report_summary=report_summary or {}, retrieved_case_ids=retrieved or [],
        cost={}, latency_ms=0.0, outputs_json=outputs_json, error=error,
    )
    eval_store.record_case(rec)
    return rec


def _seed_judgement(run_id, task_id, variant, scores=None, reason="",
                    judge_id="j1", judge_model="fake-judge"):
    j = BlindJudgement(
        run_id=run_id, task_id=task_id, judge_id=judge_id, judge_model=judge_model,
        rubric_hash="rh", variant=variant, scores=scores if scores is not None else {"useful": 3},
        reason=reason,
    )
    eval_store.record_judgement(j)
    return j


def _tl(res):
    """timeline → {(task_id, variant): stages}。"""
    return {(r["task_id"], r["variant"]): r["stages"] for r in res["timeline"]}


# ── store read 路径 + provenance ──────────────────────────────────────────────

def test_record_run_stamps_provenance(eval_root):
    _seed_run("r1", summary={"per_variant": {}})
    run = eval_store.fetch_run("r1")
    prov = run["summary"]["__provenance__"]
    assert prov["summary_schema"] == eval_store.SUMMARY_SCHEMA_VERSION
    assert "brainregion_version" in prov


def test_record_run_does_not_overwrite_prestamped_provenance(eval_root):
    pre = {"per_variant": {}, "__provenance__": {"brainregion_version": "9.9", "summary_schema": 7}}
    _seed_run("r1", summary=pre)
    run = eval_store.fetch_run("r1")
    assert run["summary"]["__provenance__"] == {"brainregion_version": "9.9", "summary_schema": 7}


def test_record_run_does_not_mutate_caller_summary(eval_root):
    summ = {"per_variant": {}}
    _seed_run("r1", summary=summ)
    assert "__provenance__" not in summ  # caller 的 dict 未被污染


def test_fetch_run_not_found(eval_root):
    assert eval_store.fetch_run("nope") is None


def test_list_runs_orders_desc_and_limits(eval_root):
    for i in range(5):
        _seed_run(f"r{i}", summary={"per_variant": {}}, date=f"2026-07-0{i+1}T00:00:00+00:00")
    rows = eval_store.list_runs(limit=3)
    assert len(rows) == 3
    assert [r["run_id"] for r in rows] == ["r4", "r3", "r2"]  # 最新优先


def test_connect_readonly_rejects_writes(eval_root):
    eval_store._connect()  # 建表
    conn = eval_store._connect_readonly()
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO eval_runs(run_id) VALUES('x')")


def test_fetch_funcs_parameterized_no_injection(eval_root):
    _seed_run("r1", summary={"per_variant": {}})
    _seed_case("r1", "t1", "retrieve_on", retrieved=["c1"])
    _seed_judgement("r1", "t1", "retrieve_on")
    # 含引号/分号的外部 run_id 不应破坏查询（参数化）
    assert eval_store.fetch_run("r1'; DROP TABLE eval_runs;--") is None
    assert eval_store.fetch_run("r1") is not None
    assert len(eval_store.fetch_cases("r1")) == 1
    assert len(eval_store.fetch_judgements("r1")) == 1


# ── 5 态 timeline ─────────────────────────────────────────────────────────────

def test_timeline_retrieve_states(eval_root):
    _seed_run("r1", summary={"per_variant": {}})
    _seed_case("r1", "t1", "retrieve_off", retrieved=[])          # off+0 → SUCCESS
    _seed_case("r1", "t2", "retrieve_off", retrieved=["c1"])      # off+有 → FAILED
    _seed_case("r1", "t3", "retrieve_on", retrieved=["c1"])       # on+有 → SUCCESS
    _seed_case("r1", "t4", "retrieve_on", retrieved=[])           # on+0 无 error → UNKNOWN
    tl = _tl(run_view.inspect_run(run_id="r1"))
    assert tl[("t1", "retrieve_off")]["retrieve"] == "SUCCESS"
    assert tl[("t2", "retrieve_off")]["retrieve"] == "FAILED"
    assert tl[("t3", "retrieve_on")]["retrieve"] == "SUCCESS"
    assert tl[("t4", "retrieve_on")]["retrieve"] == "UNKNOWN"


def test_timeline_memory_skipped_not_failed(eval_root):
    """GPT 三① 核心：wake 没召回 → Memory=SKIPPED（⏭）非 FAILED；老 run 无字段=NOT_INSTRUMENTED。"""
    _seed_run("r1", summary={"per_variant": {}})
    _seed_case("r1", "t1", "retrieve_on", report_summary={})                            # 无 memory 字段 → NOT_INSTRUMENTED
    _seed_case("r1", "t2", "retrieve_on", report_summary={"memory": {"retrieved": 2, "injected": 2}})  # SUCCESS
    _seed_case("r1", "t3", "retrieve_on", report_summary={"memory": {"retrieved": 0, "injected": 0}})  # SKIPPED
    _seed_case("r1", "t4", "retrieve_on", report_summary={"memory": {"retrieved": 3, "injected": 0}})  # FAILED（被 budget 截）
    tl = _tl(run_view.inspect_run(run_id="r1"))
    assert tl[("t1", "retrieve_on")]["memory"] == "NOT_INSTRUMENTED"
    assert tl[("t2", "retrieve_on")]["memory"] == "SUCCESS"
    assert tl[("t3", "retrieve_on")]["memory"] == "SKIPPED"
    assert tl[("t4", "retrieve_on")]["memory"] == "FAILED"


def test_timeline_consult_and_judge_states(eval_root):
    _seed_run("r1", summary={"per_variant": {}})
    _seed_case("r1", "t1", "retrieve_on", outputs_json='{"x":1}')   # consult SUCCESS
    _seed_case("r1", "t2", "retrieve_on", error="boom")             # consult FAILED
    _seed_case("r1", "t3", "retrieve_on", outputs_json='{"x":1}')   # 无 judgement → judge SKIPPED
    _seed_case("r1", "t4", "retrieve_on", outputs_json='{"x":1}')   # judge parse 全失败 → FAILED
    _seed_judgement("r1", "t1", "retrieve_on", scores={"useful": 3})             # judge SUCCESS
    _seed_judgement("r1", "t4", "retrieve_on", scores={}, reason="parse fail")   # judge FAILED
    tl = _tl(run_view.inspect_run(run_id="r1"))
    assert tl[("t1", "retrieve_on")]["consult"] == "SUCCESS"
    assert tl[("t1", "retrieve_on")]["judge"] == "SUCCESS"
    assert tl[("t2", "retrieve_on")]["consult"] == "FAILED"
    assert tl[("t3", "retrieve_on")]["judge"] == "SKIPPED"
    assert tl[("t4", "retrieve_on")]["judge"] == "FAILED"


def test_timeline_outcome_variant_retrieve_is_skipped(eval_root):
    """consult/outcome 变体不检索知识库 → retrieve 阶段 SKIPPED（非 FAILED）。"""
    _seed_run("r1", summary={"per_variant": {}}, n_tasks=1)
    _seed_case("r1", "t1", "routed", retrieved=[])  # 非 retrieve_ 前缀
    tl = _tl(run_view.inspect_run(run_id="r1"))
    assert tl[("t1", "routed")]["retrieve"] == "SKIPPED"


# ── inspect_run：读已存 summary（不重算）+ history + not-found + provenance ──

def test_inspect_run_reads_stored_summary_verbatim(eval_root):
    summ = {
        "per_variant": {"retrieve_on": {"useful_advice_rate": 0.8, "inference_cost_usd": 0.5}},
        "memory_diagnostics": {"primary": "relevant_vs_irrelevant"},
        "gate": {"decision": "GO"},
    }
    _seed_run("r1", summary=summ)
    res = run_view.inspect_run(run_id="r1")
    assert res["summary"]["per_variant"]["retrieve_on"]["useful_advice_rate"] == 0.8  # 原样未重算
    assert res["summary"]["memory_diagnostics"]["primary"] == "relevant_vs_irrelevant"
    assert res["gate"]["decision"] == "GO"
    assert res["provenance"]["summary_schema"] == eval_store.SUMMARY_SCHEMA_VERSION


def test_inspect_run_not_found_returns_error(eval_root):
    res = run_view.inspect_run(run_id="nope")
    assert "error" in res and "not found" in res["error"]


def test_inspect_run_no_run_id_returns_history(eval_root):
    _seed_run("r1", summary={"per_variant": {"retrieve_on": {"inference_cost_usd": 0.5}},
                             "gate": {"decision": "GO"}}, date="2026-07-01T00:00:00+00:00")
    _seed_run("r2", summary={"per_variant": {}, "sanity": {"errors": ["boom"]}},
              date="2026-07-02T00:00:00+00:00")
    res = run_view.inspect_run(run_id=None, history_limit=10)
    assert res["n"] == 2
    by_id = {r["run_id"]: r for r in res["history"]}
    assert by_id["r1"]["status"] == "GO"
    assert by_id["r1"]["cost_usd"] == 0.5
    assert by_id["r2"]["status"] == "FAIL"


def test_inspect_run_old_run_provenance_unknown(eval_root):
    """老 run（绕过 record_run 直写，无 __provenance__）→ Inspector 显示 unknown。"""
    conn = eval_store._connect()
    conn.execute(
        "INSERT INTO eval_runs(run_id,date,git_sha,n_tasks,summary) VALUES(?,?,?,?,?)",
        ("old", "2026-01-01", "x", 1, json.dumps({"per_variant": {}})),
    )
    conn.commit()
    res = run_view.inspect_run(run_id="old")
    assert res["provenance"]["summary_schema"] == "unknown"
    assert res["provenance"]["brainregion_version"] == "unknown"


# ── activation ────────────────────────────────────────────────────────────────

def test_activation_summarize_missed_and_unscored(eval_root, monkeypatch):
    def fake_wake(**k):
        gold = k.get("gold_regions") or []
        woken = ["a"]
        return {
            "activated_regions": {
                "woken": woken,
                "retrieved": [{"id": "a", "score": 3, "source": "trigger"}],
                "escalated": ["a"], "shadow": [], "reasons": {"a": "x"}, "confidence": {"a": 0.9},
            },
            "wake_metrics": {
                "hit": ["a"],
                "missed": sorted(set(gold) - set(woken)),
                "false_wake": [],
                "metrics_status": "scored" if gold else "unscored",
            },
            "suggested_actions": [],
            "trace": {"shadow_promoted": 0, "sentinel_hits": [], "models_called": False},
        }
    monkeypatch.setattr("brainregion.inspector.activation.wake_gate", fake_wake)
    r = activation.inspect_activation(problem="x", gold_regions=["a", "b"])
    assert r["woken"] == ["a"]
    assert r["wake_metrics"]["missed"] == ["b"]
    assert "漏唤醒" in r["explain"]
    # 无 gold → unscored，绝不伪装 0-漏
    r2 = activation.inspect_activation(problem="x")
    assert r2["wake_metrics"]["metrics_status"] == "unscored"
    assert "unscored" in r2["explain"]


def test_activation_reports_region_matrix_and_call_status(eval_root, monkeypatch):
    def fake_wake(**_k):
        return {
            "activated_regions": {
                "woken": ["memory"],
                "retrieved": [
                    {"id": "memory", "score": 8, "source": "text"},
                    {"id": "review", "score": 2, "source": "text"},
                ],
                "escalated": ["memory"],
                "shadow": [{"id": "review", "score": 2, "confidence": 0.25,
                            "promoted": False, "reason": "below shadow threshold"}],
                "reasons": {"memory": "matched 4 trigger(s)", "review": "matched 1 trigger(s)"},
                "confidence": {"memory": 1.0, "review": 0.25},
            },
            "wake_metrics": {"hit": ["memory"], "missed": [], "false_wake": [],
                             "metrics_status": "scored"},
            "suggested_actions": [
                {"tool": "recall_experiences", "source_regions": ["memory"],
                 "requires_user_approval": True},
            ],
            "trace": {"shadow_promoted": 0, "sentinel_hits": [], "models_called": False},
        }

    monkeypatch.setattr("brainregion.inspector.activation.wake_gate", fake_wake)
    res = activation.inspect_activation(problem="project understanding", gold_regions=["memory"])

    by_id = {row["id"]: row for row in res["region_matrix"]}
    assert by_id["memory"]["phase"] == "woken"
    assert by_id["memory"]["confidence"] == 1.0
    assert by_id["memory"]["suggested_actions"] == 1
    assert by_id["memory"]["action_tools"] == ["recall_experiences"]
    assert by_id["review"]["phase"] == "shadow"
    assert by_id["debugging"]["phase"] == "quiet"
    assert res["call_status"]["models_called"] is False
    assert res["call_status"]["suggested_actions_count"] == 1
    assert res["call_status"]["requires_user_approval_count"] == 1


# ── memory view ───────────────────────────────────────────────────────────────

def test_inspect_memory_counts_age_preview(eval_root):
    memory_store.record_experience(summary="hide char", triggers=["hide"], region="unity_ecs")
    memory_store.record_experience(summary="flowfield", triggers=["path"], region="unity_ecs")
    memory_store.record_experience(summary="global", triggers=["x"], region="")
    res = mem_view.inspect_memory(region=None, preview_k=2)
    assert res["total"] == 3
    assert res["by_region"]["unity_ecs"] == 2
    assert res["by_region"]["(global)"] == 1
    assert len(res["preview"]) == 2
    assert res["preview"][0]["age_days"] is not None


# ── calibration view ──────────────────────────────────────────────────────────

def test_inspect_calibration_blocked(eval_root):
    rec_ok = CalibrationRecord(judge_id="j1", judge_model="m1", rubric_hash="rh",
                               prompt_hash="ph", passed=True, agreement_rate=0.9,
                               wilson_lower=0.8, threshold=0.7)
    rec_bad = CalibrationRecord(judge_id="j2", judge_model="m2", rubric_hash="rh",
                                prompt_hash="ph", passed=False, agreement_rate=0.4,
                                wilson_lower=0.3, threshold=0.7)
    eval_store.record_calibration(rec_ok, {"note": "ok"})
    eval_store.record_calibration(rec_bad, {"note": "bad"})
    res = calibration.inspect_calibration()
    assert res["n"] == 2
    assert res["passed_count"] == 1
    assert res["am_i_blocked"] is True
    assert any(r["judge_id"] == "j2" for r in res["not_passed"])


# ── facade ────────────────────────────────────────────────────────────────────

def test_facade_unknown_view_raises(eval_root):
    with pytest.raises(ValueError):
        inspector.inspect(view="bogus")


def test_facade_memory_view_does_not_call_wakegate(eval_root, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("wake_gate must not be called for view=memory")
    monkeypatch.setattr("brainregion.inspector.activation.wake_gate", _boom)
    res = inspector.inspect(view="memory")
    assert set(res.keys()) == {"memory"}


def test_facade_run_view_without_runid_returns_history_not_fullscan(eval_root, monkeypatch):
    _seed_run("r1", summary={"per_variant": {}}, date="2026-07-01T00:00:00+00:00")
    res = inspector.inspect(view="run")  # 无 run_id
    assert "history" in res["run"]


def test_facade_all_returns_all_sections(eval_root, monkeypatch):
    monkeypatch.setattr(
        "brainregion.inspector.activation.wake_gate",
        lambda **k: {"activated_regions": {}, "wake_metrics": {"metrics_status": "unscored"},
                     "suggested_actions": [], "trace": {}},
    )
    res = inspector.inspect(view="all")
    assert set(res.keys()) == {"activation", "memory", "run", "calibration", "model_health"}


# ── 只读硬约束：全 view 跑一遍，record_* / record_experience 不得被调 ──────────

def test_inspect_never_writes(eval_root, monkeypatch):
    def _raising(name):
        def _f(*a, **k):
            raise AssertionError(f"Inspector must not call {name}")
        return _f
    for name in ("record_run", "record_case", "record_judgement", "record_calibration"):
        monkeypatch.setattr(eval_store, name, _raising(f"eval.store.{name}"))
    monkeypatch.setattr(memory_store, "record_experience", _raising("memory.record_experience"))
    monkeypatch.setattr(
        "brainregion.inspector.activation.wake_gate",
        lambda **k: {"activated_regions": {"woken": ["x"]},
                     "wake_metrics": {"metrics_status": "scored", "hit": ["x"],
                                      "missed": [], "false_wake": []},
                     "suggested_actions": [], "trace": {"shadow_promoted": 0, "sentinel_hits": []}},
    )
    res = inspector.inspect(view="all")  # 不应抛
    assert "activation" in res


# ── CLI 接线（build_parser + run_inspect）────────────────────────────────────

def test_cli_inspect_subcommand_wires_through(eval_root, monkeypatch, capsys):
    from brainregion.cli import build_parser, run_inspect

    memory_store.record_experience(summary="a fact", triggers=["t"], region="unity_ecs")
    args = build_parser().parse_args(["inspect", "--view", "memory", "--region", "unity_ecs"])
    run_inspect(args)
    out = json.loads(capsys.readouterr().out)
    assert out["memory"]["total"] == 1
    assert out["memory"]["region_filter"] == "unity_ecs"
