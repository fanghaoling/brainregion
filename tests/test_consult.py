from __future__ import annotations

import json

import pytest

from brainregion import reviews_db
from brainregion.core.consult import ConsultEngine, ConsultRequest
from brainregion.core.consult.guard import prepare_request
from brainregion.core.consult.parse import parse_advice
from brainregion.core.consult.prompt import render_consult_prompt
from brainregion.core.consult.report import ConsultAdvice, ConsultReport
from brainregion.core.consultants import CONSULTANTS_DIR
from brainregion.core.consultants.loader import load_consultant
from brainregion.providers.base import ModelResponse


class _ConsultBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(
        self,
        *,
        model,
        system,
        user,
        temperature=0.3,
        top_p=0.95,
        max_tokens=4096,
        effort=None,
        endpoint_id=None,
    ):
        self.calls.append(
            {
                "model": model,
                "system": system,
                "user": user,
                "effort": effort,
                "endpoint_id": endpoint_id,
            }
        )
        if model == "bad-model":
            return ModelResponse(model=model, error="401 invalid api key")
        content = {
            "summary": f"{model} suggests isolating the failing boundary.",
            "likely_causes": ["configuration drift"],
            "next_experiments": ["run the smallest reproduction"],
            "solution_options": ["add a guard and a fake-backend test"],
            "risks": ["external calls may leak secrets"],
            "recommended_plan": ["start with consult_problem MVP"],
            "confidence": 0.8,
        }
        return ModelResponse(model=model, content=json.dumps(content), usage={"total_tokens": 42}, cost_usd=0.002)


def _panel(model: str) -> dict:
    return {"label": model, "model": model, "endpoint_id": None}


def test_parse_advice_from_fenced_json():
    content = """```json
    {"summary":"s","likely_causes":["c"],"confidence":0.7}
    ```"""
    advice = parse_advice(content, model="m", consultant="debugger", advice_id="m/debugger-0")
    assert advice is not None
    assert advice.summary == "s"
    assert advice.likely_causes == ["c"]
    assert advice.confidence == 0.7


def test_guard_redacts_and_truncates():
    req = ConsultRequest(problem="debug this", logs="OPENAI_API_KEY=sk-abcdef1234567890\n" + "x" * 200)
    sanitized, meta = prepare_request(req, max_input_chars=80)
    assert "sk-abcdef" not in sanitized.logs
    assert "[REDACTED]" in sanitized.logs
    assert meta["redacted_items"] >= 1
    assert "logs" in meta["truncated_fields"]


def test_prompt_includes_semantic_fields():
    role = load_consultant("debugger", CONSULTANTS_DIR)
    req = ConsultRequest(
        problem="flaky tests",
        current_attempt="restarted the MCP server",
        why_stuck="tool schema still looks stale",
        question="what should I check next?",
        desired_output="one likely cause and one experiment",
    )
    _, user = render_consult_prompt(req, role)
    assert "## 当前尝试" in user
    assert "restarted the MCP server" in user
    assert "## 卡住原因" in user
    assert "## 想请外援回答的问题" in user
    assert "## 期望输出" in user


def test_consult_mode_resolution():
    from brainregion.server import _resolve_consultants

    defaults = {"consult_consultants": ["debugger", "critic"], "consult_mode": None}
    assert _resolve_consultants(None, "challenge", defaults) == (["challenge", "critic"], "challenge")
    assert _resolve_consultants(["debugger"], "challenge", defaults) == (["debugger"], "challenge")
    assert _resolve_consultants(None, None, defaults) == (["debugger", "critic"], None)


@pytest.mark.asyncio
async def test_consult_problem_routing_metadata(monkeypatch):
    from brainregion import server

    class _FakeEngine:
        async def consult(self, *args, **kwargs):
            return ConsultReport(
                consultation_id="consult-route",
                summary="ok",
                confidence=0.5,
                individual=[
                    ConsultAdvice(
                        id="consult-route-0",
                        model="fast-model",
                        consultant="debugger",
                        summary="ok",
                        confidence=0.5,
                    )
                ],
                usage={"total_tokens": 1, "cost_usd": 0.0},
                budget={"jobs_run": 1, "jobs_total": 1},
                guard={},
            )

    monkeypatch.setattr(
        server._defaults_mod,
        "apply",
        lambda **kwargs: {
            "endpoints": {},
            "consult_panel": ["fast-model"],
            "panel": ["slow-model"],
            "consult_consultants": ["debugger"],
            "consult_max_input_chars": 1000,
            "consult_max_cost_usd": 0.01,
            "effort": None,
        },
    )
    monkeypatch.setattr(server, "_build_consult_engine", lambda dd: _FakeEngine())

    result = await server.consult_problem(problem="x")
    assert result["consultation_id"] == "consult-route"
    assert result["panel"] == ["fast-model"]
    assert result["routing"]["panel_source"] == "consult_panel"
    assert result["routing"]["consultants_source"] == "consult_consultants"
    assert result["routing"]["model_routes"][0]["route_type"] == "official_litellm"
    assert result["routing"]["route_warnings"] == []

    challenge = await server.consult_problem(problem="x", mode="challenge")
    assert challenge["mode"] == "challenge"
    assert challenge["consultants"] == ["challenge", "critic"]
    assert challenge["routing"]["consultants_source"] == "mode"


@pytest.mark.asyncio
async def test_consult_problem_only_resolves_selected_endpoint_credentials(monkeypatch):
    from brainregion import server

    captured: dict = {}

    class _FakeEngine:
        async def consult(self, *args, **kwargs):
            return ConsultReport(
                consultation_id="consult-selected-endpoint",
                summary="ok",
                confidence=0.5,
                individual=[],
                usage={"total_tokens": 0, "cost_usd": 0.0},
                budget={"jobs_run": 0, "jobs_total": 0},
                guard={},
            )

    monkeypatch.setenv("SELECTED_RELAY_KEY", "selected-secret")
    monkeypatch.delenv("UNUSED_RELAY_KEY", raising=False)
    monkeypatch.setattr(
        server._defaults_mod,
        "apply",
        lambda **kwargs: {
            "endpoints": {
                "selected": {
                    "provider": "anthropic",
                    "base_url": "https://selected.example",
                    "api_key_env": "SELECTED_RELAY_KEY",
                    "models": ["fast-model"],
                },
                "unused": {
                    "provider": "openai",
                    "base_url": "https://unused.example/v1",
                    "api_key_env": "UNUSED_RELAY_KEY",
                    "models": ["offline-model"],
                },
            },
            "consult_panel": ["selected/fast-model"],
            "consult_consultants": ["debugger"],
            "consult_max_input_chars": 1000,
            "consult_max_cost_usd": 0.01,
            "effort": None,
        },
    )

    def build_engine(defaults):
        captured.update(defaults["endpoints"])
        server._resolve_endpoints(defaults["endpoints"])
        return _FakeEngine()

    monkeypatch.setattr(server, "_build_consult_engine", build_engine)

    result = await server.consult_problem(problem="synthetic public test")

    assert result["consultation_id"] == "consult-selected-endpoint"
    assert set(captured) == {"selected"}


def test_record_consultation_and_mark_advice():
    from brainregion.server import mark_advice

    report = {
        "consultation_id": "consult-test",
        "routing": {"panel_source": "consult_panel"},
        "panel": ["model-a"],
        "consultants": ["debugger"],
        "mode": "debugging",
        "usage": {"total_tokens": 10},
        "budget": {"jobs_run": 1},
        "guard": {"sent_chars": 20},
        "individual": [
            {
                "id": "consult-test-0",
                "model": "model-a",
                "consultant": "debugger",
                "summary": "SECRET_PROMPT_CONTENT should not be stored",
                "confidence": 0.7,
            }
        ],
    }
    reviews_db.record_consultation(report)
    advice = reviews_db.lookup_advice("consult-test-0")
    assert advice is not None
    assert advice["consultation_id"] == "consult-test"
    assert advice["model"] == "model-a"

    res = mark_advice(
        advice_id="consult-test-0",
        consultation_id="consult-test",
        decision="accepted",
        reason="helped",
        outcome="fixed",
    )
    assert res["ok"] is True
    assert res["consultant"] == "debugger"

    rows = reviews_db._connect().execute("SELECT * FROM advice_feedback").fetchall()
    assert len(rows) == 1
    assert rows[0]["decision"] == "accepted"
    dumped = "\n".join(str(dict(row)) for row in reviews_db._connect().execute("SELECT * FROM consultation_advice"))
    assert "SECRET_PROMPT_CONTENT" not in dumped


def test_mark_advice_rejects_missing_or_mismatched_advice():
    from brainregion.server import mark_advice

    with pytest.raises(ValueError, match="找不到 advice_id"):
        mark_advice(advice_id="missing-0", decision="accepted")

    reviews_db.record_consultation(
        {
            "consultation_id": "consult-a",
            "individual": [
                {"id": "consult-a-0", "model": "model-a", "consultant": "debugger", "summary": "s"}
            ],
        }
    )
    with pytest.raises(ValueError, match="不是"):
        mark_advice(advice_id="consult-a-0", consultation_id="other", decision="accepted")
    with pytest.raises(ValueError, match="decision"):
        mark_advice(advice_id="consult-a-0", decision="maybe")


@pytest.mark.asyncio
async def test_consult_engine_success_and_failed_model():
    backend = _ConsultBackend()
    engine = ConsultEngine(backend=backend, consultants_dir=CONSULTANTS_DIR)
    report = await engine.consult(
        ConsultRequest(problem="tests are flaky", logs="AssertionError"),
        panel=[_panel("good-model"), _panel("bad-model")],
        consultants=["debugger"],
        max_cost_usd=None,
        effort="low",
    )
    got = report.to_dict()
    assert got["summary"].startswith("good-model suggests")
    assert got["likely_causes"] == ["configuration drift"]
    assert got["failed_models"][0]["model"] == "bad-model"
    assert got["failed_models"][0]["type"] == "auth_error"
    assert got["usage"]["total_tokens"] == 42
    assert backend.calls[0]["effort"] == "low"


@pytest.mark.asyncio
async def test_consult_engine_rejects_unknown_consultant():
    engine = ConsultEngine(backend=_ConsultBackend(), consultants_dir=CONSULTANTS_DIR)
    with pytest.raises(ValueError, match="未知 consultant"):
        await engine.consult(
            ConsultRequest(problem="x"),
            panel=[_panel("good-model")],
            consultants=["does_not_exist"],
        )


@pytest.mark.asyncio
async def test_consult_engine_budget_trims_jobs(monkeypatch):
    from brainregion.core.stages import review as review_stage

    monkeypatch.setattr(review_stage, "_estimate_job_cost", lambda job: 0.02)
    backend = _ConsultBackend()
    engine = ConsultEngine(backend=backend, consultants_dir=CONSULTANTS_DIR)
    report = await engine.consult(
        ConsultRequest(problem="x"),
        panel=[_panel("a"), _panel("b")],
        consultants=["debugger"],
        max_cost_usd=0.03,
    )
    assert len(backend.calls) == 1
    assert report.budget["jobs_total"] == 2
    assert report.budget["jobs_run"] == 1
    assert report.budget["exhausted"] is True


class _RawBackend:
    """Returns fixed raw content (no valid JSON object) to exercise parse_error."""

    def __init__(self, content: str) -> None:
        self.content = content

    async def complete(self, *, model, system, user, temperature=0.3, top_p=0.95,
                       max_tokens=4096, effort=None, endpoint_id=None):
        return ModelResponse(model=model, content=self.content, usage={"total_tokens": 5})


@pytest.mark.asyncio
async def test_consult_parse_error_no_json_object_redacted():
    # ISS-007：reasoning 模型耗尽 max_tokens、整段没吐 `{` 的 parse_error。
    backend = _RawBackend("思考后认为问题在 api_key=sk-secret1234567890")
    engine = ConsultEngine(backend=backend, consultants_dir=CONSULTANTS_DIR)
    report = await engine.consult(
        ConsultRequest(problem="stuck"),
        panel=[_panel("reasoning-model")],
        consultants=["challenge"],
        max_cost_usd=None,
    )
    failed = report.to_dict()["failed_models"][0]
    assert failed["type"] == "parse_error"
    assert failed["diagnostics"]["has_object_start"] is False
    assert "max_tokens" in failed["hint"]
    assert "sk-secret" not in failed["diagnostics"]["output_excerpt"]
    assert "[REDACTED]" in failed["diagnostics"]["output_excerpt"]


@pytest.mark.asyncio
async def test_consult_parse_error_malformed_object_hint():
    # 含 `{` 但无法修复 → has_object_start=True 分支。
    backend = _RawBackend("{not valid json")
    engine = ConsultEngine(backend=backend, consultants_dir=CONSULTANTS_DIR)
    report = await engine.consult(
        ConsultRequest(problem="stuck"),
        panel=[_panel("trunc-model")],
        consultants=["debugger"],
        max_cost_usd=None,
    )
    failed = report.to_dict()["failed_models"][0]
    assert failed["type"] == "parse_error"
    assert failed["diagnostics"]["has_object_start"] is True
    assert "解析失败" in failed["hint"]
