"""v1.5 成本上限（max_cost_usd 裁剪）+ 思考强度（effort 透传 + provider 映射）。"""
from __future__ import annotations

import asyncio

from brainregion.core import ReviewDocument
from brainregion.core.pipeline import PipelineContext
from brainregion.core.stages.review import ReviewStage, select_jobs_within_budget
from brainregion.providers.base import ModelResponse


def _job(model: str, dim: str = "ecs_perf") -> dict:
    return {
        "model": model, "label": model, "endpoint_id": None,
        "dimension": dim, "system": "s", "user": "u",
        "temperature": 0.3, "top_p": 0.9, "max_tokens": 4096,
    }


class _FakeBackend:
    """记录 effort 调用，返回空 issues。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(self, *, model, system, user, temperature=0.3, top_p=0.9, max_tokens=4096, effort=None, endpoint_id=None):
        self.calls.append({"model": model, "effort": effort})
        return ModelResponse(model=model, content='{"issues":[]}', usage={}, cost_usd=0.01)


# ===== provider 映射 =====

def test_effort_kwargs_mapping():
    from brainregion.providers.litellm import _effort_kwargs

    # Claude: output_config + adaptive thinking
    assert _effort_kwargs("claude-opus-4-8", "high") == {
        "thinking": {"type": "adaptive"},
        "extra_body": {"output_config": {"effort": "high"}},
    }
    assert _effort_kwargs("claude-sonnet-5", None, thinking=True) == {
        "thinking": {"type": "adaptive"},
    }
    assert _effort_kwargs("claude-sonnet-5", "high", thinking=False) == {}
    # OpenAI o 系列: reasoning_effort
    assert _effort_kwargs("o3-mini", "high") == {"reasoning_effort": "high"}
    assert _effort_kwargs("o4-mini", "low") == {"reasoning_effort": "low"}
    assert _effort_kwargs("gpt-5.5", "medium", thinking=True) == {
        "reasoning_effort": "medium"
    }
    assert _effort_kwargs("openai/gpt-5.4-mini", None, thinking=False) == {
        "reasoning_effort": "none"
    }
    assert _effort_kwargs("gpt-5.5", "high", thinking=False) == {
        "reasoning_effort": "none"
    }
    assert _effort_kwargs("gpt-5.5", None, thinking=None) == {}
    # 非推理模型: 不传（litellm 不会报错，effort 无效）
    assert _effort_kwargs("gpt-4o", "high") == {}
    assert _effort_kwargs("zai/glm-5.2", "high") == {}
    assert _effort_kwargs("deepseek/deepseek-v4-pro", "high") == {}
    # effort=None: 不传
    assert _effort_kwargs("claude-opus-4-8", None) == {}


# ===== 预算裁剪（纯函数）=====

def test_budget_trim_keeps_prefix(monkeypatch):
    """max_cost_usd 限住：按 panel 顺序保留前缀，超的裁掉，exhausted=True。"""
    from brainregion.core.stages import review as rev

    costs = iter([0.05, 0.05, 0.05, 0.05])
    monkeypatch.setattr(rev, "_estimate_job_cost", lambda job: next(costs))
    jobs = [_job("a"), _job("b"), _job("c"), _job("d")]
    selected, total, exhausted = select_jobs_within_budget(jobs, 0.12)
    assert len(selected) == 2  # 0.05+0.05=0.10 ≤0.12；第三个到 0.15>0.12 裁
    assert total == 0.10
    assert exhausted is True


def test_budget_trim_all_fit(monkeypatch):
    """预算够 = 全跑，exhausted=False。"""
    from brainregion.core.stages import review as rev

    monkeypatch.setattr(rev, "_estimate_job_cost", lambda job: 0.01)
    jobs = [_job("a"), _job("b"), _job("c")]
    selected, total, exhausted = select_jobs_within_budget(jobs, 1.0)
    assert len(selected) == 3 and exhausted is False


# ===== ReviewStage 集成（effort 透传 + 预算裁剪端到端）=====

def test_effort_passthrough_to_backend():
    backend = _FakeBackend()
    ctx = PipelineContext(
        document=ReviewDocument.markdown("x"), adapter=None, backend=backend, knowledge=None
    )
    ctx.prompts = [_job("a"), _job("b")]
    ctx.effort = "high"
    asyncio.run(ReviewStage().process(ctx))
    assert len(backend.calls) == 2
    assert all(c["effort"] == "high" for c in backend.calls)


def test_budget_runs_subset(monkeypatch):
    from brainregion.core.stages import review as rev

    monkeypatch.setattr(rev, "_estimate_job_cost", lambda job: 0.05)
    backend = _FakeBackend()
    ctx = PipelineContext(
        document=ReviewDocument.markdown("x"), adapter=None, backend=backend, knowledge=None
    )
    ctx.prompts = [_job("a"), _job("b"), _job("c"), _job("d")]
    ctx.max_cost_usd = 0.12
    asyncio.run(ReviewStage().process(ctx))
    assert len(backend.calls) == 2  # 裁到 2 个
    assert ctx.jobs_run == 2 and ctx.jobs_total == 4 and ctx.budget_exhausted is True


def test_no_budget_runs_all():
    backend = _FakeBackend()
    ctx = PipelineContext(
        document=ReviewDocument.markdown("x"), adapter=None, backend=backend, knowledge=None
    )
    ctx.prompts = [_job("a"), _job("b"), _job("c")]
    ctx.max_cost_usd = None  # 默认无上限
    asyncio.run(ReviewStage().process(ctx))
    assert len(backend.calls) == 3
    assert ctx.jobs_run == 3 and ctx.budget_exhausted is False


# ===== 单价表（ISS-003：旗舰模型必须有价，否则名义单价让预算护栏严重低估）=====

def test_estimate_job_cost_uses_real_price_for_flagship():
    from brainregion.core.stages.review import _estimate_job_cost, _NOMINAL_COST_USD

    # gpt-5.5 在价表里 → 用真实单价，远高于名义单价（防 ISS-003 的 ~21× 低估）
    cost = _estimate_job_cost(_job("gpt-5.5"))
    assert cost > _NOMINAL_COST_USD * 5
    # 端点前缀形式 modelbridge_openai/gpt-5.5 也能解析到同一单价
    prefixed = {**_job("modelbridge_openai/gpt-5.5"), "model": "modelbridge_openai/gpt-5.5"}
    assert _estimate_job_cost(prefixed) == cost


def test_estimate_job_cost_nominal_for_unknown_cheap_model():
    from brainregion.core.stages.review import _estimate_job_cost, _NOMINAL_COST_USD

    # glm 等不在价表的便宜模型 → 名义单价（保守，略高于实际）
    assert _estimate_job_cost(_job("zai/glm-5.2")) == _NOMINAL_COST_USD
