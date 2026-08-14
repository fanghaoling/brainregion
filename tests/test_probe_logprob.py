"""logprob LT 档测试:全部离线(FakeBackend),不打真实 API。

覆盖:提取助手(SimpleNamespace 模拟 litellm 响应)、采样运行器(支持/不支持端点)、
S 统计量与置换检验(同分布 match / 偏移 mismatch / 可复现)、工具接线。
"""
from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

from brainregion.probe import logprob_lt
from brainregion.providers.base import ModelResponse
from brainregion.providers.litellm import _extract_first_token_logprobs


def _lp_resp(top: list[dict]) -> ModelResponse:
    return ModelResponse(
        model="m", content="4", usage={}, cost_usd=0.00001,
        served_model="fake",
        first_token_logprobs={"sampled": top[0], "top": top},
    )


class LTBackend:
    """按 dist 生成带微小抖动的 top-k logprob;support=False 模拟不支持端点。"""

    def __init__(self, dist: dict[str, float], jitter=0.01, support=True, seed=1):
        self.dist, self.jitter, self.support = dist, jitter, support
        self.rng = random.Random(seed)

    async def complete(self, **kw):
        if not self.support:
            return ModelResponse(model="m", content="4", usage={}, served_model="fake")
        top = [
            {"token": t, "logprob": round(lp + self.rng.uniform(-self.jitter, self.jitter), 5)}
            for t, lp in self.dist.items()
        ]
        return _lp_resp(top)


DIST_A = {"4": -0.1, "four": -2.8, "Five": -5.4, "IV": -6.1, "zero": -7.0}
DIST_B = {"4": -1.9, "four": -1.2, "Five": -3.3, "IV": -4.0, "zero": -7.2}  # 量化/微调级偏移


def _run(backend, n=10):
    return asyncio.run(
        logprob_lt.run_logprob_probe(backend, model="m", n_samples=n)
    )


def test_extract_first_token_logprobs():
    entry = SimpleNamespace(
        token="4", logprob=-0.1,
        top_logprobs=[
            SimpleNamespace(token="4", logprob=-0.1),
            SimpleNamespace(token="four", logprob=-2.8),
        ],
    )
    resp = SimpleNamespace(
        choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[entry]))]
    )
    out = _extract_first_token_logprobs(resp)
    assert out["sampled"] == {"token": "4", "logprob": -0.1}
    assert out["top"] == [{"token": "4", "logprob": -0.1}, {"token": "four", "logprob": -2.8}]
    # 无 logprobs(端点不支持/被 drop_params 丢弃)→ None
    assert _extract_first_token_logprobs(SimpleNamespace(choices=[SimpleNamespace(logprobs=None)])) is None
    assert _extract_first_token_logprobs(SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[]))])) is None


def test_run_probe_ok_and_unsupported():
    out = _run(LTBackend(DIST_A))
    assert out["ok"] is True
    assert out["n"] == 10
    assert len(out["samples"][0]) == len(DIST_A)
    out2 = _run(LTBackend(DIST_A, support=False))
    assert out2["ok"] is False
    assert out2["error"] == "endpoint_did_not_return_logprobs"
    assert out2["hint"]


def test_compare_same_distribution_match():
    base = _run(LTBackend(DIST_A, seed=1))
    cur = _run(LTBackend(DIST_A, seed=2))
    out = logprob_lt.compare_logprob(cur, base)
    assert out["verdict"] == "match"
    assert out["p_value"] > logprob_lt.P_SUSPICIOUS
    assert out["statistic_S"] < 0.05


def test_compare_shifted_distribution_mismatch():
    base = _run(LTBackend(DIST_A, seed=1))
    cur = _run(LTBackend(DIST_B, seed=3))
    out = logprob_lt.compare_logprob(cur, base)
    assert out["verdict"] == "mismatch"
    assert out["p_value"] <= logprob_lt.P_MISMATCH
    assert out["statistic_S"] > 0.5


def test_compare_deterministic_p():
    base = _run(LTBackend(DIST_A, seed=1))
    cur = _run(LTBackend(DIST_B, seed=3))
    p1 = logprob_lt.compare_logprob(cur, base)["p_value"]
    p2 = logprob_lt.compare_logprob(cur, base)["p_value"]
    assert p1 == p2


def test_compare_unsupported_and_served_model_flag():
    out = logprob_lt.compare_logprob(
        _run(LTBackend(DIST_A, support=False)), _run(LTBackend(DIST_A))
    )
    assert out["verdict"] == "unknown"
    base = _run(LTBackend(DIST_A))
    cur = dict(_run(LTBackend(DIST_A, seed=9)))
    cur["served_model"] = "other"
    out2 = logprob_lt.compare_logprob(cur, base)
    assert "served_model_changed" in out2["flags"]
    assert out2["verdict"] == "mismatch"


def test_tool_logprob_baseline_then_compare(monkeypatch):
    import brainregion.server as srv

    monkeypatch.setattr(srv, "LiteLLMBackend", lambda **kw: LTBackend(DIST_A))
    monkeypatch.setattr(srv._defaults_mod, "get_all", lambda: {})
    out = asyncio.run(
        srv.model_fingerprint_check(model="m", mode="baseline", checks=["logprob"])
    )
    assert out["checks"]["logprob"]["verdict"] == "baseline_saved"
    out2 = asyncio.run(
        srv.model_fingerprint_check(model="m", mode="compare", checks=["logprob"])
    )
    assert out2["checks"]["logprob"]["verdict"] == "match"
