"""模型指纹探针测试:全部离线(FakeBackend),不打真实 API。

覆盖:JSD 数学、答案归一化、usage 采集与对比(match/偷换/注水/元数据漂移)、
行为采集与对比(自比 match / 换模型 mismatch / 后端轮换 split-half)、
存储 roundtrip、MCP 工具的参数校验与 baseline 全链路(monkeypatch 掉 backend)。
"""
from __future__ import annotations

import asyncio

import pytest

from brainregion.probe import fingerprint, storage
from brainregion.providers.base import ModelResponse


class FakeBackend:
    """按 prompt 关键词路由到任务答案池的假后端(确定性:按调用序号轮转)。

    pools: {marker: [答案...]} —— **顺序敏感**:先命中的 marker 先用,
    测试里把更长的关键词(如 "100")排在前面,避免 "1到10" 被 "1到10" 之外误匹配。
    """

    def __init__(
        self,
        pools: dict[str, list[str]] | None = None,
        usage: dict | None = None,
        served_model: str | None = "fake-model",
        system_fingerprint: str | None = None,
        fail: bool = False,
        answer_fn=None,
    ):
        self.pools = pools or {}
        self.usage = usage or {"prompt_tokens": 1111, "completion_tokens": 1, "total_tokens": 1112}
        self.served_model = served_model
        self.system_fingerprint = system_fingerprint
        self.fail = fail
        self.answer_fn = answer_fn
        self.calls: list[str] = []

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
        thinking=None,
    ):
        self.calls.append(user)
        if self.fail:
            return ModelResponse(model=model, error="boom")
        if self.answer_fn is not None:
            content = self.answer_fn(user, len(self.calls) - 1)
        else:
            content = "OK"
            for marker, pool in self.pools.items():
                if marker in user:
                    content = pool[(len(self.calls) - 1) % len(pool)]
                    break
        return ModelResponse(
            model=model,
            content=content,
            usage=dict(self.usage),
            cost_usd=0.0001,
            served_model=self.served_model,
            system_fingerprint=self.system_fingerprint,
        )


# 行为探针的 8 格答案池(marker 顺序即匹配顺序,"100" 必须先于 "1到10"/"1 to 10")
_POOLS_A = {
    "100": ["7", "42", "37", "73", "13", "55"],
    "1到10": ["3", "7", "8", "5"],
    "1 to 10": ["3", "7", "8", "5"],
    "六面骰": ["4", "2", "6"],
    "six-sided": ["4", "2", "6"],
    "硬币": ["正面", "反面"],
    "coin": ["heads", "tails"],
    "颜色": ["蓝", "红", "绿"],
    "color": ["蓝", "红", "绿"],
    "字母": ["q", "z", "k", "x"],
    "letter": ["q", "z", "k", "x"],
    "星期": ["星期三", "星期五"],
    "week": ["wednesday", "friday"],
    "月份": ["七月", "三月"],
    "month": ["july", "march"],
}


def _collect_behavior(backend, cells=8, samples=25, seed=7):
    return asyncio.run(
        fingerprint.run_behavior_probe(
            backend, model="fake-model", cells=cells, samples_per_cell=samples, seed=seed
        )
    )


# ---------------------------------------------------------------------------
# 数学与归一化
# ---------------------------------------------------------------------------


def test_jsd_identical_and_disjoint():
    d = {"a": 0.5, "b": 0.5}
    assert fingerprint.jsd(d, dict(d)) == 0.0
    p, q = {"a": 1.0}, {"b": 1.0}
    assert fingerprint.jsd(p, q) == pytest.approx(1.0)


def test_normalize_answer_variants():
    assert fingerprint.normalize_answer("42.") == "42"
    assert fingerprint.normalize_answer('"Heads"') == "heads"
    assert fingerprint.normalize_answer("  蓝色。") == "蓝色"
    assert fingerprint.normalize_answer("") == "<empty>"
    assert fingerprint.normalize_answer("The number is 42") == "the number is 42"
    assert fingerprint.normalize_answer("正面!") == "正面"


def test_normalize_answer_json_shell():
    # 实测 GLM/Qwen 对短约束问题爱答 JSON 壳(2026-08-14 真实验证发现)
    assert fingerprint.normalize_answer('{"answer":42}') == "42"
    assert fingerprint.normalize_answer('{"answer":"42"}') == "42"
    assert fingerprint.normalize_answer('{"random_num":37}') == "37"
    assert fingerprint.normalize_answer("```json\n{\"answer\": 7}\n```") == "7"
    assert fingerprint.normalize_answer("{\n  \"answer\": \"蓝色\"\n}") == "蓝色"
    # 截断 JSON(无闭合):regex 抠出值
    assert fingerprint.normalize_answer('{"number": 66') == "66"
    # 退化壳归并(空白形态不同但同一现象 → 同一键)
    assert fingerprint.normalize_answer("{  }") == "<unparsed>"
    assert fingerprint.normalize_answer("{") == "<unparsed>"
    assert fingerprint.normalize_answer("{ \t\t\t\t\t") == "<unparsed>"
    # 多行非 JSON:取首行
    assert fingerprint.normalize_answer("heads\ntails") == "heads"


# ---------------------------------------------------------------------------
# usage 指纹
# ---------------------------------------------------------------------------


def test_usage_probe_collects_fields():
    cur = asyncio.run(fingerprint.run_usage_probe(FakeBackend(), model="m"))
    assert cur["ok"] is True
    assert cur["prompt_tokens"] == 1111
    assert cur["served_model"] == "fake-model"


def test_compare_usage_match():
    cur = {"ok": True, "prompt_tokens": 1111, "completion_tokens": 1}
    base = {"ok": True, "prompt_tokens": 1113, "completion_tokens": 1}
    out = fingerprint.compare_usage(cur, base)
    assert out["verdict"] == "match"
    assert out["flags"] == []


def test_compare_usage_tokenizer_swap_mismatch():
    cur = {"ok": True, "prompt_tokens": 1330, "completion_tokens": 1}
    base = {"ok": True, "prompt_tokens": 1000, "completion_tokens": 1}
    out = fingerprint.compare_usage(cur, base)
    assert out["verdict"] == "mismatch"
    assert "suspected_hidden_prompt" in out["flags"]  # 凭空多 330 token


def test_compare_usage_flags():
    cur = {
        "ok": True,
        "prompt_tokens": 1111,
        "completion_tokens": 300,
        "served_model": "other-model",
        "system_fingerprint": "fp_v2",
    }
    base = {
        "ok": True,
        "prompt_tokens": 1111,
        "completion_tokens": 1,
        "served_model": "fake-model",
        "system_fingerprint": "fp_v1",
    }
    out = fingerprint.compare_usage(cur, base)
    assert out["verdict"] == "mismatch"
    for flag in ("served_model_changed", "system_fingerprint_changed", "completion_watering"):
        assert flag in out["flags"]


def test_compare_usage_error_passthrough():
    out = fingerprint.compare_usage({"ok": False, "error": "x"}, {"ok": True})
    assert out["verdict"] == "unknown"


# ---------------------------------------------------------------------------
# behavior 指纹
# ---------------------------------------------------------------------------


def test_behavior_self_match():
    base = _collect_behavior(FakeBackend(pools=_POOLS_A))
    cur = _collect_behavior(FakeBackend(pools=_POOLS_A))
    assert base["ok"] and cur["ok"]
    assert base["failure_rate"] == 0.0
    out = fingerprint.compare_behavior(cur, base)
    assert out["verdict"] == "match"
    assert out["mean_jsd"] <= fingerprint.JSD_MATCH_MAX
    assert out["common_cells"] == 8


def test_behavior_partial_drift_single_cell_flag():
    # 只换 2/8 格:均值被稀释到 <=0.25,但单格强发散必须抬到 suspicious
    pools_b = dict(_POOLS_A)
    pools_b["100"] = ["3", "55", "88", "12", "19", "64"]
    pools_b["颜色"] = ["紫", "橙", "黄"]
    pools_b["color"] = ["紫", "橙", "黄"]
    base = _collect_behavior(FakeBackend(pools=_POOLS_A))
    cur = _collect_behavior(FakeBackend(pools=pools_b))
    out = fingerprint.compare_behavior(cur, base)
    assert "single_cell_divergence" in out["flags"]
    assert out["verdict"] == "suspicious"


def test_behavior_swapped_model_mismatch():
    pools_b = dict(_POOLS_A)
    pools_b["100"] = ["3", "55", "88", "12", "19", "64"]
    pools_b["颜色"] = ["紫", "橙", "黄"]
    pools_b["color"] = ["紫", "橙", "黄"]
    pools_b["字母"] = ["a", "b", "c", "e"]
    pools_b["letter"] = ["a", "b", "c", "e"]
    pools_b["星期"] = ["星期一", "星期日"]
    pools_b["week"] = ["monday", "sunday"]
    pools_b["月份"] = ["一月", "十二月"]
    pools_b["month"] = ["january", "december"]
    base = _collect_behavior(FakeBackend(pools=_POOLS_A))
    cur = _collect_behavior(FakeBackend(pools=pools_b))
    out = fingerprint.compare_behavior(cur, base)
    assert out["verdict"] == "mismatch"
    assert out["mean_jsd"] > fingerprint.JSD_SUSPICIOUS_MAX
    assert out["worst_cell"] is not None
    assert out["worst_cell_divergences"]


def test_behavior_split_half_rotation_flag():
    # 前 10 次答 pool1、之后答 pool2:同格内两半分布完全不交 → split_half=1.0
    def switcher(_prompt, i):
        return "alpha" if i < 10 else "beta"

    backend = FakeBackend(answer_fn=switcher)
    cur = _collect_behavior(backend, cells=1, samples=20)
    assert cur["cells"]["number_1_100"]["split_half_jsd"] > fingerprint.SPLIT_HALF_FLAG
    # 基线构造为与混合后分布一致 → mean_jsd 低,但轮换旗必须把 verdict 抬到 suspicious
    base = {
        "ok": True,
        "cells": {"number_1_100": {"distribution": {"alpha": 0.5, "beta": 0.5}}},
    }
    out = fingerprint.compare_behavior(cur, base)
    assert "backend_rotation_or_cache_collapse" in out["flags"]
    assert out["verdict"] == "suspicious"


def test_behavior_probe_failures_recorded():
    cur = _collect_behavior(FakeBackend(fail=True), cells=2, samples=5)
    assert cur["n_failures"] == 10
    assert cur["failure_rate"] == 1.0


# ---------------------------------------------------------------------------
# 存储
# ---------------------------------------------------------------------------


def test_storage_baseline_roundtrip_and_supersede():
    key = fingerprint.model_key("m", "ep1")
    assert key == "m@ep1"
    storage.save_baseline(key, "usage", {"prompt_tokens": 100})
    loaded = storage.load_active_baseline(key, "usage")
    assert loaded["prompt_tokens"] == 100
    assert loaded["baseline_created_at"]
    storage.save_baseline(key, "usage", {"prompt_tokens": 200})
    assert storage.load_active_baseline(key, "usage")["prompt_tokens"] == 200


def test_storage_runs_append_and_recent():
    key = fingerprint.model_key("m")
    storage.append_run(key, "usage", "compare", "match", 0.0, {"x": 1}, 0.001)
    storage.append_run(key, "behavior", "compare", "mismatch", 0.6, {"x": 2}, 0.01)
    runs = storage.recent_runs(model_key=key)
    assert [r["kind"] for r in runs] == ["behavior", "usage"]
    assert runs[0]["verdict"] == "mismatch"
    assert "details_json" not in runs[0]


def test_inspect_model_health_view():
    from brainregion.inspector import inspect as inspect_facade

    storage.append_run("m1", "usage", "compare", "mismatch", 0.9, {}, 0.001)
    storage.append_run("m1", "behavior", "compare", "match", 0.1, {}, 0.0)
    storage.save_baseline("m1", "usage", {"prompt_tokens": 1})
    out = inspect_facade(view="model_health")
    mh = out["model_health"]
    assert any(b["model_key"] == "m1" and b["kind"] == "usage" for b in mh["active_baselines"])
    assert mh["latest_verdict_by_model"]["m1"]["kind"] == "behavior"
    assert mh["verdict_counts"] == {"match": 1, "mismatch": 1}


def test_effort_kwargs_glm_str_thinking():
    from brainregion.providers.litellm import _effort_kwargs

    assert _effort_kwargs("openai/glm-5.3", effort=None, thinking="low") == {
        "extra_body": {"thinking": {"type": "low"}}
    }
    assert _effort_kwargs("openai/glm-5.2", effort=None, thinking=False) == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }


def test_complete_adaptive_always_thinking():
    fingerprint._ALWAYS_THINKING.clear()
    calls = []

    class ATBackend:
        async def complete(self, **kw):
            calls.append(kw)
            if kw.get("thinking") is False:
                return ModelResponse(
                    model=kw["model"], error="BadRequestError: 该模型始终思考,不支持关闭思考"
                )
            return ModelResponse(model=kw["model"], content="42", served_model="fake")

    async def run():
        r1 = await fingerprint._complete(
            ATBackend(), model="always-think-fake", system="", user="x",
            temperature=1.0, max_tokens=32, thinking=False,
        )
        r2 = await fingerprint._complete(
            ATBackend(), model="always-think-fake", system="", user="x",
            temperature=1.0, max_tokens=32, thinking=False,
        )
        return r1, r2

    r1, r2 = asyncio.run(run())
    assert r1.content == "42" and r2.content == "42"
    # 第一次:失败 + 自适应重试(2 次调用);第二次:缓存命中直接 low(1 次调用)
    assert len(calls) == 3
    assert calls[0]["thinking"] is False
    assert calls[1]["thinking"] == "low" and calls[1]["max_tokens"] == 1024
    assert calls[2]["thinking"] == "low"
    fingerprint._ALWAYS_THINKING.clear()


# ---------------------------------------------------------------------------
# MCP 工具(离线:monkeypatch 掉 LiteLLMBackend)
# ---------------------------------------------------------------------------


def test_tool_validates_params():
    import brainregion.server as srv

    with pytest.raises(ValueError):
        asyncio.run(srv.model_fingerprint_check(model="m", mode="bogus"))
    with pytest.raises(ValueError):
        asyncio.run(srv.model_fingerprint_check(model="m", checks=["nope"]))


def test_tool_baseline_then_compare(monkeypatch):
    import brainregion.server as srv

    fake = FakeBackend(pools=_POOLS_A)
    monkeypatch.setattr(srv, "LiteLLMBackend", lambda **kw: fake)
    # 隔离仓库根的 brain_region_config.json:defaults 打成空(不解析真实 endpoints)
    monkeypatch.setattr(srv._defaults_mod, "get_all", lambda: {})
    out = asyncio.run(
        srv.model_fingerprint_check(model="fake-model", mode="baseline", checks=["usage"])
    )
    assert out["ok"] is True
    assert out["checks"]["usage"]["verdict"] == "baseline_saved"
    # compare 同一假后端 → match;history 里应已有两次运行
    out2 = asyncio.run(
        srv.model_fingerprint_check(model="fake-model", mode="compare", checks=["usage"])
    )
    assert out2["checks"]["usage"]["verdict"] == "match"
    assert len(out2["history"]) >= 2
