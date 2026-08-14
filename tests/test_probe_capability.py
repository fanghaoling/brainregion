"""能力基准探针测试:全部离线(FakeBackend),不打真实 API。

关键手法:_perfect_answer 由 build_items 的 check 反推正确答案 → "满分后端",
端到端验证每类判分;全错后端验证 degraded 判定。
"""
from __future__ import annotations

import asyncio
import json

from brainregion.probe import capability
from brainregion.providers.base import ModelResponse


def _perfect_answer(item: dict) -> str:
    c = item["check"]
    k = c["kind"]
    if k == "number":
        v = float(c["value"])
        return str(int(v)) if v.is_integer() else str(v)
    if k == "exact":
        return str(c["value"])
    if k == "regex":
        return "THE CAT SLEEPS QUIETLY"
    if k == "prime3":
        return "【答案】137"
    if k == "word_count_eq":
        return " ".join(["word"] * c["n"])
    if k == "sentence_count_eq":
        return "长城很长。" * c["n"]
    if k == "line_count_eq":
        return "\n".join(["法国", "德国", "意大利", "西班牙", "波兰"][: c["n"]])
    if k == "len_le":
        return "约365天"
    if k == "not_contains":
        return "植物利用光能把水和二氧化碳合成有机物。"
    if k == "json_field_number":
        return json.dumps({c["field"]: c["value"]})
    return "zzz"


class ScriptedBackend:
    """按 build_items 的顺序返回正确/统一错误答案。"""

    def __init__(self, items, correct=True, fail=False):
        self.items, self.correct, self.fail = items, correct, fail
        self.calls = []

    async def complete(self, **kw):
        self.calls.append(kw.get("user"))
        if self.fail:
            return ModelResponse(model=kw.get("model", ""), error="boom")
        idx = min(len(self.calls) - 1, len(self.items) - 1)
        content = _perfect_answer(self.items[idx]) if self.correct else "zzz 胡说"
        return ModelResponse(
            model=kw.get("model", ""),
            content=content,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            cost_usd=0.0001,
            served_model="fake",
        )


# ---------------------------------------------------------------------------
# 判分单元
# ---------------------------------------------------------------------------


def test_grade_number_takes_last_number_in_first_line():
    check = {"kind": "number", "value": 3901}
    assert capability.grade("47 × 83 = 3901", check) is True
    assert capability.grade("3901", check) is True
    assert capability.grade("47 × 83 = 3902", check) is False
    assert capability.grade("x = 11", {"kind": "number", "value": 11}) is True


def test_grade_exact_and_regex():
    assert capability.grade(" 土豆。", {"kind": "exact", "value": "土豆"}) is True
    assert capability.grade("苹果", {"kind": "exact", "value": "土豆"}) is False
    caps = {"kind": "regex", "pattern": r"[A-Z' ,.-]{10,}"}
    assert capability.grade("THE CAT SLEEPS QUIETLY", caps) is True
    assert capability.grade("the cat sleeps quietly", caps) is False


def test_grade_json_shell_extraction():
    # 实测 GLM 对短约束问题爱答 JSON 壳(2026-08-14 真实验证发现):所有 kind 先剥壳
    assert capability.grade('{"answer":"己"}', {"kind": "exact", "value": "己"}) is True
    assert capability.grade('{"answer":"灯塔"}', {"kind": "exact", "value": "灯塔"}) is True
    assert capability.grade('{"answer":"星期日"}', {"kind": "exact", "value": "星期日"}) is True
    assert capability.grade('{"answer":"LTUJR"}', {"kind": "exact", "value": "LTUJR"}) is True
    assert capability.grade('{"answer":"法国\\n德国\\n意大利\\n西班牙\\n葡萄牙"}', {"kind": "line_count_eq", "n": 5}) is True
    assert capability.grade('{"answer":"THE CAT SLEEPS QUIETLY"}', {"kind": "regex", "pattern": r"[A-Z' ,.-]{10,}"}) is True
    # 壳内答案真错 → 仍然判负(剥壳不洗白)
    assert capability.grade('{"answer":"星期二"}', {"kind": "exact", "value": "星期日"}) is False
    # 词数不符 → 真失败不洗白
    assert capability.grade('{"answer":"The sunny sky quickly turned into a storm."}', {"kind": "word_count_eq", "n": 7}) is False


def test_grade_prime3():
    check = {"kind": "prime3"}
    assert capability.grade("【答案】137", check) is True
    assert capability.grade("【答案】100", check) is False  # 非质数
    assert capability.grade("【答案】1000", check) is False  # 非 3 位
    assert capability.grade("137", check) is False  # 缺标记


def test_grade_counts_and_json():
    assert capability.grade("word word word", {"kind": "word_count_eq", "n": 3}) is True
    assert capability.grade("word word", {"kind": "word_count_eq", "n": 3}) is False
    assert capability.grade("长城很长。长城很老。", {"kind": "sentence_count_eq", "n": 2}) is True
    assert capability.grade("法国\n德国\n意大利", {"kind": "line_count_eq", "n": 3}) is True
    assert capability.grade("约365天", {"kind": "len_le", "n": 20, "contains": r"365|一年"}) is True
    assert capability.grade("地球绕太阳公转一圈的时间大约是三百六十五天左右哦", {"kind": "len_le", "n": 20, "contains": r"365|一年"}) is False
    assert capability.grade("植物利用光合成养分。", {"kind": "not_contains", "value": "太阳", "min_len": 5}) is True
    assert capability.grade("植物利用太阳光合成养分。", {"kind": "not_contains", "value": "太阳", "min_len": 5}) is False
    jf = {"kind": "json_field_number", "field": "result", "value": 42}
    assert capability.grade('{"result": 42}', jf) is True
    assert capability.grade('```json\n{"result": 42}\n```', jf) is True
    assert capability.grade('{"result": 43}', jf) is False


# ---------------------------------------------------------------------------
# 模板
# ---------------------------------------------------------------------------


def test_build_items_deterministic_by_seed():
    a = capability.build_items(seed=7)
    b = capability.build_items(seed=7)
    c = capability.build_items(seed=8)
    assert [i["id"] for i in a] == [i["id"] for i in b]
    assert [i["prompt"] for i in a] == [i["prompt"] for i in b]
    assert [i["prompt"] for i in a] != [i["prompt"] for i in c]
    assert len(a) == 30
    cats = {i["category"] for i in a}
    assert cats == {"math", "instruction", "logic", "code_output", "niah"}


def test_niah_marker_once_and_mid_position():
    items = capability.build_items(seed=3)
    for item in items:
        if item["category"] != "niah":
            continue
        word = item["check"]["value"]
        assert item["prompt"].count(word) == 1
        assert len(item["prompt"]) > 2000
        pos = item["prompt"].index(word)
        assert 0.1 < pos / len(item["prompt"]) < 0.85


# ---------------------------------------------------------------------------
# 运行与对比
# ---------------------------------------------------------------------------


def _run(backend, seed=7):
    return asyncio.run(
        capability.run_capability_probe(backend, model="m", seed=seed)
    )


def test_run_perfect_backend_all_pass():
    items = capability.build_items(seed=7)
    out = _run(ScriptedBackend(items, correct=True))
    assert out["ok"] and out["overall_rate"] == 1.0
    assert all(c["rate"] == 1.0 for c in out["categories"].values())


def test_run_broken_backend_zero_and_errors():
    items = capability.build_items(seed=7)
    out = _run(ScriptedBackend(items, correct=False))
    assert out["overall_rate"] == 0.0
    out2 = _run(ScriptedBackend(items, fail=True))
    assert out2["n_errors"] == out2["n_items"]
    assert out2["overall_rate"] == 0.0


def test_truncation_rescue_for_always_thinking():
    """小上限答空、1024 上限给真答案(实测 glm-5.3 形态):救援后计真实分且亮旗。"""
    items = capability.build_items(seed=7)

    class TruncatingBackend:
        def __init__(self, ref):
            self.ref, self.calls = ref, []

        async def complete(self, **kw):
            self.calls.append(kw.get("max_tokens"))
            if kw.get("max_tokens", 0) < 1024:
                return ModelResponse(model="m", content="", served_model="fake")
            return await self.ref.complete(**kw)

    out = _run(TruncatingBackend(ScriptedBackend(items, correct=True)))
    assert out["n_truncation_rescues"] == out["n_items"]
    assert out["overall_rate"] == 1.0  # 救援后全部真实通过
    # 对比时亮旗
    base = _run(ScriptedBackend(items, correct=True))
    cmp = capability.compare_capability(out, base)
    assert "many_truncation_rescues" in cmp["flags"]
    assert cmp["verdict"] == "match"


def test_compare_capability_verdicts():
    items = capability.build_items(seed=7)
    perfect = _run(ScriptedBackend(items, correct=True))
    broken = _run(ScriptedBackend(items, correct=False))
    # 下降 100pp → degraded + 全类目下降旗
    out = capability.compare_capability(broken, perfect)
    assert out["verdict"] == "degraded"
    assert any(f.startswith("category_drop:") for f in out["flags"])
    assert out["failed_items"]
    # 自比 → match
    out2 = capability.compare_capability(perfect, perfect)
    assert out2["verdict"] == "match"
    # 中等下降(手工构造 15pp)→ suspicious
    mid = dict(perfect)
    mid = {**perfect, "overall_rate": perfect["overall_rate"] - 0.15}
    out3 = capability.compare_capability(mid, perfect)
    assert out3["verdict"] == "suspicious"
    # 改善旗
    up = {**perfect, "overall_rate": perfect["overall_rate"] + 0.2}
    out4 = capability.compare_capability(up, perfect)
    assert "improved_vs_baseline" in out4["flags"]


# ---------------------------------------------------------------------------
# MCP 工具接线(离线)
# ---------------------------------------------------------------------------


def test_tool_capability_baseline_then_compare(monkeypatch):
    import brainregion.server as srv

    items = capability.build_items(seed=5)
    # 每次工具调用都新建 backend(fake 的 calls 计数器才能与 item 序号对齐)
    monkeypatch.setattr(
        srv, "LiteLLMBackend", lambda **kw: ScriptedBackend(items, correct=True)
    )
    monkeypatch.setattr(srv._defaults_mod, "get_all", lambda: {})
    out = asyncio.run(
        srv.model_fingerprint_check(model="m", mode="baseline", checks=["capability"], seed=5)
    )
    assert out["checks"]["capability"]["verdict"] == "baseline_saved"
    assert out["checks"]["capability"]["overall_rate"] == 1.0
    out2 = asyncio.run(
        srv.model_fingerprint_check(model="m", mode="compare", checks=["capability"], seed=5)
    )
    assert out2["checks"]["capability"]["verdict"] == "match"
