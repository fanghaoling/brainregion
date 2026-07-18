"""ParseStage：JSON 提取（鲁棒 fallback）+ normalize_finding（放宽校验 + evidence 强制）。"""
from __future__ import annotations

from brainregion.core.stages.parse import extract_json_object, normalize_finding

_GOOD = {
    "dimension": "ecs_perf",
    "severity": "high",
    "title": "t",
    "evidence_quote": "q",
    "location": "l",
    "suggestion": "s",
    "confidence": 0.9,
}


def test_extract_json_block():
    assert extract_json_object('prefix ```json\n{"issues":[]}\n``` suffix') == {"issues": []}


def test_extract_raw():
    assert extract_json_object('{"issues":[]}') == {"issues": []}


def test_extract_prose_plus_json():
    """说明文字 + JSON（无 code block）—— 挽回 glm-5.2 这类输出。"""
    t = '以下是审查结果：\n{"issues":[{"dimension":"x","title":"t","evidence_quote":"q"}]}'
    obj = extract_json_object(t)
    assert obj is not None and "issues" in obj


def test_extract_json_ignores_trailing_prose():
    text = '{"issues":[]}\nNote: the object above is the requested machine-readable result.'
    assert extract_json_object(text) == {"issues": []}


def test_extract_does_not_infer_object_from_natural_language():
    text = 'thought=inspect; tool=read_text; args(path="ranges.py")'
    assert extract_json_object(text) is None


def test_extract_does_not_consume_object_nested_in_top_level_array():
    text = '[{"title":"first"},{"title":"second"}]'
    assert extract_json_object(text) is None


def test_extract_invalid():
    assert extract_json_object("not json at all") is None


def test_normalize_ok():
    nf = normalize_finding(dict(_GOOD))
    assert nf is not None and nf["severity"] == "high"


def test_normalize_fills_defaults():
    """缺 confidence/location/suggestion/case_ref → 补默认（不丢弃）。"""
    f = {"dimension": "x", "title": "t", "evidence_quote": "q"}
    nf = normalize_finding(f)
    assert nf is not None
    assert nf["confidence"] == 0.5 and nf["location"] == "" and nf["case_ref"] is None


def test_normalize_rejects_no_evidence():
    f = dict(_GOOD)
    f["evidence_quote"] = ""
    assert normalize_finding(f) is None


def test_normalize_rejects_no_dimension():
    f = dict(_GOOD)
    del f["dimension"]
    assert normalize_finding(f) is None


def test_normalize_rejects_no_title():
    f = dict(_GOOD)
    del f["title"]
    assert normalize_finding(f) is None


def test_normalize_fixes_bad_severity():
    """非法 severity → 补 medium（放宽，不丢弃）。"""
    f = dict(_GOOD)
    f["severity"] = "critical"
    nf = normalize_finding(f)
    assert nf is not None and nf["severity"] == "medium"


def test_extract_truncated_recovers_complete():
    """glm-5.2 超长输出被截断（外层 } ] 没闭合、``` 也没收尾）：截断修复挽回已写完的 issue。

    半截的最后一条（缺 evidence_quote）会在 ParseStage 的 normalize_finding 被丢，
    但至少完整的 findings 不再整条陪葬。
    """
    t = (
        '```json\n{"issues":['
        '{"dimension":"ecs_perf","title":"完整的","evidence_quote":"q","severity":"high"}, '
        '{"dimension":"planner","severity":"high","title":"半截'
    )
    obj = extract_json_object(t)
    assert obj is not None and "issues" in obj
    assert len(obj["issues"]) == 2  # 修复后两条都回来（半截的留给 normalize 丢）


def test_extract_truncated_no_fence():
    """无围栏、截断的裸 JSON 也能修。"""
    t = '{"issues":[{"dimension":"x","title":"t","evidence_quote":"q"}'
    obj = extract_json_object(t)
    assert obj is not None and len(obj["issues"]) == 1
