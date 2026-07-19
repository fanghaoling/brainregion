"""ParseStage：解析 LLM JSON 输出 → Finding，强制 evidence_quote。

Pipeline 第 5 步。提 ```json 块 / 整段 json.loads → 校验 finding schema → 丢弃无
evidence_quote 的（防幻觉）。schema 不符 best-effort 丢弃并记日志。失败模型跳过。
"""
from __future__ import annotations

import json
import logging
import re

from ..pipeline import PipelineContext
from ..report import Finding
from ..schema import get_schema

logger = logging.getLogger("brainregion.stage.parse")

_FENCE_OPEN_RE = re.compile(r"```(?:json)?\s*")
_schema_cache: dict | None = None


def _schema() -> dict:
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = get_schema("finding")
    return _schema_cache


def _strip_fence(text: str) -> str:
    """去掉 ```json 围栏（闭合或未闭合都处理），返回内侧 JSON 文本。无围栏返回原文本。"""
    m = _FENCE_OPEN_RE.search(text)
    inner = text[m.end():] if m else text
    close = re.search(r"```", inner)
    return inner[: close.start()] if close else inner


def _repair_truncated_json(s: str) -> dict | None:
    """修复被截断的 JSON：补上未闭合的字符串/数组/对象后重试解析。

    glm-5.2 等模型超长输出偶尔被截断（外层 { 和 [ 没闭合，``` 也没收尾），直接 json.loads
    全失败、整条响应被丢。这里按字符扫描跟踪未闭合的 string/`{`/`[`，末尾补对应闭合符，
    挽回已写完的 findings（最后半截那条会被 normalize_finding 因缺 evidence_quote 丢弃）。
    """
    out: list[str] = []
    in_string = False
    escape = False
    stack: list[str] = []
    closers = {"{": "}", "[": "]"}
    for ch in s:
        out.append(ch)
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    if in_string:
        out.append('"')
    out.extend(closers[o] for o in reversed(stack))
    try:
        obj = json.loads("".join(out))
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def extract_json_object_diagnostic(text: str) -> tuple[dict | None, str]:
    """提取 JSON 对象并返回内容无关的恢复模式。"""
    if not text:
        return None, "failed"
    inner = _strip_fence(text)
    brace = inner.find("{")
    if brace < 0:
        return None, "failed"
    bracket = inner.find("[")
    if 0 <= bracket < brace:
        # This helper extracts objects only. Do not consume the first object nested in a
        # top-level array; planner has a separate list-aware parser that needs the whole value.
        return None, "failed"
    cand = inner[brace:]
    # raw_decode 保持 JSON 语法严格，但允许对象后出现 provider 文本回退常见的尾注。
    # 后续调用方仍负责 schema / 工具白名单校验，提取器不把自然语言解释成动作。
    try:
        obj, end = json.JSONDecoder().raw_decode(cand)
        if isinstance(obj, dict):
            mode = "strict_json" if not cand[end:].strip() else "trailing_text"
            return obj, mode
    except Exception:  # noqa: BLE001
        pass
    repaired = _repair_truncated_json(cand)
    if repaired is not None:
        return repaired, "truncated_repair"
    return None, "failed"


def extract_json_object(text: str) -> dict | None:
    """提 JSON 对象，3 级 fallback（挽回模型非纯 JSON / 截断输出）：
    1) 去 ```json 围栏后从首个 { 起 raw_decode，允许完整对象后带简短说明
    2) 仍要求首个值本身是 JSON object，不从自然语言猜测结构或动作
    3) 截断修复（补未闭合括号/字符串）——模型超长输出被截断时
    """
    obj, _mode = extract_json_object_diagnostic(text)
    return obj


_VALID_SEVERITY = {"high", "medium", "low"}


def normalize_finding(f) -> dict | None:
    """补默认 + 校验关键字段。返回补全后的 dict 或 None（丢弃无效）。

    放宽 jsonschema 严校验（best-effort），挽回输出不规范的国产模型 finding；
    但 evidence_quote + dimension + title 仍强制（防幻觉 + 保证可用）。
    """
    if not isinstance(f, dict):
        return None
    out = dict(f)
    out.setdefault("confidence", 0.5)
    out.setdefault("case_ref", None)
    out.setdefault("location", "")
    out.setdefault("suggestion", "")
    if not out.get("evidence_quote") or not out.get("dimension") or not out.get("title"):
        return None
    if out.get("severity") not in _VALID_SEVERITY:
        out["severity"] = "medium"
    try:
        out["confidence"] = float(out.get("confidence", 0.5))
    except Exception:  # noqa: BLE001
        out["confidence"] = 0.5
    return out


class ParseStage:
    name = "parse"

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        counters: dict[str, int] = {}  # v2 label → 已产出 finding 数（生成评审内稳定 id）
        for item in ctx.responses:
            r = item["response"]
            model = item["model"]  # = label（review.py:101 身份标识设计）
            dim = item["dimension"]
            if not r.ok:
                continue
            obj = extract_json_object(r.content)
            if obj is None:
                logger.warning("模型 %s(%s) 输出无法解析为 JSON", model, dim)
                ctx.parse_failed.append(model)  # v1.8 parse 失败可见性（→ failed_models）
                continue
            issues = obj.get("issues") if isinstance(obj.get("issues"), list) else []
            for f in issues:
                nf = normalize_finding(f)
                if nf is None:
                    logger.info(
                        "丢弃无效 finding（缺 evidence/dimension/title）: %s/%s",
                        model,
                        str(f.get("title", ""))[:40] if isinstance(f, dict) else "?",
                    )
                    continue
                seq = counters.get(model, 0)
                counters[model] = seq + 1
                ctx.findings.append(
                    Finding(
                        id=f"{model}-{seq}",  # v2 评审内稳定 id，mark_finding 引用
                        model=model,
                        # v2 修复：强制 reviewer dim（稳定）。LLM 常把 dimension 自由填成子维度
                        # （Migration/Rollback/Testing/...），不稳定 → (label, dim) reliability key
                        # 永不命中。细粒度分类信息已在 title/location，dimension 统一为 reviewer。
                        dimension=dim,
                        severity=nf["severity"],
                        title=nf["title"],
                        evidence_quote=nf["evidence_quote"],
                        location=nf.get("location", ""),
                        suggestion=nf.get("suggestion", ""),
                        confidence=nf["confidence"],
                        case_ref=nf.get("case_ref"),
                    )
                )
        return ctx
