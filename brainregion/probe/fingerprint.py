"""模型指纹探针:usage(tokenizer 计数)指纹 + behavior(随机数分布 JSD)指纹。

背景:中转/聚合 API 可能偷换模型(Opus 实际跑 Haiku、GPT 实际跑开源套壳)、注水计量、
注入隐藏 prompt,或上游静默换 snapshot/量化导致"降智"。本模块提供两类主动探针:

- **usage 指纹**(1 次请求):固定 canonical prompt 的 usage.prompt_tokens 由 tokenizer
  决定,同 tokenizer 下确定性一致;o200k / cl100k / GLM / Qwen 对同一文本计数差异通常
  >5%。顺带检查:completion 注水(答一个词却烧几百 token)、隐藏 prompt 注入
  (prompt_tokens 远超实发)、served_model / system_fingerprint 漂移。
- **behavior 指纹**(~200 次小请求):LLM 说不出真随机数——"1-100 随机数/颜色/硬币"这类
  单 token 问题的答案分布每个模型有稳定且独特的偏置(arXiv:2607.10252)。与基线的
  Jensen-Shannon 散度:同人自比≈0.14,跨模型≈0.46。阈值 match<=0.25 /
  suspicious<=0.35 / mismatch>0.35。无需 logprobs,普通 chat API 即可;split-half
  一致性顺带抓后端轮换/缓存作弊。

局限(诚实声明):恶意中转可识别探针并针对性放行真模型——探针 phrasing 随机化只为提高
对抗成本,不承诺防住;偏差≠欺诈,量化/snapshot 更新同样触发漂移。
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: JSD 判定阈值(arXiv:2607.10252 / tosea.ai 实测:自比≈0.14,跨模型≈0.46)。
JSD_MATCH_MAX = 0.25
JSD_SUSPICIOUS_MAX = 0.35

#: split-half JSD 超过此值视为"同一次采集内部自相矛盾"→ 后端轮换/缓存塌缩信号。
SPLIT_HALF_FLAG = 0.60

#: usage 探针的固定 canonical prompt。**内容永不改动**——改了等于换指纹,历史基线全部作废。
CANONICAL_USAGE_PROMPT = """请阅读以下内容后,不要解释,只回复:OK

【中文段】
萧炎抬头望着穹顶之上那缓缓旋转的漩涡,三年时间,足以让一个天才沦为废物,也足以让废物
重新站回云端。斗之气三段的测评石碑前,人群的哄笑声像潮水一样涌来,又像潮水一样退去。
"三十年河东,三十年河西,莫欺少年穷!"——这句话他记得很清楚,标点符号也记得:顿号、
逗号、分号;问号?感叹号!还有省略号……

【English paragraph】
The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.
"How much wood would a woodchuck chuck," she asked, "if a woodchuck could chuck wood?"
Numbers matter too: 3.14159, 2.71828, 1,729 (the taxicab number), 6.02e23, and -40
(the point where Celsius meets Fahrenheit).

【代码段】
```python
def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
print([fib(i) for i in range(10)])  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

【混合符号】
希腊字母:α β γ δ ε ζ η θ λ μ π σ φ ω Ω
特殊符号:© ® ™ § ¶ † ‡ • … ‰ € £ ¥ ¢ ± × ÷ ≠ ≤ ≥ ∞ √ ∂ ∇ ≈ ∝
全角符号:￥ ￥ ① ② ③ 「」『』【】（）——……、。：；？！
Emoji:🔥💡🚀🐛✅❌🤔📦🔍📈🧠⚡🌍

【表格】
| 名称 | 值 | 说明 |
|------|-----|------|
| alpha | 0.05 | 显著性水平 |
| beta | 0.20 | 统计功效 |
| n | 1200 | 样本量 |

以上内容共若干段。现在不要总结、不要解释,只回复:OK"""

#: 行为探针格子(顺序即 cells 切片顺序;phrasings 采样时随机选一个,提高对抗成本)。
PROBE_TASKS: list[dict] = [
    {
        "id": "number_1_100",
        "prompts": [
            "随机给我一个1到100之间的整数,只输出数字本身,不要任何其他文字。",
            "Give me a random integer between 1 and 100. Output only the number itself.",
        ],
    },
    {
        "id": "number_1_10",
        "prompts": [
            "从1到10里随机挑一个整数,只输出数字本身。",
            "Pick a random integer from 1 to 10. Output only the number itself.",
        ],
    },
    {
        "id": "dice_1_6",
        "prompts": [
            "掷一个标准的六面骰子,只输出点数数字。",
            "Roll a standard six-sided die. Output only the number of pips.",
        ],
    },
    {
        "id": "coin",
        "prompts": [
            "抛一枚硬币,只输出'正面'或'反面'。",
            "Flip a coin. Output only 'heads' or 'tails'.",
        ],
    },
    {
        "id": "color",
        "prompts": [
            "随机说出一种颜色,只输出颜色名,一个词。",
            "Name a random color. Output just one word.",
        ],
    },
    {
        "id": "letter_az",
        "prompts": [
            "随机选一个英文字母,只输出这个字母。",
            "Pick a random letter of the English alphabet. Output only the letter.",
        ],
    },
    {
        "id": "weekday",
        "prompts": [
            "随机说一个星期几(星期一到星期日),只输出它。",
            "Name a random day of the week. Output only the day name.",
        ],
    },
    {
        "id": "month",
        "prompts": [
            "随机说一个月份(一到十二月),只输出它。",
            "Name a random month of the year. Output only the month name.",
        ],
    },
]

DISCLAIMER = (
    "偏差≠欺诈:量化版本、官方 snapshot 静默更新、sampling 参数变化都会触发漂移;"
    "结果是信号强度,不是欺诈判定。"
)


def model_key(model: str, endpoint_id: str | None = None) -> str:
    """指纹身份键:同一 model 字符串在不同 endpoint 是不同的被测对象。"""
    return f"{model}@{endpoint_id}" if endpoint_id else model


# ---------------------------------------------------------------------------
# 数学与归一化
# ---------------------------------------------------------------------------


def jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon 散度(base 2,值域 0..1)。p/q 为归一化分布(dict)。"""

    def kl(a: dict[str, float], b: dict[str, float]) -> float:
        total = 0.0
        for k, av in a.items():
            bv = b.get(k, 0.0)
            if av > 0.0 and bv > 0.0:
                total += av * math.log2(av / bv)
        return total

    m: dict[str, float] = {}
    for k in set(p) | set(q):
        m[k] = 0.5 * (p.get(k, 0.0) + q.get(k, 0.0))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


_ANSWER_STRIP = "\"'`*#·。.,!?！？:;：； \t"


def normalize_answer(text: str) -> str:
    """探针答案归一化:首行 → 去包裹符号 → 纯数字保持数字 → 截断 32 → 小写。"""
    s = (text or "").strip()
    if not s:
        return "<empty>"
    s = s.splitlines()[0].strip().strip(_ANSWER_STRIP)
    if not s:
        return "<empty>"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        s = re.fullmatch(r"-?\d+(?:\.\d+)?", s).group(0)
    return s[:32].lower()


def distribution(samples: list[str]) -> dict[str, float]:
    if not samples:
        return {}
    c = Counter(samples)
    return {k: v / len(samples) for k, v in c.items()}


# ---------------------------------------------------------------------------
# 后端调用(兼容纯 ModelBackend 协议实现:不认识 thinking 参数时降级重试)
# ---------------------------------------------------------------------------


async def _complete(backend, **kwargs) -> "object":
    try:
        return await backend.complete(**kwargs)
    except TypeError:
        # FakeBackend/第三方实现可能不接受 thinking 等扩展参数
        kwargs.pop("thinking", None)
        return await backend.complete(**kwargs)


# ---------------------------------------------------------------------------
# usage 指纹(1 次请求)
# ---------------------------------------------------------------------------


async def run_usage_probe(backend, *, model: str, endpoint_id: str | None = None) -> dict:
    """发 canonical prompt,记录 usage 计数 + 元数据。不判分(对比在 compare_usage)。"""
    resp = await _complete(
        backend,
        model=model,
        system="You are a verification endpoint. Follow the user instruction exactly.",
        user=CANONICAL_USAGE_PROMPT,
        temperature=0.0,
        max_tokens=8,
        endpoint_id=endpoint_id,
        thinking=False,
    )
    if not getattr(resp, "ok", False):
        return {"ok": False, "error": getattr(resp, "error", "unknown")}
    usage = getattr(resp, "usage", None) or {}
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    return {
        "ok": True,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "served_model": getattr(resp, "served_model", None),
        "system_fingerprint": getattr(resp, "system_fingerprint", None),
        "content_head": (getattr(resp, "content", "") or "")[:24],
        "cost_usd": getattr(resp, "cost_usd", None),
    }


def compare_usage(cur: dict, base: dict) -> dict:
    """当前 usage 指纹 vs 基线。判定与信号旗分开呈现。"""
    if not cur.get("ok") or not base.get("ok"):
        return {
            "check": "usage",
            "verdict": "unknown",
            "error": cur.get("error") or base.get("error"),
            "note": DISCLAIMER,
        }
    flags: list[str] = []
    b_pt, c_pt = base.get("prompt_tokens"), cur.get("prompt_tokens")
    if b_pt is None or c_pt is None:
        # usage 缺失本身可疑(部分中转会剥掉 usage),但不下结论
        flags.append("usage_missing")
        verdict = "unknown"
        delta_ratio = None
    else:
        delta = c_pt - b_pt
        delta_ratio = delta / b_pt if b_pt else None
        if abs(delta) <= 4 or (delta_ratio is not None and abs(delta_ratio) <= 0.05):
            verdict = "match"
        elif delta_ratio is not None and abs(delta_ratio) <= 0.15:
            verdict = "suspicious"
        else:
            verdict = "mismatch"
        if delta > 50 and (delta_ratio or 0) > 0.05:
            flags.append("suspected_hidden_prompt")  # prompt 侧凭空多出大量 token
    b_ct, c_ct = base.get("completion_tokens"), cur.get("completion_tokens")
    if b_ct is not None and c_ct is not None:
        if c_ct > 64 and max(b_ct, 8) * 4 < c_ct:
            flags.append("completion_watering")  # 只让答 OK 却烧了几百 completion token
    b_sm, c_sm = base.get("served_model"), cur.get("served_model")
    served_changed = bool(b_sm and c_sm and b_sm != c_sm)
    if served_changed:
        flags.append("served_model_changed")
        verdict = "mismatch"
    b_fp, c_fp = base.get("system_fingerprint"), cur.get("system_fingerprint")
    if b_fp and c_fp and b_fp != c_fp:
        flags.append("system_fingerprint_changed")  # snapshot 更新提示,单独不下结论
    return {
        "check": "usage",
        "verdict": verdict,
        "baseline_created_at": base.get("baseline_created_at"),
        "prompt_tokens": {"baseline": b_pt, "current": c_pt, "delta_ratio": delta_ratio},
        "completion_tokens": {"baseline": b_ct, "current": c_ct},
        "served_model": {"baseline": b_sm, "current": c_sm, "changed": served_changed},
        "system_fingerprint": {"baseline": b_fp, "current": c_fp},
        "flags": flags,
        "note": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# behavior 指纹(随机数分布,~cells x samples 次小请求)
# ---------------------------------------------------------------------------


async def run_behavior_probe(
    backend,
    *,
    model: str,
    endpoint_id: str | None = None,
    cells: int = 8,
    samples_per_cell: int = 25,
    seed: int | None = None,
) -> dict:
    """采集行为指纹:每格采样 answers → 经验分布 + split-half 自一致性。"""
    n_cells = max(1, min(int(cells), len(PROBE_TASKS)))
    n_samples = max(5, min(int(samples_per_cell), 100))
    rng = random.Random(seed)
    cell_payloads: dict[str, dict] = {}
    failures: list[dict] = []
    cost = 0.0
    served_model = None
    system_fingerprint = None
    for task in PROBE_TASKS[:n_cells]:
        answers: list[str] = []
        for _ in range(n_samples):
            resp = await _complete(
                backend,
                model=model,
                system="",
                user=rng.choice(task["prompts"]),
                temperature=1.0,
                top_p=1.0,
                max_tokens=16,
                endpoint_id=endpoint_id,
                thinking=False,
            )
            if getattr(resp, "ok", False):
                answers.append(normalize_answer(getattr(resp, "content", "") or ""))
                cost += getattr(resp, "cost_usd", None) or 0.0
                served_model = getattr(resp, "served_model", None) or served_model
                system_fingerprint = (
                    getattr(resp, "system_fingerprint", None) or system_fingerprint
                )
            else:
                failures.append({"task": task["id"], "error": getattr(resp, "error", "")})
        half = len(answers) // 2
        split_half = (
            jsd(distribution(answers[:half]), distribution(answers[half:]))
            if half >= 3
            else None
        )
        cell_payloads[task["id"]] = {
            "n": len(answers),
            "distribution": distribution(answers),
            "split_half_jsd": split_half,
        }
    total_calls = n_cells * n_samples
    return {
        "ok": True,
        "cells": cell_payloads,
        "cells_requested": n_cells,
        "samples_per_cell": n_samples,
        "seed": seed,
        "n_failures": len(failures),
        "failure_rate": round(len(failures) / total_calls, 4) if total_calls else 1.0,
        "failures_sample": failures[:10],
        "cost_usd": round(cost, 6),
        "served_model": served_model,
        "system_fingerprint": system_fingerprint,
    }


def compare_behavior(cur: dict, base: dict) -> dict:
    """当前行为指纹 vs 基线:逐格 JSD → 均值判定 + 轮换/失败率信号。"""
    if not cur.get("ok") or not base.get("ok"):
        return {
            "check": "behavior",
            "verdict": "unknown",
            "error": cur.get("error") or base.get("error"),
            "note": DISCLAIMER,
        }
    cur_cells: dict = cur.get("cells") or {}
    base_cells: dict = base.get("cells") or {}
    common = [c for c in cur_cells if c in base_cells]
    per_cell = {
        c: round(jsd(cur_cells[c]["distribution"], base_cells[c]["distribution"]), 4)
        for c in common
    }
    mean_jsd = round(sum(per_cell.values()) / len(per_cell), 4) if per_cell else None
    flags: list[str] = []
    if mean_jsd is None or len(common) < 4:
        flags.append("low_confidence_few_cells")
        verdict = "unknown" if mean_jsd is None else (
            "suspicious" if mean_jsd > JSD_SUSPICIOUS_MAX else "match"
        )
    else:
        verdict = (
            "match"
            if mean_jsd <= JSD_MATCH_MAX
            else ("suspicious" if mean_jsd <= JSD_SUSPICIOUS_MAX else "mismatch")
        )
    # 均值会被未漂移的格子稀释:任一单格强发散(>0.6)时至少抬到 suspicious,
    # 对应"只换了部分行为"的降智/量化场景。
    max_cell_jsd = max(per_cell.values()) if per_cell else None
    if max_cell_jsd is not None and max_cell_jsd > 0.6:
        flags.append("single_cell_divergence")
        if verdict == "match":
            verdict = "suspicious"
    split_half_max = None
    for c, payload in cur_cells.items():
        sh = payload.get("split_half_jsd")
        if sh is not None and (split_half_max is None or sh > split_half_max):
            split_half_max = sh
    if split_half_max is not None and split_half_max > SPLIT_HALF_FLAG:
        flags.append("backend_rotation_or_cache_collapse")
        if verdict == "match":
            verdict = "suspicious"
    if (cur.get("failure_rate") or 0) > 0.2:
        flags.append("high_probe_failure_rate")
        verdict = "suspicious" if verdict == "match" else verdict
    b_sm, c_sm = base.get("served_model"), cur.get("served_model")
    if b_sm and c_sm and b_sm != c_sm:
        flags.append("served_model_changed")
        verdict = "mismatch"
    # 差异最大的格子 + 该格内分歧最大的答案(人读得懂的"哪里变了")
    worst_cell, divergences = None, []
    if per_cell:
        worst_cell = max(per_cell, key=lambda c: per_cell[c])
        p = cur_cells[worst_cell]["distribution"]
        q = base_cells[worst_cell]["distribution"]
        divergences = [
            {"answer": k, "current": round(p.get(k, 0.0), 3), "baseline": round(q.get(k, 0.0), 3)}
            for k in sorted(set(p) | set(q), key=lambda k: abs(p.get(k, 0.0) - q.get(k, 0.0)), reverse=True)[:5]
        ]
    return {
        "check": "behavior",
        "verdict": verdict,
        "baseline_created_at": base.get("baseline_created_at"),
        "common_cells": len(common),
        "per_cell_jsd": per_cell,
        "mean_jsd": mean_jsd,
        "thresholds": {"match_max": JSD_MATCH_MAX, "suspicious_max": JSD_SUSPICIOUS_MAX},
        "worst_cell": worst_cell,
        "worst_cell_divergences": divergences,
        "split_half_jsd_max": split_half_max,
        "failure_rate": cur.get("failure_rate"),
        "flags": flags,
        "cost_usd": cur.get("cost_usd"),
        "note": DISCLAIMER,
    }
