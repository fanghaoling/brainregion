"""logprob 置换检验档(LT, Log-probability Tracking, arXiv:2512.03816)。

**最灵敏的同模型变更检测**:对固定短 prompt 只生成 1 个 token,取 top-k logprob 当作
分布采样;两侧各 N 次采样,对每个 token 算两侧平均 logprob 的平均绝对距离 S,再用置换
检验(pool 2N 份随机对分 B 次)算 p 值。论文灵敏度:能检出**一次微调**、2^-10 级剪枝;
每次测试只要 ~50 token(比 MMLU 基线便宜 ~1000 倍)。

限制(论文实测 + 我们的现实约束):
- 端点必须支持 chat_completions 的 top_logprobs(OpenRouter 上仅 ~23% 支持);
  drop_params 下不支持的端点会静默丢参 → 本模块识别为 unsupported,不算失败。
- responses api_mode 不支持(结构不同,未接)。
- 无法区分权重变更与推理 infra 变更;可被 provider 缓存/针对性响应欺骗。
"""
from __future__ import annotations

import random

from .fingerprint import _complete

#: 固定探针 prompt(永不改动——改了等于换基线)。首 token 分布要有信息量。
LT_PROMPT = "Q: What is 2 + 2?\nA:"

#: 置换次数(统计精度 ~1/B;B=1000 时 p 分辨率 0.001,足够 0.01 显著阈值)
N_PERMUTATIONS = 1000

#: 显著阈值:p<=0.01 判 mismatch(分布变了),p<=0.05 判 suspicious
P_MISMATCH = 0.01
P_SUSPICIOUS = 0.05

DISCLAIMER = (
    "LT 抓'同模型分布级变更'(微调/量化/snapshot);无法区分权重变更与推理 infra 差异,"
    "也可能被 provider 缓存欺骗。与 behavior/usage 档交叉印证,单独不结论。"
)


async def run_logprob_probe(
    backend,
    *,
    model: str,
    endpoint_id: str | None = None,
    n_samples: int = 10,
    top_k: int = 20,
    prompt: str | None = None,
) -> dict:
    """采 N 份首 token top-k logprob(temperature=1 取原始分布,不进采样缩放)。"""
    n = max(3, min(int(n_samples), 50))
    samples: list[list[dict]] = []
    n_unsupported = 0
    error: str | None = None
    cost = 0.0
    served_model = None
    for _ in range(n):
        resp = await _complete(
            backend,
            model=model,
            system="",
            user=prompt or LT_PROMPT,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1,
            endpoint_id=endpoint_id,
            thinking=False,
            logprobs_top_k=top_k,
        )
        if not getattr(resp, "ok", False):
            error = getattr(resp, "error", "unknown")
            continue
        cost += getattr(resp, "cost_usd", None) or 0.0
        served_model = getattr(resp, "served_model", None) or served_model
        lp = getattr(resp, "first_token_logprobs", None)
        if lp and lp.get("top"):
            samples.append(lp["top"])
        else:
            n_unsupported += 1
    if len(samples) < max(3, n // 2):
        return {
            "ok": False,
            "error": error or "endpoint_did_not_return_logprobs",
            "n_samples_ok": len(samples),
            "n_unsupported": n_unsupported,
            "hint": "端点/chat模式不支持 top_logprobs(OpenRouter 实测仅 ~23% 端点支持);responses api_mode 未接",
        }
    return {
        "ok": True,
        "samples": samples,
        "n": len(samples),
        "top_k": top_k,
        "cost_usd": round(cost, 6),
        "served_model": served_model,
        "n_unsupported": n_unsupported,
    }


# ---------------------------------------------------------------------------
# 统计:S = 两侧逐 token 平均 logprob 的平均绝对距离;置换检验出 p
# ---------------------------------------------------------------------------


def _sample_maps(samples: list[list[dict]]) -> list[dict[str, float]]:
    """每份采样 → {token: logprob};并记录该份的最小 logprob(缺失 token 的保守下界)。"""
    out = []
    for top in samples:
        m = {t["token"]: float(t["logprob"]) for t in top}
        m["<min>"] = min(m.values()) if m else 0.0
        out.append(m)
    return out


def _mean_vector(maps: list[dict[str, float]], tokens: list[str]) -> dict[str, float]:
    acc = {t: 0.0 for t in tokens}
    for m in maps:
        floor = m["<min>"]
        for t in tokens:
            acc[t] += m.get(t, floor)
    return {t: v / len(maps) for t, v in acc.items()}


def _statistic(a_maps: list[dict[str, float]], b_maps: list[dict[str, float]]) -> float:
    tokens = sorted({t for m in a_maps + b_maps for t in m if t != "<min>"})
    if not tokens:
        return 0.0
    va, vb = _mean_vector(a_maps, tokens), _mean_vector(b_maps, tokens)
    return sum(abs(va[t] - vb[t]) for t in tokens) / len(tokens)


def compare_logprob(cur: dict, base: dict) -> dict:
    """LT 对比:S 统计量 + 置换检验 p 值 → match/suspicious/mismatch。"""
    if not cur.get("ok") or not base.get("ok"):
        return {
            "check": "logprob",
            "verdict": "unknown",
            "error": cur.get("error") or base.get("error"),
            "hint": (cur.get("hint") or base.get("hint")) if (cur.get("hint") or base.get("hint")) else None,
            "note": DISCLAIMER,
        }
    a_maps = _sample_maps(base["samples"])
    b_maps = _sample_maps(cur["samples"])
    observed = _statistic(a_maps, b_maps)
    pooled = a_maps + b_maps
    n_a = len(a_maps)
    rng = random.Random(0)  # 固定种子:p 值可复现
    ge = 0
    for _ in range(N_PERMUTATIONS):
        perm = pooled[:]
        rng.shuffle(perm)
        if _statistic(perm[:n_a], perm[n_a:]) >= observed:
            ge += 1
    p = (ge + 1) / (N_PERMUTATIONS + 1)
    verdict = "mismatch" if p <= P_MISMATCH else ("suspicious" if p <= P_SUSPICIOUS else "match")
    b_sm, c_sm = base.get("served_model"), cur.get("served_model")
    flags = []
    if b_sm and c_sm and b_sm != c_sm:
        flags.append("served_model_changed")
        verdict = "mismatch"
    return {
        "check": "logprob",
        "verdict": verdict,
        "statistic_S": round(observed, 4),
        "p_value": round(p, 4),
        "n_baseline": len(a_maps),
        "n_current": len(b_maps),
        "thresholds": {"mismatch_p": P_MISMATCH, "suspicious_p": P_SUSPICIOUS},
        "served_model": {"baseline": b_sm, "current": c_sm},
        "flags": flags,
        "cost_usd": cur.get("cost_usd"),
        "note": DISCLAIMER,
    }
