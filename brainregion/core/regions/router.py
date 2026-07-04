"""Router API —— 跨区域路由 seam(三级结构 Region 层;Phase 5)。

接口先行,算法可替换:`KeywordRouter`(== route_regions,reference 现有行为)+ `ManifestRouter`
(skill 描述 corpus,deterministic baseline)。wake_gate 经 `USE_ROUTER_API` flag 接入(真消费者)。
LLM/Embedding router 下 phase 同接口插。

设计(review_plan + GPT round-2):
- Router **构造时注入 corpus**(regions / skill-text),**不 import SkillRegistry**(decouple;匹配
  `route_regions(regions=...)` 模式)——caller 从 SkillRegistry.manifests_for_router() 建 corpus 后注入。
- `compare_routers(a, b, query)` 是 **Comparator**(非 strategy、非 Router),组合任意两 router
  (本期 Keyword×Manifest;以后 LLM×Embedding 直接复用)。
- `RouterResult` = route_regions 返 shape + `router` 字段;`KeywordRouter` 无损 wrap route_regions。
"""
from __future__ import annotations

import os
from typing import Protocol

from .loader import RegionDefinition, _contains, _normalize, load_regions, route_regions

# ManifestRouter 词袋 tokenization 的极小停用词(去常见噪声;baseline 用,LLMRouter 不依赖)。
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "via", "over", "to", "of", "in", "on",
    "a", "an", "is", "are", "be", "by", "or", "as", "at", "it", "its", "from", "into",
    "use", "used", "uses", "using", "can", "will", "not", "but", "which", "when",
})


def use_router_api() -> bool:
    """wake_gate 接入开关:env `BRAIN_REGION_USE_ROUTER_API`(默认关=现状)。开=走 KeywordRouter seam。"""
    return os.environ.get("BRAIN_REGION_USE_ROUTER_API", "").strip().lower() in ("1", "true", "yes", "on")


class RouterResult(dict):
    """route_regions 返 shape(selected/candidates/trace)+ `router` 字段。子类 dict 便于直传 wake_gate。"""


class Router(Protocol):
    """跨区域路由策略。构造时注入 corpus;`route(query) -> RouterResult`。"""

    name: str

    def route(self, *, goal: str = "", problem: str = "", context: str = "",
              files: dict[str, str] | None = None, top_k: int = 3, min_score: int = 2) -> RouterResult: ...


class KeywordRouter:
    """reference 现有行为:无损 wrap `route_regions`(+`router="keyword"`)。等价回归测试锁定。"""

    name = "keyword"

    def __init__(self, regions: list[RegionDefinition]) -> None:
        self.regions = regions

    def route(self, *, goal: str = "", problem: str = "", context: str = "",
              files: dict[str, str] | None = None, top_k: int = 3, min_score: int = 2) -> RouterResult:
        out = RouterResult(route_regions(
            goal=goal, problem=problem, context=context, files=files,
            top_k=top_k, min_score=min_score, regions=self.regions,
        ))
        out["router"] = "keyword"
        return out


class ManifestRouter:
    """reference **deterministic baseline**:keyword 打分 over skill 描述 corpus(3E 式读 metadata)。

    corpus = list[(region_id, text)];caller 从 SkillRegistry.manifests_for_router() 聚合每 region 的
    skill 描述+tags 成 text 后注入。**Router 不碰 registry**(decouple)。定位 = 永久 baseline,
    未来 LLM/Embedding/Hybrid router 都拿它做对比基线。空 corpus → 空选 + trace `no_corpus`。
    """

    name = "manifest"

    def __init__(self, corpus: list[tuple[str, str]]) -> None:
        self.corpus = corpus

    def _terms(self, text: str) -> list[str]:
        out: list[str] = []
        for tok in _normalize(text).split(" "):
            if len(tok) >= 3 and tok not in _STOPWORDS and tok not in out:
                out.append(tok)
        return out

    def route(self, *, goal: str = "", problem: str = "", context: str = "",
              files: dict[str, str] | None = None, top_k: int = 3, min_score: int = 2) -> RouterResult:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if min_score < 0:
            raise ValueError("min_score must be >= 0")
        text = _normalize("\n".join(p for p in (goal, problem, context) if p))
        file_text = _normalize("\n".join(str(p) for p in (files or {})))
        candidates: list[dict] = []
        for region_id, corpus_text in self.corpus:
            score = 0
            matched: list[dict] = []
            for term in self._terms(corpus_text):
                if _contains(text, term):
                    score += 2
                    matched.append({"trigger": term, "source": "text", "weight": 2})
                elif file_text and _contains(file_text, term):
                    score += 1
                    matched.append({"trigger": term, "source": "files", "weight": 1})
            if score > 0:
                candidates.append({
                    "id": region_id, "name": region_id, "description": "",
                    "score": score, "confidence": round(min(1.0, score / 8.0), 3),
                    "matched_triggers": matched, "negative_triggers": [],
                    "reasons": [f"matched {len(matched)} skill-term(s)"],
                })
        candidates.sort(key=lambda c: (-c["score"], c["id"]))
        positive = [c for c in candidates if c["score"] >= min_score]
        selected = positive[:top_k]
        result: RouterResult = RouterResult()
        result["selected"] = selected
        result["candidates"] = candidates
        result["router"] = "manifest"
        result["trace"] = {
            "strategy": "manifest_keyword_v1", "top_k": top_k, "min_score": min_score,
            "corpus_items": len(self.corpus), "no_corpus": not self.corpus,
            "available_regions": len({rid for rid, _ in self.corpus}),
        }
        return result


def route(*, strategy: str = "keyword",
          regions: list[RegionDefinition] | None = None,
          skill_corpus: list[tuple[str, str]] | None = None,
          goal: str = "", problem: str = "", context: str = "",
          files: dict[str, str] | None = None, top_k: int = 3, min_score: int = 2) -> RouterResult:
    """统一入口(恒返 RouterResult;shadow/对比走 compare_routers,非 strategy)。"""
    query = dict(goal=goal, problem=problem, context=context, files=files, top_k=top_k, min_score=min_score)
    if strategy == "keyword":
        return KeywordRouter(regions if regions is not None else load_regions()).route(**query)
    if strategy == "manifest":
        return ManifestRouter(skill_corpus or []).route(**query)
    raise ValueError(f"unknown strategy {strategy!r}; allowed: [keyword, manifest]")


def compare_routers(
    a: Router, b: Router, *, goal: str = "", problem: str = "", context: str = "",
    files: dict[str, str] | None = None, top_k: int = 3, min_score: int = 2,
) -> dict:
    """Comparator:跑两 Router,返 {a, b, agreement}。组合任意两 router(本期 Keyword×Manifest;以后 LLM×Embedding)。

    agreement: jaccard_selected(空集安全:union 空→None;单边空→0.0)、same_top(任一空 top→None)、
    per_region_disagree(选区对称差)。
    """
    ra = a.route(goal=goal, problem=problem, context=context, files=files, top_k=top_k, min_score=min_score)
    rb = b.route(goal=goal, problem=problem, context=context, files=files, top_k=top_k, min_score=min_score)
    sa = {s["id"] for s in ra.get("selected", [])}
    sb = {s["id"] for s in rb.get("selected", [])}
    union = sa | sb
    inter = sa & sb
    jaccard = None if not union else round(len(inter) / len(union), 4)   # 双空→None(不除零)
    ta = ra["selected"][0]["id"] if ra.get("selected") else None
    tb = rb["selected"][0]["id"] if rb.get("selected") else None
    same_top = None if (ta is None or tb is None) else (ta == tb)
    return {
        "a": {"router": ra.get("router"), "selected": sorted(sa)},
        "b": {"router": rb.get("router"), "selected": sorted(sb)},
        "agreement": {
            "jaccard_selected": jaccard,
            "same_top": same_top,
            "per_region_disagree": sorted(union - inter),
        },
    }
