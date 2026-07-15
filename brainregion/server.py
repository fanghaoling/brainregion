"""BrainRegion MCP server — AI collaboration infrastructure.

工具：
  审查：review_document / review_plan / review_code
  会诊：consult_problem / list_consultants / mark_advice
  自省：list_adapters / list_reviewers / list_knowledge / list_defaults / panel_stats
  健康：ping

设计要点：
- adapter="auto" 检测 Packages/manifest.json → UnityAdapter，否则 GenericAdapter。
- review_document 内部：先 retrieve 算缓存 key → 命中返回 → 未命中跑 8-Stage pipeline → record。
- 同步工具包 asyncio.run(engine.review)（engine 是 async，ReviewStage/NormalizeStage 内 gather/await）。
- 照搬 asset-gen：FastMCP + dict 返回 + stderr 日志 + 工具内直接 raise（FastMCP 自动 ToolError→isError）。
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# MCP stdio：stdout 必须干净（只走 JSON-RPC），日志统一写 stderr。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("brainregion")

# 加载 .env（若存在）到 os.environ：litellm 据此读 API key。.env 已 gitignore，不进 git。
# 系统环境变量优先（load_dotenv 默认不覆盖已存在的 env）。
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

mcp = FastMCP("brainregion")

from . import defaults as _defaults_mod  # noqa: E402
from . import output, prior as prior_mod, reviews_db  # noqa: E402
from .adapters.generic import GenericAdapter  # noqa: E402
from .adapters.unity import UnityAdapter  # noqa: E402
from .core.activation import ActivationSignal as _ActivationSignal  # noqa: E402
from .core.cognitive_workspace import CognitiveWorkspace as _CognitiveWorkspace  # noqa: E402
from .core.context_export import (  # noqa: E402
    bypass_context_export as _bypass_context_export,
    context_export_mode as _context_export_mode,
    endpoint_context_trust as _endpoint_context_trust,
    evaluate_context_export as _evaluate_context_export,
)
from .core.context_loader import load_activation_context as _load_activation_context  # noqa: E402
from .core.consult import ConsultEngine, ConsultRequest  # noqa: E402
from .core.consultants import CONSULTANTS_DIR, list_consultants as _list_consultant_files  # noqa: E402
from .core.engine import ReviewEngine  # noqa: E402
from .core.planner import PlanRequest, PlannerEngine  # noqa: E402
from .core.region_expert import RegionExpertEngine as _RegionExpertEngine  # noqa: E402
from .core.region_reporting import (  # noqa: E402
    RegionContextReceipt as _RegionContextReceipt,
    RegionCoordinationBoard as _RegionCoordinationBoard,
)
from .core.regions import REGIONS_DIR, load_regions as _load_regions  # noqa: E402
from .core.regions import route_regions as _route_regions  # noqa: E402
from .core.skills import (  # noqa: E402
    SKILLS_DIR as _SKILLS_DIR,
    SkillRegistry as _SkillRegistry,
    load_skills as _load_skills,
    setup_resolvers as _setup_skill_resolvers,
)
from .core.report import CanonicalFinding, Finding, ReviewReport  # noqa: E402
from .core.reviewers.loader import list_reviewers as _list_reviewer_files  # noqa: E402
from .core.workflow import suggest_workflow as _suggest_workflow  # noqa: E402
from .core.wake import wake_gate as _wake_gate  # noqa: E402
from .core.task_coordination import TaskCoordinationBoard as _TaskCoordinationBoard  # noqa: E402
from .core.stages import CORE_REVIEWERS_DIR, build_default_pipeline  # noqa: E402
from .core.stages.review import select_jobs_within_budget as _select_jobs_within_budget  # noqa: E402
from .core import ReviewDocument  # noqa: E402
from .knowledge import YamlKnowledgeProvider  # noqa: E402
from .core.context import ContextQuery as _ContextQuery  # noqa: E402
from .core.context import default_provider_registry as _default_provider_registry  # noqa: E402
from .memory import MemoryProvider, governance, store as memory_store  # noqa: E402
from .git import GitProvider  # noqa: E402
from .privacy import build_policy  # noqa: E402
from .providers import LiteLLMBackend  # noqa: E402
from .workspace import apply_text_patch as _apply_text_patch  # noqa: E402
from .workspace import inspect_file as _inspect_file  # noqa: E402
from .workspace import list_allowed_roots as _list_allowed_roots  # noqa: E402
from .workspace import read_text as _read_text  # noqa: E402
from .workspace import search_text as _search_text  # noqa: E402
from .workspace import workspace_run_check as _workspace_run_check  # noqa: E402

_ADAPTERS = {"unity": UnityAdapter, "generic": GenericAdapter}

_CONSULT_MODE_CONSULTANTS = {
    "debugging": ["debugger"],
    "architecture": ["architect", "critic"],
    "performance": ["performance", "critic"],
    "simplicity": ["simplicity", "maintenance"],
    "game_design": ["game_design", "critic"],
    "challenge": ["challenge", "critic"],
    "planning": ["architect", "test_designer", "critic"],
}


def _resolve_adapter(name: str, project_root: str):
    if name == "auto":
        if (Path(project_root) / "Packages" / "manifest.json").exists():
            return UnityAdapter(project_root)
        return GenericAdapter(project_root)
    cls = _ADAPTERS.get(name)
    if cls is None:
        raise ValueError(f"未知 adapter: {name}，可用: {sorted(list(_ADAPTERS) + ['auto'])}")
    return cls(project_root)


def _knowledge_dirs(adapter) -> list:
    """framework 通用知识库 + 项目本地 overlay（本地存在才加）。"""
    dirs = [adapter.knowledge_dir()]
    if hasattr(adapter, "local_knowledge_dirs"):
        candidates = adapter.local_knowledge_dirs()
    else:
        local = getattr(adapter, "local_knowledge_dir", lambda: None)()
        candidates = [local] if local else []
    for local in candidates:
        if local and Path(str(local)).exists():
            dirs.append(local)
    return dirs


def _resolve_endpoints(cfg: dict) -> dict:
    """config endpoints 块 -> {id: EndpointConfig{provider, base_url, api_key, headers, timeout}}。

    api_key_env 优先（os.environ.get），fallback api_key 明文，都无 raise 清晰配置错误。
    credential **只在此处解析**并交给 backend 持有，不进 PipelineContext。
    """
    registry: dict = {}
    for eid, ep in (cfg or {}).items():
        if not isinstance(ep, dict):
            raise ValueError(f"endpoint {eid!r} 配置必须是对象")
        provider = ep.get("provider")
        if provider not in ("openai", "anthropic"):
            raise ValueError(
                f"endpoint {eid!r} provider 必须是 openai|anthropic（中转兼容网关协议），得到 {provider!r}。"
                f"gemini/bedrock/vertex 等原生 provider 请用 litellm model 字符串（如 zai/glm-5.2）走 env，不走 endpoint。"
            )
        base_url = ep.get("base_url")
        if not base_url:
            raise ValueError(f"endpoint {eid!r} 缺 base_url")
        api_key = None
        env_name = ep.get("api_key_env")
        if env_name:
            api_key = os.environ.get(env_name)
            if not api_key:
                raise ValueError(f"endpoint {eid!r} api_key_env={env_name!r} 环境变量未设置或为空")
        elif ep.get("api_key"):
            api_key = ep["api_key"]
        else:
            raise ValueError(f"endpoint {eid!r} 缺 api_key_env 或 api_key")
        registry[eid] = {
            "provider": provider,
            "base_url": base_url,
            "api_key": api_key,
            "headers": ep.get("headers") or {},
            "timeout": ep.get("timeout"),
        }
    return registry


def _normalize_panel(
    panel: list, endpoint_ids: set, endpoints_cfg: dict | None = None
) -> list[dict]:
    """panel（list[str|dict]）-> list[PanelEntry{label, model, endpoint_id}]。

    str 四态（v2.3 加通配/短引用/全展开，向后兼容 litellm 原生）：
    - == "endpoints" → 通配：展开所有 endpoints.<id>.models（一行引全部中转站，厂商/模型都不用逐个填）
    - == endpoint_id → 全展开该 endpoint 的 models
    - endpoint_id/model → 短引用（label=id/model）
    - 否则 → litellm 原生官方（endpoint_id=None，走 env）
    dict={endpoint, model, label} 引用中转站（v1.6，自定义 label）。
    校验 label 全局唯一（撞名报错——label 是身份标识，撞名会让 consensus 错误合并）。
    PanelEntry **不含 credential**（key 只在 endpoint_registry，backend 边缘解析）。
    """
    entries: list[dict] = []
    labels: set[str] = set()
    for item in panel or []:
        if isinstance(item, dict):
            specs = [_dict_spec(item, endpoint_ids)]
        elif isinstance(item, str):
            specs = _str_specs(item, endpoint_ids, endpoints_cfg)
        else:
            raise ValueError(
                f"panel 项必须是 str（官方模型/短引用/全展开）或 dict（中转引用），得到 {type(item).__name__}"
            )
        for eid, model, label in specs:
            if label in labels:
                raise ValueError(f"panel label 撞名：{label!r}（label 是模型身份标识，撞名会让 consensus 错误合并）")
            labels.add(label)
            entries.append({"label": label, "model": model, "endpoint_id": eid})
    return entries


def _str_specs(item: str, endpoint_ids: set, endpoints_cfg: dict | None) -> list[tuple]:
    """str panel 项 → [(endpoint_id, model, label)]（通配/全展开返回多个）。

    - item == "endpoints" → 通配：展开所有 endpoints.<id>.models（一行引全部中转站模型，v2.3）
    - item == endpoint_id → 全展开该 endpoint 的 models
    - item = endpoint_id/model → 短引用（label=item）
    - 否则 → litellm 原生官方（endpoint_id=None）
    """
    # 通配：str == "endpoints" → 展开所有 endpoints 的所有 models（各 endpoint 须声明 models）
    if item == "endpoints":
        all_models: list[tuple] = []
        for eid, ep in (endpoints_cfg or {}).items():
            for m in _endpoint_model_ids(ep):
                all_models.append((eid, m, f"{eid}/{m}"))
        if not all_models:
            raise ValueError(
                "panel 'endpoints' 通配但无任何 endpoints.<id>.models 声明（每个中转站须声明 models）"
            )
        return all_models
    # 全展开：str 本身是 endpoint_id
    if item in endpoint_ids:
        models = _endpoint_model_ids((endpoints_cfg or {}).get(item) or {})
        if not models:
            raise ValueError(
                f"panel {item!r} 是 endpoint_id 但 endpoints.{item}.models 未声明（全展开需 models 列表）"
            )
        return [(item, m, f"{item}/{m}") for m in models]
    # 短引用：endpoint_id/model
    if "/" in item:
        prefix, _, model = item.partition("/")
        if prefix in endpoint_ids:
            return [(prefix, model, item)]  # label = id/model
    # litellm 原生官方
    return [(None, item, item)]


def _dict_spec(item: dict, endpoint_ids: set) -> tuple:
    """dict panel 项 → (endpoint_id, model, label)。引用中转站（v1.6，自定义 label）。"""
    eid = item.get("endpoint")
    if eid not in endpoint_ids:
        raise ValueError(f"panel 项引用了未定义的 endpoint {eid!r}（config endpoints 里没声明）")
    model = item.get("model")
    if not model:
        raise ValueError(f"panel 项（endpoint={eid}）缺 model")
    label = item.get("label") or f"{eid}/{model}"
    return (eid, model, label)


def _normalize_one(spec, endpoint_ids: set, endpoints_cfg: dict | None = None) -> dict:
    """单个 model 规格（str|dict）-> PanelEntry。供 normalizer 复用（schema 与 panel 统一）。

    QoL（单模型路径,--main-brain / solver / judge / router / normalizer 等用）:bare 模型名
    （无 ``/``、非 endpoint_id）若**唯一**匹配某 endpoint 的 ``models`` → 自动归端（省得手写
    ``endpoint/model``）;多 endpoint 命中 → 报错让用户用 ``endpoint/model`` 消歧;0 命中 → 走既有
    panel 路径（litellm 原生）。**只作用单模型路径**,不改 panel 语义（panel 里 bare 名仍走原生）。
    """
    if (
        isinstance(spec, str)
        and "/" not in spec
        and spec not in endpoint_ids
        and endpoints_cfg
    ):
        hits = [eid for eid, ep in endpoints_cfg.items() if spec in _endpoint_model_ids(ep)]
        if len(hits) == 1:
            eid = hits[0]
            return {"label": f"{eid}/{spec}", "model": spec, "endpoint_id": eid}
        if len(hits) > 1:
            raise ValueError(
                f"模型名 {spec!r} 在多个 endpoint 命中({hits});用 endpoint/model 形式消歧,如 {hits[0]}/{spec}"
            )
    return _normalize_panel([spec], endpoint_ids, endpoints_cfg)[0]


_PROFILE_KEYS = {
    "tier",
    "cost",
    "latency",
    "activation_role",
    "quality_score",
    "cost_score",
    "speed_score",
    "structured_output_score",
    "context_score",
    "tags",
    "capabilities",
    "notes",
}
_SCORE_KEYS = {
    "quality_score",
    "cost_score",
    "speed_score",
    "structured_output_score",
    "context_score",
}


def _model_id(spec) -> str:
    if isinstance(spec, dict):
        model = spec.get("id") or spec.get("model") or spec.get("name")
        if not model:
            raise ValueError(f"endpoint model object must include id or model: {spec!r}")
        return str(model)
    return str(spec)


def _endpoint_model_specs(ep: dict | None) -> list:
    return list((ep or {}).get("models") or [])


def _endpoint_model_ids(ep: dict | None) -> list[str]:
    return [_model_id(spec) for spec in _endpoint_model_specs(ep)]


def _as_profile_list(value) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in values:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _normalize_profile(profile: dict | None) -> dict:
    if not isinstance(profile, dict):
        return {}
    normalized: dict = {}
    for key, value in profile.items():
        if key in _SCORE_KEYS:
            try:
                normalized[key] = round(max(0.0, min(1.0, float(value))), 3)
            except Exception:  # noqa: BLE001
                continue
        elif key in ("tags", "capabilities"):
            normalized[key] = _as_profile_list(value)
        elif key in _PROFILE_KEYS or key == "profile_source":
            normalized[key] = value
    return {key: value for key, value in normalized.items() if value not in ("", [], None)}


def _merge_profiles(*profiles: dict | None) -> dict:
    merged: dict = {}
    sources: list[str] = []
    for profile in profiles:
        normalized = _normalize_profile(profile)
        if not normalized:
            continue
        source = normalized.pop("profile_source", None)
        if source:
            sources.extend(_as_profile_list(source))
        for key, value in normalized.items():
            if key in ("tags", "capabilities"):
                merged[key] = _as_profile_list(merged.get(key, []) + _as_profile_list(value))
            else:
                merged[key] = value
    if sources:
        merged["profile_source"] = _as_profile_list(sources)
    return merged


def _profile_from_model_spec(spec) -> dict:
    if not isinstance(spec, dict):
        return {}
    profile = dict(spec.get("profile") or {})
    for key, value in spec.items():
        if key in _PROFILE_KEYS:
            profile[key] = value
    if profile:
        profile["profile_source"] = "endpoint_model"
    return profile


def _endpoint_inline_profile(endpoint_id: str | None, model: str, endpoints_cfg: dict) -> dict:
    if endpoint_id is None:
        return {}
    for spec in _endpoint_model_specs((endpoints_cfg or {}).get(endpoint_id)):
        if _model_id(spec) == model:
            return _profile_from_model_spec(spec)
    return {}


def _inferred_model_profile(model: str) -> dict:
    """Coarse, non-authoritative profile used only for visibility."""
    m = str(model or "").casefold()
    profile = {
        "tier": "standard",
        "cost": "medium",
        "latency": "medium",
        "quality_score": 0.7,
        "cost_score": 0.5,
        "speed_score": 0.5,
        "structured_output_score": 0.6,
        "tags": ["general"],
        "profile_source": "heuristic",
    }
    if any(marker in m for marker in ("opus", "gpt-5.5", "o3", "max")):
        profile.update(
            {
                "tier": "flagship",
                "cost": "high",
                "quality_score": 0.95,
                "cost_score": 0.25,
                "speed_score": 0.45,
                "tags": ["flagship", "deep_reasoning"],
            }
        )
    elif any(marker in m for marker in ("mini", "haiku", "flash", "lite", "nano")):
        profile.update(
            {
                "tier": "economy",
                "cost": "low",
                "latency": "fast",
                "quality_score": 0.65,
                "cost_score": 0.85,
                "speed_score": 0.85,
                "tags": ["cheap", "fast"],
            }
        )
    if any(marker in m for marker in ("gpt", "claude", "opus", "o3", "o4")):
        profile["capabilities"] = ["reasoning", "coding", "review"]
    return profile


def _configured_profile(defaults: dict, *keys: str) -> dict:
    profiles = defaults.get("model_profiles") or {}
    if not isinstance(profiles, dict):
        return {}
    merged: dict = {}
    for key in keys:
        value = profiles.get(key)
        if isinstance(value, dict):
            value = {**value, "profile_source": f"model_profiles.{key}"}
            merged = _merge_profiles(merged, value)
    return merged


def _model_profile(
    *,
    model: str,
    label: str,
    endpoint_id: str | None,
    defaults: dict,
    endpoints_cfg: dict,
) -> dict:
    endpoint_ref = f"{endpoint_id}/{model}" if endpoint_id else ""
    return _merge_profiles(
        _inferred_model_profile(model),
        _configured_profile(defaults, model, label, endpoint_ref),
        _endpoint_inline_profile(endpoint_id, model, endpoints_cfg),
    )


def _official_credential_hint(model: str) -> str:
    """Best-effort hint for bare LiteLLM model strings."""
    m = str(model or "").lower()
    if m.startswith("claude-") or m.startswith("anthropic/"):
        return "ANTHROPIC_API_KEY"
    if m.startswith(("gpt-", "o1", "o3", "o4", "openai/")):
        return "OPENAI_API_KEY"
    if m.startswith("zai/"):
        return "ZAI_API_KEY"
    if m.startswith("deepseek/"):
        return "DEEPSEEK_API_KEY"
    if m.startswith("gemini/"):
        return "GEMINI_API_KEY"
    return "provider-specific environment variable"


def _endpoint_key_status(ep: dict) -> str:
    env_name = ep.get("api_key_env")
    if env_name:
        return "set" if os.environ.get(str(env_name)) else "missing"
    if ep.get("api_key"):
        return "plaintext_configured"
    return "missing"


def _configured_endpoint_models(endpoints_cfg: dict) -> list[dict]:
    endpoints: list[dict] = []
    for eid, ep in sorted((endpoints_cfg or {}).items()):
        model_specs = _endpoint_model_specs(ep)
        models = [_model_id(spec) for spec in model_specs]
        endpoints.append(
            {
                "id": eid,
                "provider": ep.get("provider"),
                "base_url": ep.get("base_url"),
                "api_key_env": ep.get("api_key_env") or "",
                "api_key_status": _endpoint_key_status(ep),
                "models": models,
                "model_refs": [f"{eid}/{model}" for model in models],
                "model_profiles": [
                    {
                        "id": _model_id(spec),
                        "ref": f"{eid}/{_model_id(spec)}",
                        "profile": _normalize_profile(_profile_from_model_spec(spec)),
                    }
                    for spec in model_specs
                    if _profile_from_model_spec(spec)
                ],
            }
        )
    return endpoints


_GATEWAY_PREFIX_MARKERS = ("modelbridge", "newapi", "oneapi", "gateway", "relay", "proxy")


def _unknown_gateway_prefix(label: str, endpoint_ids: set[str]) -> str:
    if "/" not in label:
        return ""
    prefix = label.split("/", 1)[0]
    if prefix in endpoint_ids:
        return ""
    normalized = prefix.replace("-", "_").casefold()
    return prefix if any(marker in normalized for marker in _GATEWAY_PREFIX_MARKERS) else ""


def _route_warnings(routes: list[dict], ambiguous_models: list[dict], endpoint_ids: set[str]) -> list[dict]:
    warnings: list[dict] = []
    for route in routes:
        if route.get("route_type") == "configured_endpoint" and route.get("api_key_status") == "missing":
            warnings.append(
                {
                    "type": "missing_endpoint_key",
                    "model": route.get("model"),
                    "label": route.get("label"),
                    "endpoint_id": route.get("endpoint_id"),
                    "message": f"Endpoint {route.get('endpoint_id')!r} key is not available in the current process.",
                }
            )
        if route.get("route_type") == "official_litellm":
            prefix = _unknown_gateway_prefix(str(route.get("label") or ""), endpoint_ids)
            if prefix:
                warnings.append(
                    {
                        "type": "unknown_endpoint_prefix",
                        "model": route.get("model"),
                        "label": route.get("label"),
                        "endpoint_id": prefix,
                        "message": (
                            f"Model spec {route.get('label')!r} looks like an endpoint/model ref, "
                            f"but endpoint {prefix!r} is not configured; it will use official LiteLLM routing."
                        ),
                    }
                )
    for item in ambiguous_models:
        model = item["model"]
        refs = item["endpoint_refs"]
        if "bare_model_string_also_used" in item["reasons"]:
            warnings.append(
                {
                    "type": "bare_model_has_endpoint_ref",
                    "model": model,
                    "official_ref": item.get("official_ref"),
                    "endpoint_refs": refs,
                    "message": f"Bare model {model!r} bypasses configured endpoints; use {refs[0]!r} to route through that gateway.",
                }
            )
        if "declared_under_multiple_endpoints" in item["reasons"]:
            warnings.append(
                {
                    "type": "model_declared_under_multiple_endpoints",
                    "model": model,
                    "endpoint_refs": refs,
                    "message": f"Model {model!r} is declared under multiple endpoints; use an endpoint prefix to choose explicitly.",
                }
            )
    return warnings


def _describe_model_routes(panel: list | None, defaults: dict, *, panel_source: str = "explicit") -> dict:
    """Describe how model specs resolve without touching credentials or calling models."""
    endpoints_cfg = defaults.get("endpoints") or {}
    raw_panel = list(panel if panel is not None else defaults.get("panel") or [])
    endpoint_ids = set(endpoints_cfg.keys())
    resolved = _normalize_panel(raw_panel, endpoint_ids, endpoints_cfg)
    endpoints = _configured_endpoint_models(endpoints_cfg)

    endpoint_model_refs: dict[str, list[str]] = {}
    for endpoint in endpoints:
        for model in endpoint["models"]:
            endpoint_model_refs.setdefault(model, []).append(f"{endpoint['id']}/{model}")

    routes: list[dict] = []
    bare_models = set()
    for entry in resolved:
        endpoint_id = entry.get("endpoint_id")
        model = entry["model"]
        if endpoint_id is None:
            bare_models.add(model)
            routes.append(
                {
                    "label": entry["label"],
                    "model": model,
                    "endpoint_id": None,
                    "route_type": "official_litellm",
                    "credential_hint": _official_credential_hint(model),
                    "profile": _model_profile(
                        model=model,
                        label=entry["label"],
                        endpoint_id=None,
                        defaults=defaults,
                        endpoints_cfg=endpoints_cfg,
                    ),
                    "note": "Bare model strings bypass configured endpoints. Use endpoint_id/model to route through a gateway.",
                }
            )
            continue
        ep = endpoints_cfg.get(endpoint_id) or {}
        routes.append(
            {
                "label": entry["label"],
                "model": model,
                "endpoint_id": endpoint_id,
                "route_type": "configured_endpoint",
                "provider": ep.get("provider"),
                "base_url": ep.get("base_url"),
                "api_key_env": ep.get("api_key_env") or "",
                "api_key_status": _endpoint_key_status(ep),
                "profile": _model_profile(
                    model=model,
                    label=entry["label"],
                    endpoint_id=endpoint_id,
                    defaults=defaults,
                    endpoints_cfg=endpoints_cfg,
                ),
            }
        )

    ambiguous_models: list[dict] = []
    for model, refs in sorted(endpoint_model_refs.items()):
        reasons: list[str] = []
        if len(refs) > 1:
            reasons.append("declared_under_multiple_endpoints")
        if model in bare_models:
            reasons.append("bare_model_string_also_used")
        if reasons:
            ambiguous_models.append(
                {
                    "model": model,
                    "endpoint_refs": refs,
                    "official_ref": model if model in bare_models else "",
                    "reasons": reasons,
                }
            )

    return {
        "panel_source": panel_source,
        "panel": raw_panel,
        "resolved_panel": routes,
        "endpoints": endpoints,
        "available_model_refs": sorted(
            ref for refs in endpoint_model_refs.values() for ref in refs
        ),
        "ambiguous_models": ambiguous_models,
        "warnings": _route_warnings(routes, ambiguous_models, endpoint_ids),
        "notes": [
            "A bare model string such as 'claude-opus-4-8' uses the official LiteLLM provider route.",
            "Use 'endpoint_id/model' such as 'modelbridge_anthropic/claude-opus-4-8' to use a configured gateway key.",
            "Profiles are descriptive metadata for preflight and suggest_panel; they never auto-call models.",
        ],
    }


_PANEL_STRATEGIES = {
    "balanced": {
        "quality_score": 0.35,
        "cost_score": 0.25,
        "speed_score": 0.20,
        "structured_output_score": 0.15,
        "context_score": 0.05,
    },
    "cheap_fast": {
        "cost_score": 0.45,
        "speed_score": 0.35,
        "structured_output_score": 0.15,
        "quality_score": 0.05,
    },
    "best_reasoning": {
        "quality_score": 0.60,
        "structured_output_score": 0.20,
        "context_score": 0.10,
        "speed_score": 0.05,
        "cost_score": 0.05,
    },
    "sleep": {
        "cost_score": 0.40,
        "speed_score": 0.30,
        "quality_score": 0.15,
        "structured_output_score": 0.15,
    },
    "awake": {
        "quality_score": 0.65,
        "structured_output_score": 0.15,
        "context_score": 0.10,
        "speed_score": 0.05,
        "cost_score": 0.05,
    },
    "structured_output": {
        "structured_output_score": 0.45,
        "quality_score": 0.25,
        "cost_score": 0.15,
        "speed_score": 0.15,
    },
}


def _official_key_status(route: dict) -> str:
    hint = route.get("credential_hint") or ""
    if hint.endswith("_API_KEY"):
        return "set" if os.environ.get(str(hint)) else "missing"
    return "unknown"


def _route_key_status(route: dict) -> str:
    if route.get("route_type") == "configured_endpoint":
        return str(route.get("api_key_status") or "missing")
    return _official_key_status(route)


def _route_key_available(route: dict) -> bool:
    return _route_key_status(route) in ("set", "plaintext_configured", "unknown")


def _task_tag_boost(profile: dict, task: str) -> tuple[float, list[str]]:
    text = str(task or "").casefold()
    if not text:
        return 0.0, []
    tags = set(_as_profile_list(profile.get("tags")) + _as_profile_list(profile.get("capabilities")))
    matched: list[str] = []
    checks = {
        "architecture": ["architecture", "design", "\u67b6\u6784", "\u8bbe\u8ba1"],
        "coding": ["code", "coding", "\u4ee3\u7801", "\u5b9e\u73b0"],
        "review": ["review", "audit", "\u5ba1\u67e5", "\u8bc4\u5ba1"],
        "reasoning": ["reason", "planning", "plan", "\u63a8\u7406", "\u89c4\u5212"],
        "debugging": ["debug", "bug", "failure", "\u8c03\u8bd5", "\u62a5\u9519"],
        "performance": ["performance", "latency", "cost", "\u6027\u80fd", "\u5ef6\u8fdf", "\u6210\u672c"],
    }
    for tag, needles in checks.items():
        if tag in tags and any(needle in text for needle in needles):
            matched.append(tag)
    return min(0.12, 0.03 * len(matched)), matched


def _score_route(route: dict, strategy: str, task: str = "") -> dict:
    profile = route.get("profile") or {}
    weights = _PANEL_STRATEGIES.get(strategy, _PANEL_STRATEGIES["balanced"])
    components: dict[str, float] = {}
    score = 0.0
    for key, weight in weights.items():
        value = profile.get(key)
        try:
            numeric = float(value)
        except Exception:  # noqa: BLE001
            numeric = 0.0
        component = round(numeric * weight, 4)
        components[key] = component
        score += component

    bonuses: dict[str, float | list[str]] = {}
    role = str(profile.get("activation_role") or "").casefold()
    tier = str(profile.get("tier") or "").casefold()
    tags = set(_as_profile_list(profile.get("tags")))
    if strategy == "sleep" and role == "sleep":
        score += 0.15
        bonuses["activation_role"] = 0.15
    if strategy == "awake" and (role == "awake" or tier == "flagship"):
        score += 0.15
        bonuses["activation_role_or_tier"] = 0.15
    if strategy == "best_reasoning" and ("deep_reasoning" in tags or tier == "flagship"):
        score += 0.08
        bonuses["reasoning_tag_or_tier"] = 0.08

    boost, matched_tags = _task_tag_boost(profile, task)
    if boost:
        score += boost
        bonuses["task_match"] = round(boost, 4)
        bonuses["matched_tags"] = matched_tags

    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "score_breakdown": components,
        "bonuses": bonuses,
    }


def _candidate_panel(defaults: dict) -> tuple[list, str]:
    endpoints_cfg = defaults.get("endpoints") or {}
    has_endpoint_models = any(_endpoint_model_ids(ep) for ep in endpoints_cfg.values())
    if has_endpoint_models:
        return ["endpoints"], "configured_endpoints"
    return list(defaults.get("panel") or []), "panel"


def _suggest_panel(
    *,
    defaults: dict,
    strategy: str = "balanced",
    task: str = "",
    panel: list[str] | None = None,
    max_models: int = 2,
    require_available_key: bool = True,
) -> dict:
    if max_models <= 0:
        raise ValueError("max_models must be greater than 0")
    effective_strategy = strategy if strategy in _PANEL_STRATEGIES else "balanced"
    raw_panel, source = (list(panel), "explicit") if panel is not None else _candidate_panel(defaults)
    route_info = _describe_model_routes(raw_panel, defaults, panel_source=source)

    candidates: list[dict] = []
    for route in route_info["resolved_panel"]:
        key_status = _route_key_status(route)
        scoring = _score_route(route, effective_strategy, task)
        candidate = {
            **route,
            "key_status": key_status,
            "selectable": (not require_available_key) or _route_key_available(route),
            **scoring,
        }
        if not candidate["selectable"]:
            candidate["excluded_reason"] = "credential_missing"
        candidates.append(candidate)

    ranked = sorted(candidates, key=lambda item: (-item["selectable"], -item["score"], item["label"]))
    selected = [item for item in ranked if item["selectable"]][:max_models]
    return {
        "strategy": effective_strategy,
        "requested_strategy": strategy,
        "task": task,
        "selected_panel": [item["label"] for item in selected],
        "selected": selected,
        "candidates": ranked,
        "warnings": route_info["warnings"],
        "ambiguous_models": route_info["ambiguous_models"],
        "trace": {
            "candidate_source": source,
            "max_models": max_models,
            "require_available_key": require_available_key,
            "models_called": False,
            "auto_execute": False,
            "available_strategies": sorted(_PANEL_STRATEGIES),
            "no_selection_reason": "" if selected else "no_selectable_models",
        },
    }


def _build_engine(adapter, dd: dict) -> ReviewEngine:
    registry = _resolve_endpoints(dd.get("endpoints") or {})
    endpoint_ids = set(registry.keys())
    backend = LiteLLMBackend(timeout=float(dd.get("timeout", 90)), endpoint_registry=registry)
    knowledge = YamlKnowledgeProvider(_knowledge_dirs(adapter))
    # v1.7 隐私策略：解析 trusted（复用 endpoint）+ build_policy（off→None / strict→StrictPolicy）
    privacy_cfg = dd.get("privacy_policy")
    trusted_entry = (
        _normalize_one(privacy_cfg["trusted"], endpoint_ids, dd.get("endpoints"))
        if (isinstance(privacy_cfg, dict) and privacy_cfg.get("trusted"))
        else None
    )
    policy = build_policy(privacy_cfg, trusted_entry)
    pipeline = build_default_pipeline(
        normalizer=_normalize_one(dd.get("normalizer_model", "claude-opus-4-8"), endpoint_ids, dd.get("endpoints")),
        threshold=int(dd.get("consensus_threshold", 2)),
        policy=policy,
    )
    return ReviewEngine(
        adapter=adapter, backend=backend, knowledge=knowledge,
        pipeline=pipeline, defaults=dd, policy=policy,
    )


def _build_consult_engine(dd: dict) -> ConsultEngine:
    registry = _resolve_endpoints(dd.get("endpoints") or {})
    backend = LiteLLMBackend(timeout=float(dd.get("timeout", 90)), endpoint_registry=registry)
    return ConsultEngine(backend=backend, consultants_dir=CONSULTANTS_DIR)


def _build_planner_engine(dd: dict) -> PlannerEngine:
    registry = _resolve_endpoints(dd.get("endpoints") or {})
    backend = LiteLLMBackend(timeout=float(dd.get("timeout", 90)), endpoint_registry=registry)
    return PlannerEngine(backend=backend)


def _build_region_expert_engine(dd: dict, endpoint_registry: dict) -> _RegionExpertEngine:
    backend = LiteLLMBackend(
        timeout=float(dd.get("timeout", 90)), endpoint_registry=endpoint_registry
    )
    return _RegionExpertEngine(backend=backend)


def _resolve_consultants(consultants: list[str] | None, mode: str | None, defaults: dict) -> tuple[list[str], str | None]:
    """Resolve consultant roles. Explicit consultants win; mode picks a preset."""
    effective_mode = mode if mode is not None else defaults.get("consult_mode")
    if consultants is not None:
        return list(consultants), effective_mode
    if effective_mode:
        preset = _CONSULT_MODE_CONSULTANTS.get(effective_mode)
        if preset is None:
            raise ValueError(f"未知 consult mode: {effective_mode!r}，可用: {sorted(_CONSULT_MODE_CONSULTANTS)}")
        return list(preset), effective_mode
    return list(defaults.get("consult_consultants") or []), None


def _resolve_consult_panel(panel: list[str] | None, defaults: dict) -> tuple[list, str]:
    """Resolve consult panel and track the source for debugging/testing."""
    if panel is not None:
        return panel, "explicit"
    if defaults.get("consult_panel"):
        return defaults.get("consult_panel") or [], "consult_panel"
    return defaults.get("panel") or [], "panel"


def _resolve_planner_panel(panel: list[str] | None, defaults: dict) -> tuple[list, str]:
    """Resolve planner panel without making planning depend on the full review panel by default."""
    if panel is not None:
        return panel, "explicit"
    if defaults.get("planner_panel"):
        return defaults.get("planner_panel") or [], "planner_panel"
    if defaults.get("consult_panel"):
        return defaults.get("consult_panel") or [], "consult_panel"
    return defaults.get("panel") or [], "panel"


def _resolve_consultants_with_source(
    consultants: list[str] | None, mode: str | None, defaults: dict
) -> tuple[list[str], str | None, str, str]:
    """Resolve consultant roles and source labels for routing metadata."""
    effective_mode = mode if mode is not None else defaults.get("consult_mode")
    mode_source = "explicit" if mode is not None else ("consult_mode" if defaults.get("consult_mode") else "none")
    if consultants is not None:
        return list(consultants), effective_mode, "explicit", mode_source
    if effective_mode:
        resolved, mode_used = _resolve_consultants(None, effective_mode, defaults)
        return resolved, mode_used, "mode", mode_source
    return list(defaults.get("consult_consultants") or []), None, "consult_consultants", mode_source


def _rebuild_report(d: dict) -> ReviewReport:
    """从缓存的 dict 重建 ReviewReport（dataclass 字段过滤，忽略 cache_hit 等额外字段）。"""
    cf_fields = CanonicalFinding.__dataclass_fields__
    f_fields = Finding.__dataclass_fields__

    def _cf(c: dict) -> CanonicalFinding:
        return CanonicalFinding(**{k: v for k, v in c.items() if k in cf_fields})

    def _f(f: dict, fallback_id: str) -> Finding:
        kw = {k: v for k, v in f.items() if k in f_fields}
        if not kw.get("id"):  # v2 旧缓存 finding 无 id → 就地补填（让旧 review 也能被 mark_finding）
            kw["id"] = fallback_id
        return Finding(**kw)

    return ReviewReport(
        document_type=d.get("document_type", ""),
        adapter=d.get("adapter", ""),
        project_version=d.get("project_version", {}),
        panel=d.get("panel", []),
        failed_models=d.get("failed_models", []),
        retrieved_cases=d.get("retrieved_cases", []),
        consensus=[_cf(c) for c in d.get("consensus", [])],
        majority=[_cf(c) for c in d.get("majority", [])],
        individual={
            k: [_f(f, f"{k}-{idx}") for idx, f in enumerate(v)]
            for k, v in d.get("individual", {}).items()
        },
        knowledge_hit=d.get("knowledge_hit", []),
        usage=d.get("usage", {}),
        summary=d.get("summary", ""),
        risk=d.get("risk", {}),
        privacy=d.get("privacy", {}),  # v2 修 bug：缓存命中补回（原 :197 risk= 后停漏了）
        context_compression=d.get("context_compression", {}),
    )


def _common_review_kwargs():
    """review_plan/review_code 共享的显式参数（FastMCP 需显式 schema）。"""
    return dict(
        adapter="auto", panel=None, dimensions=None, retrieve_top_k=5,
        extra_context="", output_format="json",
    )


@mcp.tool()
def ping() -> dict:
    """健康检查：确认 BrainRegion MCP server 可达。"""
    from . import __version__

    return {"ok": True, "name": "brainregion", "legacy_name": "brain_region", "version": __version__}


# v2 Review Memory：标记 finding 采纳，写入 reliability 飞轮
_FINDING_ID_RE = re.compile(r"^.+-\d+$")


def _label_from_id(finding_id: str) -> str:
    """从 finding_id '{label}-{seq}' 解析 label（rsplit 仅 fallback；主路径查 report）。
    label 可含 '-'/'/'（如 "智谱-Anthropic端点"、"zai/glm-5.2"）——rsplit('-',1) 只切末尾 seq。"""
    return finding_id.rsplit("-", 1)[0] if "-" in finding_id else finding_id


@mcp.tool()
def mark_finding(
    finding_id: str,
    decision: str,
    params_hash: str | None = None,
    note: str = "",
    invalidate_cache: bool = True,
) -> dict:
    """标记一条 finding 的采纳情况，写入 Review Memory，供下次 review 模型可信度加权。

    finding_id/params_hash 从 review_document 返回取。未传 params_hash 时按 finding_id
    反查最近含此 id 的 review（扫 consensus+majority+individual+deduped_ids）。
    decision: accepted|rejected|partial。标记后默认失效该 review 缓存，下次同内容审查重算
    reliability（该模型该维度按历史采纳率降/升权）。note 是 decision reason 自由文本。
    """
    if not finding_id or not _FINDING_ID_RE.match(finding_id):
        raise ValueError(f"finding_id 格式无效（应为 '{{label}}-{{seq}}'）: {finding_id!r}")
    if decision not in reviews_db.VALID_DECISIONS:
        raise ValueError(f"decision 必须是 {sorted(reviews_db.VALID_DECISIONS)}，得到 {decision!r}")
    try:
        if params_hash is not None:
            phash = params_hash
            report = reviews_db.lookup_report(phash)
            if report is None:
                raise ValueError(f"params_hash={phash[:8]}… 找不到缓存 review")
            scanned = reviews_db._scan_report_for_finding(report, finding_id)
            if scanned is None:
                raise ValueError(f"review {phash[:8]}… 中找不到 finding_id={finding_id!r}")
            label, dimension = scanned
        else:
            phash, label, dimension = reviews_db.lookup_review_by_finding(finding_id)
            if phash is None:
                raise ValueError(f"找不到含 finding_id={finding_id!r} 的 review，请显式传 params_hash")
            if not label:  # deduped_ids 分支 label 可能空 → rsplit fallback
                label = _label_from_id(finding_id)
            if not dimension:
                dimension = ""
        if not label:
            raise ValueError(f"无法确定 finding_id={finding_id!r} 的 model label")

        reviews_db.record_feedback(
            finding_id=finding_id, params_hash=phash, label=label,
            dimension=dimension, decision=decision, note=note,
        )
        invalidated = reviews_db.invalidate_review_cache(phash) if invalidate_cache else False
        return {
            "ok": True, "finding_id": finding_id, "params_hash": phash,
            "label": label, "dimension": dimension, "decision": decision,
            "cache_invalidated": invalidated,
        }
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 — 不抛错，返回 ok=False（v1.8 降级规范）
        return {"ok": False, "finding_id": finding_id, "error": str(e)}


@mcp.tool()
def mark_advice(
    advice_id: str,
    decision: str,
    consultation_id: str | None = None,
    reason: str = "",
    outcome: str = "",
) -> dict:
    """标记一条外援 advice 是否有用，写入 Advice Memory。

    advice_id/consultation_id 从 consult_problem 返回取。decision:
    accepted|rejected|partial|unknown。只记录最小反馈元数据和用户反馈文本，不保存原始
    prompt、问题正文或 advice 全文。
    """
    res = reviews_db.record_advice_feedback(
        advice_id=advice_id,
        consultation_id=consultation_id,
        decision=decision,
        reason=reason,
        outcome=outcome,
    )
    return {"ok": True, **res}


@mcp.tool()
def record_experience(
    summary: str,
    details: str = "",
    triggers: list[str] | None = None,
    region: str = "",
    source: str = "",
    status: str = "active",
    valid_until_ts: int = 0,
    supersedes: str = "",
) -> dict:
    """记录一条经验到 Experience Memory（append-only），供后续按关键词召回注入 consult context。

    summary 必填；triggers 是召回关键词（词面命中）；region 可空（全局）。
    v6 stage 1 治理：status(active|pending|superseded|wrong,默认 active)、valid_until_ts(Unix 秒,0=永不过期)、
    supersedes(旧记忆 id——记录新后自动把旧记忆标 superseded)。返回 {ok, id}。
    注入由 config memory_inject 门控（默认关）；召回检视见 recall_experiences；改状态见 set_experience_status。
    """
    try:
        normalized_region = _normalize_experience_region(region)
        return memory_store.record_experience(
            summary=summary, details=details, triggers=triggers or [],
            region=normalized_region, source=source, status=status,
            valid_until_ts=valid_until_ts, supersedes=supersedes,
        )
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 — 降级规范
        return {"ok": False, "error": str(e)}


@mcp.tool()
def set_experience_status(
    id: str,
    status: str,
    superseded_by: str = "",
    valid_until_ts: int | None = None,
) -> dict:
    """更新一条经验的治理状态（v6 stage 1，人工纠错）。

    status ∈ {active, pending, superseded, wrong}。自由可逆（误标可改回）：
    status→active 时自动 stamp last_reviewed。superseded_by（status=superseded 时指向替代者 id）。
    valid_until_ts(Unix 秒,0=永不过期)。默认召回只含 active/pending 且未过期(superseded/wrong/expired 退出)。
    """
    try:
        return memory_store.set_experience_status(
            id, status, superseded_by=superseded_by or None,
            valid_until_ts=valid_until_ts,
        )
    except Exception as e:  # noqa: BLE001 — 降级规范
        return {"ok": False, "id": id, "error": str(e)}


@mcp.tool()
def mark_superseded(old_id: str, new_id: str) -> dict:
    """把 old 记忆标 superseded（被 new 覆盖,退出召回）。set_experience_status 的便利封装。"""
    try:
        return memory_store.mark_superseded(old_id, new_id)
    except Exception as e:  # noqa: BLE001 — 降级规范
        return {"ok": False, "id": old_id, "error": str(e)}


@mcp.tool()
def recall_experiences(
    text: str,
    top_k: int = 5,
    region: str | None = None,
    include_inactive: bool = False,
) -> dict:
    """按关键词召回相关经验（只读，不调模型）。用于检视 Experience Memory 会召回什么。

    默认 ``include_inactive=False`` 镜像生产 retrieve(只含 active/pending 且未过期);
    True=含 superseded/wrong/expired 供排查。返回 {count, experiences:[...完整 to_dict...]}。
    """
    try:
        hits = memory_store.search(text, top_k=top_k, region=region)
        if not include_inactive:
            hits = [e for e in hits if governance.is_recallable(e)]
        return {"count": len(hits), "experiences": [e.to_dict() for e in hits]}
    except Exception as e:  # noqa: BLE001 — 降级规范
        return {"count": 0, "experiences": [], "error": str(e)}


@mcp.tool()
async def review_document(
    content: str,
    document_type: str = "markdown",
    files: dict | None = None,
    adapter: str = "auto",
    panel: list[str] | None = None,
    dimensions: list[str] | None = None,
    retrieve_top_k: int | None = None,
    extra_context: str = "",
    output_format: str | None = None,
    timeout: float | None = None,
    effort: str | None = None,
    max_cost_usd: float | None = None,
) -> dict:
    """审查一份文档（markdown/code/adr/rfc/config）。

    多模型 fan-out（panel × dimensions）+ 知识库 retrieve（版本过滤）+ canonical 归一
    + 校准共识。返回结构化报告（consensus/majority/individual + calibrated_confidence）。

    Args:
        content: 文档正文（markdown/adr/rfc/config）。
        document_type: 文档类型，影响 prompt 模板。
        files: 代码文件 {路径: 源码}（code 模式）。
        adapter: "auto" 自动检测，或 "unity"/"generic"。
        panel: 模型列表，None=默认面板（需配 OPENAI/ANTHROPIC/ARK key）。
        dimensions: 审查维度，None=自动（core planner/safety + adapter 特定）。
        retrieve_top_k: 知识库 retrieve 案例数。
        extra_context: 额外补充 context（核心 context 由 adapter 自动聚合）。
        output_format: json|markdown|sarif。json 返回结构化；其余额外加 rendered 字段。
        timeout: 单模型超时秒。
        effort: 思考强度 low/medium/high/xhigh/max；None=各模型默认。仅 Claude（output_config+thinking adaptive）/ OpenAI o 系列（reasoning_effort）生效，其余丢弃。Claude 默认 high 较贵，routine 方案可降 medium 省 token。
        max_cost_usd: 单次 review 总成本上限（USD）；None=无上限。设了则预 flight 估每 job 成本、按 panel 顺序裁剪直到估算超预算，report.budget.exhausted 标记是否裁过。

    Returns:
        报告 dict + cache_hit/reuse_count（+ rendered 若非 json）。
    """
    dd = _defaults_mod.apply(
        panel=panel, dimensions=dimensions, retrieve_top_k=retrieve_top_k,
        output_format=output_format, timeout=timeout, effort=effort, max_cost_usd=max_cost_usd,
    )
    panel_used = _normalize_panel(
        dd["panel"], set((dd.get("endpoints") or {}).keys()), dd.get("endpoints")
    )
    dims_used = dd["dimensions"]
    root = os.environ.get("UNITY_PROJECT_ROOT", ".")
    ad = _resolve_adapter(adapter, root)
    knowledge = YamlKnowledgeProvider(_knowledge_dirs(ad))
    version = ad.read_version()
    text = content or ""
    if files:
        text += "\n" + "\n".join(files.values())
    retrieved = knowledge.retrieve(text, version, int(dd["retrieve_top_k"]))
    retrieved_ids = [c.id for c in retrieved]

    phash = reviews_db.compute_hash(
        document_content=content, document_files=files, panel=panel_used,
        dimensions=dims_used, adapter=ad.name, project_version=version,
        retrieved_cases_ids=retrieved_ids, extra_context=extra_context,
        effort=dd.get("effort"), max_cost_usd=dd.get("max_cost_usd"),
    )
    cached = reviews_db.lookup(phash)
    effective_output_format = dd["output_format"]
    if cached is not None:
        result = dict(cached["report"])
        result["cache_hit"] = True
        result["reuse_count"] = cached["reuse_count"]
        result["params_hash"] = phash  # v2 mark_finding 引用
        if effective_output_format != "json":
            result["rendered"] = output.render(_rebuild_report(cached["report"]), effective_output_format)
        return result

    engine = _build_engine(ad, dd)
    doc = ReviewDocument(type=document_type, content=content or "", files=files)
    # v1.8 context_modes 校验（Fail Fast：用户配置错不该偷偷 fallback）+ 透传
    context_modes = dd.get("context_modes") or {}
    for dim, mode in context_modes.items():
        if mode not in ("full", "compressed", "minimal"):
            raise ValueError(f"context_modes.{dim}={mode!r} 无效（full|compressed|minimal）")
    # v2 模型可信度（纯 dict 注入 core，core 不依赖 reviews_db；命中分支不重算——用缓存的 calibrated）
    # v2.2 加 warm-start 先验（prior.load：mode 三态，默认 builtin 今天空=v2.1，official 填入自动生效）
    reliability = reviews_db.model_reliability(
        [e["label"] for e in panel_used],
        prior=prior_mod.load(dd.get("model_reliability_prior")),
    )
    ctx = await engine.review(
        doc, panel=panel_used, dimensions=dims_used,
        retrieve_top_k=int(dd["retrieve_top_k"]), extra_context=extra_context,
        effort=dd.get("effort"), max_cost_usd=dd.get("max_cost_usd"),
        context_modes=context_modes, reliability=reliability,
    )
    report = ctx.report
    report_dict = report.to_dict()
    reviews_db.record(phash, report_dict=report_dict, adapter=ad.name, panel=panel_used)
    result = dict(report_dict)
    result["cache_hit"] = False
    result["params_hash"] = phash  # v2 mark_finding 引用
    if effective_output_format != "json":
        result["rendered"] = output.render(report, effective_output_format)
    return result


@mcp.tool()
async def review_plan(
    plan_text: str,
    adapter: str = "auto",
    panel: list[str] | None = None,
    dimensions: list[str] | None = None,
    retrieve_top_k: int | None = None,
    extra_context: str = "",
    output_format: str | None = None,
    effort: str | None = None,
    max_cost_usd: float | None = None,
) -> dict:
    """审查实现方案/计划（design-question 模式）。等价 review_document(document_type="markdown")。"""
    return await review_document(
        content=plan_text, document_type="markdown", files=None, adapter=adapter,
        panel=panel, dimensions=dimensions, retrieve_top_k=retrieve_top_k,
        extra_context=extra_context, output_format=output_format,
        effort=effort, max_cost_usd=max_cost_usd,
    )


@mcp.tool()
async def review_code(
    files: dict[str, str],
    adapter: str = "auto",
    panel: list[str] | None = None,
    dimensions: list[str] | None = None,
    retrieve_top_k: int | None = None,
    extra_context: str = "",
    output_format: str | None = None,
    effort: str | None = None,
    max_cost_usd: float | None = None,
) -> dict:
    """审查代码实现（code-review 模式）。等价 review_document(document_type="code")。"""
    return await review_document(
        content="", document_type="code", files=files, adapter=adapter,
        panel=panel, dimensions=dimensions, retrieve_top_k=retrieve_top_k,
        extra_context=extra_context, output_format=output_format,
        effort=effort, max_cost_usd=max_cost_usd,
    )


@mcp.tool()
async def plan_task(
    goal: str,
    context: str = "",
    constraints: list[str] | None = None,
    success_criteria: list[str] | None = None,
    existing_plan: str = "",
    files: dict[str, str] | None = None,
    panel: list[str] | None = None,
    effort: str | None = None,
    max_cost_usd: float | None = None,
    max_input_chars: int | None = None,
) -> dict:
    """把目标拆成可执行、可审查的计划。

    Planner MVP 只返回结构化计划，不执行命令、不修改文件。它优先使用 planner_panel；
    未配置时回退 consult_panel，再回退 review panel。首版按 panel 顺序尝试模型，
    取第一个可解析计划作为结果，其余模型只作为失败回退，不做多模型 debate。
    """
    dd = _defaults_mod.apply(effort=effort)
    endpoint_ids = set((dd.get("endpoints") or {}).keys())
    raw_panel, panel_source = _resolve_planner_panel(panel, dd)
    panel_used = _normalize_panel(raw_panel, endpoint_ids, dd.get("endpoints"))
    route_info = _describe_model_routes(raw_panel, dd, panel_source=panel_source)
    cost_limit = max_cost_usd if max_cost_usd is not None else dd.get("planner_max_cost_usd")
    if cost_limit is None:
        cost_limit = dd.get("consult_max_cost_usd")
    if cost_limit is None:
        cost_limit = dd.get("max_cost_usd")
    input_limit = int(
        max_input_chars
        if max_input_chars is not None
        else dd.get("planner_max_input_chars", dd.get("consult_max_input_chars", 24000))
    )

    engine = _build_planner_engine(dd)
    report = await engine.plan(
        PlanRequest(
            goal=goal,
            context=context,
            constraints=constraints or [],
            success_criteria=success_criteria or [],
            existing_plan=existing_plan,
            files=files or {},
        ),
        panel=panel_used,
        max_input_chars=input_limit,
        max_cost_usd=cost_limit,
        effort=dd.get("effort"),
    )
    result = report.to_dict()
    result["routing"] = {
        "panel_source": panel_source,
        "resolved_panel": [entry["label"] for entry in panel_used],
        "model_routes": route_info["resolved_panel"],
        "route_warnings": route_info["warnings"],
        "ambiguous_models": route_info["ambiguous_models"],
        "strategy": "first_parseable_plan",
    }
    return result


@mcp.tool()
async def consult_problem(
    problem: str,
    context: str = "",
    files: dict[str, str] | None = None,
    logs: str = "",
    attempts: list[str] | None = None,
    goal: str = "",
    current_attempt: str = "",
    why_stuck: str = "",
    question: str = "",
    desired_output: str = "",
    constraints: list[str] | None = None,
    panel: list[str] | None = None,
    consultants: list[str] | None = None,
    mode: str | None = None,
    effort: str | None = None,
    max_cost_usd: float | None = None,
    max_input_chars: int | None = None,
) -> dict:
    """外援会诊：当主模型卡住、没把握、连续调试失败或需要第三方视角时调用。

    该工具只返回结构化建议，不执行命令、不修改文件。mode 可选 debugging/architecture/
    performance/simplicity/game_design/challenge/planning。发送给外部模型前会做基础敏感信息
    脱敏、输入长度上限控制和 consultant 白名单校验。panel None 时优先使用 consult_panel，
    未配置则回退 review panel；consultants None 时使用 consult_consultants。
    """
    dd = _defaults_mod.apply(effort=effort)
    endpoint_ids = set((dd.get("endpoints") or {}).keys())
    raw_panel, panel_source = _resolve_consult_panel(panel, dd)
    panel_used = _normalize_panel(raw_panel, endpoint_ids, dd.get("endpoints"))
    route_info = _describe_model_routes(raw_panel, dd, panel_source=panel_source)
    consultants_used, mode_used, consultants_source, mode_source = _resolve_consultants_with_source(consultants, mode, dd)
    cost_limit = max_cost_usd if max_cost_usd is not None else dd.get("consult_max_cost_usd")
    if cost_limit is None:
        cost_limit = dd.get("max_cost_usd")
    input_limit = int(max_input_chars if max_input_chars is not None else dd.get("consult_max_input_chars", 24000))

    # ContextProvider 召回(Phase 7 provider-loop):memory 自 scope(query.regions)/ git scopeless /
    # 未来 provider 注册即注入。memory_inject 门控(默认关 = 不注入)。loop **纯 merge**,无
    # if git/if memory 逻辑(GPT④ 不变量:server 不知 provider 内部,各 provider 自返 meta)。
    context_blocks: list = []
    providers_meta: dict = {}
    if dd.get("memory_inject"):
        anchor = "\n".join(x for x in (problem, context) if x)
        # Phase A selective context:memory 召回 scope 到 wake 激活的 region(∪ 全局),防跨项目 bleed。
        # 故意用 _route_regions(静态 trigger 路由)而非 full wake_gate —— 只需「该任务属于哪些 region」。
        # memory 自取 scope(memory/provider.py 优先级 _scope > query.regions);git scopeless 忽略 regions。
        routing = _route_regions(goal=goal, problem=problem, context=context, files=files or {})
        woken = {c["id"] for c in routing.get("selected", [])}
        regions = frozenset(woken) if dd.get("memory_scope", "woken") == "woken" else None
        query = _ContextQuery(text=anchor, regions=regions, top_k=int(dd.get("context_top_k", 5)))
        _ensure_default_providers()
        for name in _default_provider_registry.list_names():  # sorted()(context.py)→ 确定性顺序 git,memory
            try:  # review① per-provider 异常隔离:单 provider 崩不拖垮其余(memory 不被 git 拖垮)
                rr = _default_provider_registry.get(name).retrieve(query)
            except Exception as e:  # memory/git 内部已降级返空;此为意外兜底
                providers_meta[name] = {"provider": name, "available": False, "error": str(e)[:200]}
                continue
            context_blocks += rr.blocks
            providers_meta[name] = {"provider": rr.provider, **rr.meta}
    engine = _build_consult_engine(dd)
    report = await engine.consult(
        ConsultRequest(
            problem=problem,
            context=context,
            files=files or {},
            logs=logs,
            attempts=attempts or [],
            goal=goal,
            current_attempt=current_attempt,
            why_stuck=why_stuck,
            question=question,
            desired_output=desired_output,
            constraints=constraints or [],
        ),
        panel=panel_used,
        consultants=consultants_used,
        max_input_chars=input_limit,
        max_cost_usd=cost_limit,
        effort=dd.get("effort"),
        context_blocks=context_blocks,
    )
    result = report.to_dict()
    result["panel"] = [entry["label"] for entry in panel_used]
    result["consultants"] = list(consultants_used)
    result["mode"] = mode_used
    result["routing"] = {
        "panel_source": panel_source,
        "mode_source": mode_source,
        "consultants_source": consultants_source,
        "resolved_panel": [entry["label"] for entry in panel_used],
        "resolved_consultants": list(consultants_used),
        "model_routes": route_info["resolved_panel"],
        "route_warnings": route_info["warnings"],
        "ambiguous_models": route_info["ambiguous_models"],
    }
    result["context_providers"] = providers_meta
    result["memory"] = providers_meta.get("memory", {})  # review② 兼容别名(旧消费者不破;transitional)
    # 只记录 consult 元数据与 advice id，不记录 prompt/问题正文/advice 全文。
    reviews_db.record_consultation(result)
    return result


@mcp.tool()
def list_adapters() -> dict:
    """列出可用 ProjectAdapter + auto 检测结果。"""
    root = os.environ.get("UNITY_PROJECT_ROOT", ".")
    detected = "unity" if (Path(root) / "Packages" / "manifest.json").exists() else "generic"
    return {
        "adapters": [
            {"name": "unity", "desc": "Unity ECS（entities/netcode/physics）"},
            {"name": "generic", "desc": "通用（无项目特定，用 core 通用 reviewer）"},
        ],
        "auto_detected": detected,
    }


@mcp.tool()
def list_reviewers(adapter: str = "auto") -> dict:
    """列出可用 reviewer 角色（core 通用 + adapter 特定）。"""
    root = os.environ.get("UNITY_PROJECT_ROOT", ".")
    ad = _resolve_adapter(adapter, root)
    core = _list_reviewer_files(CORE_REVIEWERS_DIR)
    specific = _list_reviewer_files(ad.reviewers_dir()) if ad.reviewers_dir().exists() else []
    return {"adapter": ad.name, "core": core, "adapter_specific": specific}


@mcp.tool()
def list_consultants() -> dict:
    """列出可用外援会诊角色。"""
    return {"consultants": _list_consultant_files(CONSULTANTS_DIR)}


@mcp.tool()
def list_regions() -> dict:
    """List available Brain Regions."""
    regions = [region.to_dict() for region in _load_regions(REGIONS_DIR)]
    return {"regions": regions}


@mcp.tool()
def list_allowed_roots() -> dict:
    """List workspace roots that BrainRegion file tools may read."""
    return _list_allowed_roots()


@mcp.tool()
def inspect_file(path: str) -> dict:
    """Inspect an allowed workspace file without returning contents."""
    return _inspect_file(path)


@mcp.tool()
def read_text(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
    max_bytes: int = 64000,
) -> dict:
    """Read UTF-8 text from an allowed workspace file with line and byte limits."""
    return _read_text(path, start_line=start_line, end_line=end_line, max_bytes=max_bytes)


@mcp.tool()
def search_text(
    query: str,
    root: str = "",
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    case_sensitive: bool = False,
    regex: bool = False,
    max_results: int = 50,
    context_lines: int = 0,
    max_file_bytes: int = 1000000,
) -> dict:
    """Search UTF-8 text files inside allowed workspace roots."""
    return _search_text(
        query,
        root=root,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        case_sensitive=case_sensitive,
        regex=regex,
        max_results=max_results,
        context_lines=context_lines,
        max_file_bytes=max_file_bytes,
    )


@mcp.tool()
def apply_text_patch(
    path: str,
    expected_sha256: str,
    replacements: list[dict],
    dry_run: bool = True,
    max_diff_bytes: int = 128000,
) -> dict:
    """Apply exact UTF-8 text replacements with a required sha256 guard."""
    return _apply_text_patch(
        path,
        expected_sha256=expected_sha256,
        replacements=replacements,
        dry_run=dry_run,
        max_diff_bytes=max_diff_bytes,
    )

@mcp.tool()
def workspace_run_check(
    argv: list[str],
    cwd: str = "",
    timeout_sec: int = 60,
    max_output_chars: int = 20000,
) -> dict:
    """Run an allowed test/lint check command inside an allowed workspace root."""
    return _workspace_run_check(
        argv,
        cwd=cwd,
        timeout_sec=timeout_sec,
        max_output_chars=max_output_chars,
    )

def _normalize_experience_region(region: str | None) -> str:
    """Normalize MCP-facing Experience Memory regions to registered brain regions."""
    text = str(region or "").strip()
    if not text:
        return ""
    normalized = re.sub(r"[\s-]+", "_", text.casefold())
    available = {r.id for r in _load_regions(REGIONS_DIR)}
    if normalized in available:
        return normalized
    choices = ", ".join(["(global)", *sorted(available)])
    raise ValueError(f"unknown experience region: {text!r}; available: {choices}")


# 进程内短生命周期认知状态：重启清空，不写 Experience Memory。
_cognitive_workspace = _CognitiveWorkspace()
_region_coordination_board = _RegionCoordinationBoard()
_task_coordination_board = _TaskCoordinationBoard()

# ── Phase 4:Skill/Region Manifest + Registry(三级结构地基;list_skills = discovery surface)──
# lazy bootstrap(首次访问建):ProviderRegistry 注册 MemoryProvider → load skill YAML(校验 provider ref)
# → SkillRegistry。declares skills 但**不接 production routing**(status=experimental;与硬编码 map 共存)。
_skill_registry_singleton: _SkillRegistry | None = None


def _ensure_default_providers() -> None:
    """注册默认 ContextProvider(memory/git)到 default_provider_registry。

    idempotent + **分别 has() 判断**(review:不只判 git,防部分初始化时 memory 重复注册)。
    _skill_registry(校验 skill ref)与 consult(provider-loop)都调 —— 抽出解耦:provider 注册
    不再依赖 skill bootstrap。register 是 warn+overwrite upsert,分别 has() 避免无谓 warn。
    """
    if not _default_provider_registry.has("memory"):
        _default_provider_registry.register("memory", MemoryProvider.from_store())  # drift:与 server 内联 wiring 同源
    if not _default_provider_registry.has("git"):
        _default_provider_registry.register("git", GitProvider.from_repo())  # Phase 6:零成本注册(git 惰性跑)


def _skill_registry() -> _SkillRegistry:
    """Build-once SkillRegistry(模块级 lazy;bootstrap 顺序:provider 先于 skill YAML 校验)。"""
    global _skill_registry_singleton
    if _skill_registry_singleton is not None:
        return _skill_registry_singleton
    _ensure_default_providers()
    region_ids = {r.id for r in _load_regions(REGIONS_DIR)}
    manifests = _load_skills(
        _SKILLS_DIR,
        region_exists=lambda rid: rid in region_ids,
        provider_exists=lambda name: _default_provider_registry.has(name),
    )
    reg = _SkillRegistry()
    for m in manifests:
        reg.register(m)
    _skill_registry_singleton = reg
    return reg


@mcp.tool()
def list_skills(region: str | None = None) -> dict:
    """List registered Skill manifests(manifest-only, sanitized;status=experimental = 未接 production routing)。

    Phase 4 discovery surface(Router API 调用点);不泄露 body ref,不触发 resolve。
    """
    reg = _skill_registry()
    manifests = reg.by_region(region) if region else reg.all_manifests()
    return {"skills": [m.to_public_dict() for m in manifests], "count": len(manifests)}


@mcp.tool()
def create_task(
    task_id: str,
    goal: str,
    parent_task_id: str = "",
    success_criteria: list[str] | None = None,
    constraints: list[str] | None = None,
    status: str = "queued",
) -> dict:
    """Register a main-brain task before delegating independent expert work."""
    task = _task_coordination_board.create_task(
        {
            "task_id": task_id,
            "goal": goal,
            "parent_task_id": parent_task_id,
            "success_criteria": success_criteria or [],
            "constraints": constraints or [],
            "status": status,
        }
    )
    return {"task": task, "contains_context_content": False}


@mcp.tool()
def delegate_task(
    task_id: str,
    assignment_id: str,
    region: str,
    question: str,
    scope: str = "",
    depends_on: list[str] | None = None,
    memory_request: dict | None = None,
    expected_output: str = "region_report",
    status: str = "queued",
) -> dict:
    """Create one independent expert assignment with a directional memory request."""
    assignment = _task_coordination_board.delegate(
        task_id,
        {
            "assignment_id": assignment_id,
            "region": region,
            "question": question,
            "scope": scope,
            "depends_on": depends_on or [],
            "memory_request": memory_request or {},
            "expected_output": expected_output,
            "status": status,
        },
    )
    return {"assignment": assignment, "contains_context_content": False}


@mcp.tool()
def request_evidence_wake(
    task_id: str,
    assignment_id: str,
    reason: str,
    ttl_reads: int = 1,
) -> dict:
    """Wake selective evidence for one exact assignment for bounded provider reads.

    This records routing provenance only. It accepts no context body and provides
    architectural delivery isolation, not caller authentication.
    """
    wake = _task_coordination_board.request_evidence_wake(
        task_id,
        assignment_id,
        reason=reason,
        source="mcp_request",
        ttl_reads=ttl_reads,
    )
    return {
        "wake": wake,
        "contains_context_content": False,
        "authorization_boundary": False,
    }


@mcp.tool()
def task_status(task_id: str) -> dict:
    """Inspect task assignments plus public report counts and latest decisions."""
    status = _task_coordination_board.status(task_id)
    reports = _region_coordination_board.reports(task_id)["reports"]
    report_groups: dict[str, list[dict]] = {}
    for published in reports:
        assignment_id = published["report"].get("assignment_id", "")
        report_groups.setdefault(assignment_id, []).append(published)
    assignments: list[dict] = []
    for assignment in status["assignments"]:
        item = dict(assignment)
        published = report_groups.get(item["assignment_id"], [])
        item["report_count"] = len(published)
        item["latest_report"] = published[-1] if published else None
        assignments.append(item)
    return {
        **status,
        "assignments": assignments,
        "unassigned_report_count": len(report_groups.get("", [])),
        "contains_private_context": False,
    }


@mcp.tool()
def collect_reports(task_id: str, assignment_id: str = "") -> dict:
    """Collect validated independent RegionReports without private workspace blocks."""
    return _region_coordination_board.reports(
        task_id,
        assignment_id=assignment_id if assignment_id else None,
    )


@mcp.tool()
def plan_delegation_experiment(
    task_id: str,
    repeats: int = 1,
    arms: list[str] | None = None,
) -> dict:
    """Build a matched main-only/single/multi run plan without calling models."""
    from .eval.delegation import DelegationEvalTask, build_delegation_plan

    task = DelegationEvalTask.from_task_status(
        _task_coordination_board.status(task_id)
    )
    return build_delegation_plan(task, repeats=repeats, arms=arms)


@mcp.tool()
def summarize_delegation_experiment(
    records: list[dict],
    run_id: str = "",
    bootstrap_samples: int | None = None,
) -> dict:
    """Aggregate metric-only delegation records with task-level paired bootstrap."""
    from .eval.delegation import summarize_delegation_records

    return summarize_delegation_records(
        records,
        run_id=run_id,
        bootstrap_samples=bootstrap_samples,
    )


@mcp.tool()
def plan_region_activation(
    task_intents: list[str] | None = None,
    events: list[str] | None = None,
    target_apps: list[str] | None = None,
    running_apps: list[str] | None = None,
    available_tools: list[str] | None = None,
    available_capabilities: list[str] | None = None,
    cooldowns: dict[str, int] | None = None,
    attributes: dict | None = None,
    regions: list[str] | None = None,
    max_regions: int = 3,
    max_context_tokens: int = 4000,
) -> dict:
    """Plan structured Skill/Region wakes and bounded context requests.

    This hard gate is read-only and deterministic. It does not call models,
    retrieve memory, or execute tools. Ambiguous candidates return defer for
    a future cheap semantic gate; missing prerequisites and deny conditions
    always remain blocked.
    """
    signal = _ActivationSignal.from_dict({
        "task_intents": task_intents or [],
        "events": events or [],
        "target_apps": target_apps or [],
        "running_apps": running_apps or [],
        "available_tools": available_tools or [],
        "available_capabilities": available_capabilities or [],
        "cooldowns": cooldowns or {},
        "attributes": attributes or {},
    })
    return _skill_registry().plan_activation(
        signal,
        regions=regions,
        max_regions=max_regions,
        max_context_tokens=max_context_tokens,
    ).to_dict()


@mcp.tool()
def load_region_context(
    query: str,
    task_intents: list[str] | None = None,
    events: list[str] | None = None,
    target_apps: list[str] | None = None,
    running_apps: list[str] | None = None,
    available_tools: list[str] | None = None,
    available_capabilities: list[str] | None = None,
    cooldowns: dict[str, int] | None = None,
    attributes: dict | None = None,
    regions: list[str] | None = None,
    scope_regions: list[str] | None = None,
    top_k: int = 5,
    max_blocks: int = 12,
    max_regions: int = 3,
    max_context_tokens: int = 4000,
) -> dict:
    """Wake eligible provider Skills and load bounded, short-lived context.

    The operation retrieves existing ContextBlocks but never calls a model,
    writes memory, executes action Skills, or retains context in the runtime.
    Advisory/action activations remain visible in the returned plan and are
    skipped by this provider-only lifecycle step.
    """
    signal = _ActivationSignal.from_dict({
        "task_intents": task_intents or [],
        "events": events or [],
        "target_apps": target_apps or [],
        "running_apps": running_apps or [],
        "available_tools": available_tools or [],
        "available_capabilities": available_capabilities or [],
        "cooldowns": cooldowns or {},
        "attributes": attributes or {},
    })
    registry = _skill_registry()
    activation = registry.plan_activation(
        signal,
        regions=regions,
        max_regions=max_regions,
        max_context_tokens=max_context_tokens,
    )
    _ensure_default_providers()
    return _load_activation_context(
        activation,
        query_text=query,
        skill_registry=registry,
        resolvers=_setup_skill_resolvers(provider_registry=_default_provider_registry),
        scope_regions=frozenset(scope_regions) if scope_regions is not None else None,
        top_k=top_k,
        max_blocks=max_blocks,
    ).to_dict()


@mcp.tool()
def stage_region_context(
    task_id: str,
    query: str,
    audience: str = "region",
    target_region: str = "",
    assignment_id: str = "",
    ttl_steps: int = 3,
    task_intents: list[str] | None = None,
    events: list[str] | None = None,
    target_apps: list[str] | None = None,
    running_apps: list[str] | None = None,
    available_tools: list[str] | None = None,
    available_capabilities: list[str] | None = None,
    cooldowns: dict[str, int] | None = None,
    attributes: dict | None = None,
    regions: list[str] | None = None,
    scope_regions: list[str] | None = None,
    top_k: int = 5,
    max_blocks: int = 12,
    max_regions: int = 3,
    max_context_tokens: int = 4000,
) -> dict:
    """Load activated provider context into a task workspace without returning its contents.

    This is an architectural delivery boundary, not an authorization boundary.
    Region-private blocks remain hidden from main views, while the receipt exposes
    only routing metadata, evidence references, budgets, and TTL.
    """
    signal = _ActivationSignal.from_dict({
        "task_intents": task_intents or [],
        "events": events or [],
        "target_apps": target_apps or [],
        "running_apps": running_apps or [],
        "available_tools": available_tools or [],
        "available_capabilities": available_capabilities or [],
        "cooldowns": cooldowns or {},
        "attributes": attributes or {},
    })
    registry = _skill_registry()
    activation = registry.plan_activation(
        signal,
        regions=regions,
        max_regions=max_regions,
        max_context_tokens=max_context_tokens,
    )
    _ensure_default_providers()
    activated = _load_activation_context(
        activation,
        query_text=query,
        skill_registry=registry,
        resolvers=_setup_skill_resolvers(provider_registry=_default_provider_registry),
        scope_regions=frozenset(scope_regions) if scope_regions is not None else None,
        top_k=top_k,
        max_blocks=max_blocks,
    )
    delivery = _cognitive_workspace.stage(
        activated,
        task_id=task_id,
        audience=audience,
        target_region=target_region,
        assignment_id=assignment_id,
        ttl_steps=ttl_steps,
    )
    recipient = target_region if str(audience).strip().casefold() == "region" else audience
    evidence_refs = tuple((delivery.entry or {}).get("evidence_refs", ()))
    context_receipt = _RegionContextReceipt.from_activated(
        activated,
        task_id=task_id,
        region=recipient,
        evidence_refs=evidence_refs,
        assignment_id=assignment_id,
    )
    _region_coordination_board.record_receipt(context_receipt)
    return {
        "context_receipt": context_receipt.to_dict(),
        "task_id": task_id,
        "activation": activation.to_dict(),
        "loads": [load.to_dict() for load in activated.loads],
        "delivery": delivery.to_dict(),
        "trace": {
            **activated.trace,
            "strategy": "activation_context_stage_v1",
            "context_blocks_returned": 0,
        },
    }


@mcp.tool()
def workspace_context(
    task_id: str,
    operation: str = "read",
    consumer: str = "main",
    report: dict | None = None,
    region: str = "",
    assignment_id: str = "",
    steps: int = 1,
    max_context_tokens: int = 2000,
    max_blocks: int = 12,
) -> dict:
    """Read or manage task-scoped cognitive workspace context.

    operation is read, inspect, advance, clear, publish_report, status, or inbox.
    Read applies audience filtering; status/inbox never include private ContextBlocks.
    """
    operation = str(operation or "").strip().casefold()
    if operation == "read":
        return _cognitive_workspace.read(
            task_id,
            consumer=consumer,
            region=region,
            assignment_id=assignment_id,
            max_context_tokens=max_context_tokens,
            max_blocks=max_blocks,
        ).to_dict()
    if operation == "inspect":
        return _cognitive_workspace.inspect(task_id)
    if operation == "advance":
        return _cognitive_workspace.advance(task_id, steps=steps)
    if operation == "clear":
        context_result = _cognitive_workspace.clear(
            task_id, assignment_id=assignment_id
        )
        coordination_result = (
            _region_coordination_board.clear(
                task_id, assignment_id=assignment_id if assignment_id else None
            )
        )
        task_result = (
            _task_coordination_board.clear(task_id) if not assignment_id else {}
        )
        wake_result = (
            _task_coordination_board.clear_evidence_wakes(
                task_id, assignment_id=assignment_id
            )
            if assignment_id
            else {}
        )
        return {
            **context_result,
            **coordination_result,
            **task_result,
            **wake_result,
        }
    if operation == "publish_report":
        report_data = dict(report or {})
        if assignment_id:
            report_data["assignment_id"] = assignment_id
        return _region_coordination_board.publish(task_id, report_data)
    if operation == "status":
        return _region_coordination_board.status(task_id)
    if operation == "inbox":
        return _region_coordination_board.inbox(task_id)
    raise ValueError(
        "operation must be one of: read, inspect, advance, clear, "
        "publish_report, status, inbox"
    )


@mcp.tool()
async def run_region_expert(
    task_id: str,
    region: str,
    task: str,
    model: str,
    assignment_id: str = "",
    max_context_tokens: int = 2000,
    max_blocks: int = 12,
    max_tokens: int = 1200,
    temperature: float = 0.1,
    effort: str | None = None,
    max_cost_usd: float | None = None,
) -> dict:
    """Run one model as a focused region expert over its private workspace view.

    The model reference uses the same endpoint/model routing as consult and review.
    Only a validated RegionReport and escalation decision return to the caller;
    private ContextBlocks and raw model output never return.
    """
    dd = _defaults_mod.apply(effort=effort)
    endpoints_cfg = dd.get("endpoints") or {}
    entry = _normalize_one(model, set(endpoints_cfg), endpoints_cfg)
    endpoint_id = entry.get("endpoint_id")
    budget = {
        "max_usd": max_cost_usd,
        "estimated_usd": None,
        "exhausted": False,
    }
    export_policy = dd.get("context_export_policy")
    export_mode = _context_export_mode(export_policy)
    export_decision = _bypass_context_export()
    if export_mode != "off":
        expert_view = _cognitive_workspace.read(
            task_id,
            consumer="region",
            region=region,
            assignment_id=assignment_id,
            max_context_tokens=max_context_tokens,
            max_blocks=max_blocks,
        )
        endpoint_trust = _endpoint_context_trust(
            endpoint_id, endpoints_cfg, export_policy
        )
        export_decision = _evaluate_context_export(
            expert_view.blocks,
            policy=export_policy,
            endpoint_trust=endpoint_trust,
        )
        if not export_decision.permits_call:
            return {
                "ok": False,
                "task_id": task_id,
                "region": region,
                "assignment_id": assignment_id,
                "model": entry["model"],
                "endpoint_id": endpoint_id,
                "published_report": None,
                "context": {
                    "blocks_used": len(expert_view.blocks),
                    "estimated_tokens": int(expert_view.trace.get("estimated_tokens") or 0),
                    "private_context_returned": False,
                },
                "usage": {},
                "cost_usd": None,
                "cost_source": None,
                "parse_ok": False,
                "error": "context_export_denied: context exceeds endpoint trust",
                "model_called": False,
                "context_export": export_decision.to_dict(),
                "budget": budget,
                "routing": {
                    "requested_model": model,
                    "resolved_model": entry["model"],
                    "endpoint_id": endpoint_id,
                },
            }
    selected_endpoints = (
        {endpoint_id: endpoints_cfg[endpoint_id]} if endpoint_id is not None else {}
    )
    endpoint_registry = _resolve_endpoints(selected_endpoints)
    if max_cost_usd is not None:
        if isinstance(max_cost_usd, bool) or float(max_cost_usd) < 0:
            raise ValueError("max_cost_usd must be a non-negative number")
        estimate_job = {
            "model": entry["model"],
            "system": "region expert structured report",
            "user": str(task or "") + ("x" * max(0, int(max_context_tokens)) * 4),
            "max_tokens": max_tokens,
        }
        selected, estimated, exhausted = _select_jobs_within_budget(
            [estimate_job], float(max_cost_usd)
        )
        budget.update({"estimated_usd": estimated, "exhausted": exhausted})
        if not selected:
            return {
                "ok": False,
                "task_id": task_id,
                "region": region,
                "assignment_id": assignment_id,
                "model": entry["model"],
                "endpoint_id": entry.get("endpoint_id"),
                "published_report": None,
                "context": {"blocks_used": 0, "estimated_tokens": 0, "private_context_returned": False},
                "usage": {},
                "cost_usd": None,
                "cost_source": None,
                "parse_ok": False,
                "error": "budget_exceeded: expert model call skipped",
                "model_called": False,
                "context_export": export_decision.to_dict(),
                "budget": budget,
                "routing": {"requested_model": model, "resolved_model": entry["model"]},
            }
    engine = _build_region_expert_engine(dd, endpoint_registry)
    result = await engine.run(
        workspace=_cognitive_workspace,
        coordination=_region_coordination_board,
        task_id=task_id,
        region=region,
        task=task,
        model=entry["model"],
        assignment_id=assignment_id,
        endpoint_id=entry.get("endpoint_id"),
        max_context_tokens=max_context_tokens,
        max_blocks=max_blocks,
        max_tokens=max_tokens,
        temperature=temperature,
        effort=dd.get("effort"),
    )
    output = result.to_dict()
    output["assignment_id"] = assignment_id
    output["budget"] = budget
    output["context_export"] = export_decision.to_dict()
    output["routing"] = {
        "requested_model": model,
        "resolved_model": entry["model"],
        "endpoint_id": entry.get("endpoint_id"),
    }
    return output


@mcp.tool()
def route_regions(
    goal: str = "",
    problem: str = "",
    context: str = "",
    files: dict[str, str] | None = None,
    top_k: int = 3,
    min_score: int = 2,
) -> dict:
    """Recommend relevant Brain Regions from local deterministic rules.

    This tool is read-only: it does not call models, read memory, or trigger
    review/consult/planner tools. File contents are ignored; file paths are
    used only as weak metadata.
    """
    return _route_regions(
        goal=goal,
        problem=problem,
        context=context,
        files=files or {},
        top_k=top_k,
        min_score=min_score,
        regions_dir=REGIONS_DIR,
    )


@mcp.tool()
def suggest_workflow(
    goal: str = "",
    problem: str = "",
    context: str = "",
    files: dict[str, str] | None = None,
    top_k: int = 3,
    min_score: int = 2,
) -> dict:
    """Suggest explicit manual next tool calls from Brain Region routing.

    This tool is advisory only: it calls the local deterministic router, then
    returns candidate next actions such as plan_task, consult_problem,
    review_document, or review_code. It never calls those tools or models.
    """
    return _suggest_workflow(
        goal=goal,
        problem=problem,
        context=context,
        files=files or {},
        top_k=top_k,
        min_score=min_score,
        regions_dir=REGIONS_DIR,
    )


@mcp.tool()
def wake_gate(
    goal: str = "",
    problem: str = "",
    context: str = "",
    files: dict[str, str] | None = None,
    escalate_confidence: float = 0.5,
    shadow_wake_threshold: float | None = None,
    top_k: int = 3,
    sentinel: bool = True,
    shadow_top_n: int = 3,
    gold_regions: list[str] | None = None,
) -> dict:
    """Region-routing wake gate with false-negative defense (read-only sidecar).

    Routes Brain Regions through retrieve -> escalate -> wake, adding sentinel
    (cross-domain risk keywords) and shadow (near-threshold) fallback wakes to
    defend against missed wakes. Returns an activation trace, wake_metrics vs
    optional gold_regions (metrics_status scored/unscored), and suggested
    actions. Never calls models or downstream tools.
    """
    return _wake_gate(
        goal=goal,
        problem=problem,
        context=context,
        files=files or {},
        escalate_confidence=escalate_confidence,
        shadow_wake_threshold=shadow_wake_threshold,
        top_k=top_k,
        sentinel=sentinel,
        shadow_top_n=shadow_top_n,
        gold_regions=gold_regions,
        regions_dir=REGIONS_DIR,
    )


@mcp.tool()
def inspect(
    view: str = "all",
    goal: str = "",
    problem: str = "",
    context: str = "",
    files: dict[str, str] | None = None,
    gold_regions: list[str] | None = None,
    run_id: str = "",
    region: str = "",
    judge_id: str = "",
    escalate_confidence: float = 0.5,
    shadow_wake_threshold: float | None = None,
    top_k: int = 3,
    memory_preview_k: int = 3,
    memory_manifest: bool = False,
    history_limit: int = 20,
) -> dict:
    """只读调试窗口（v5.x）：把系统内部状态做成立即可见的可观测面。

    view ∈ {all, activation, memory, run, calibration}，只含请求的 section（all=全部 4）。
    - activation：重跑 wake_gate（无模型）看「该醒没醒」（给 gold_regions 才判漏唤醒）。
    - memory：Experience Memory 按 region 盘点 + 年龄。memory_manifest=True 附全量清单（Brain Diff 用；默认 False 精简）。
    - run：读历史 eval run 的已存 summary + per-task 5 态阶段时间线；无 run_id → 最近 N run 历史表。
    - calibration：judge 校准状态 + am-I-blocked。

    纯只读：不调模型、不写、不重算（wake_gate 已验为 read-only sidecar）。
    """
    from .inspector import inspect as _inspect_facade  # lazy：避免 inspector↔server 循环 import

    return _inspect_facade(
        view=view, goal=goal, problem=problem, context=context, files=files or {},
        gold_regions=gold_regions, run_id=run_id or None, region=region or None,
        judge_id=judge_id or None, escalate_confidence=escalate_confidence,
        shadow_wake_threshold=shadow_wake_threshold, top_k=top_k,
        memory_preview_k=memory_preview_k, memory_manifest=memory_manifest,
        history_limit=history_limit,
    )


@mcp.tool()
def snapshot(
    goal: str = "",
    problem: str = "",
    context: str = "",
    files: dict[str, str] | None = None,
    gold_regions: list[str] | None = None,
    run_id: str = "",
    region: str = "",
    judge_id: str = "",
    history_limit: int = 20,
    memory_preview_k: int = 5,
    top_k: int = 3,
    escalate_confidence: float = 0.5,
    shadow_wake_threshold: float | None = None,
) -> dict:
    """脑状态快照（可视化 Phase 1）：投影 Inspector → 可序列化 BrainSnapshot 数据。

    返回 snapshot.to_dict()（结构化数据，含 schema_version；可落盘后用 CLI `--from` 复渲染）。
    HTML 渲染走 CLI `brain-region snapshot`（自包含静态面板）。恒取 memory/run/calibration；
    仅当 problem 或 goal 非空才取 activation（无查询的空 wake 无意义）。纯只读：不调生成模型、不写。
    """
    from .viz import build_snapshot  # lazy：避免 viz↔server 循环 import

    return build_snapshot(
        goal=goal, problem=problem, context=context, files=files or {},
        gold_regions=gold_regions, run_id=run_id or None, region=region or None,
        judge_id=judge_id or None, history_limit=history_limit,
        memory_preview_k=memory_preview_k, top_k=top_k,
        escalate_confidence=escalate_confidence, shadow_wake_threshold=shadow_wake_threshold,
    ).to_dict()


@mcp.tool()
def list_knowledge(adapter: str = "auto") -> dict:
    """列出知识库案例索引（id/title/category/triggers）。"""
    root = os.environ.get("UNITY_PROJECT_ROOT", ".")
    ad = _resolve_adapter(adapter, root)
    knowledge = YamlKnowledgeProvider(_knowledge_dirs(ad))
    return {
        "adapter": ad.name,
        "cases": [
            {"id": c.id, "title": c.title, "category": c.category, "triggers": c.triggers}
            for c in knowledge.list_cases()
        ],
    }


@mcp.tool()
def list_defaults() -> dict:
    """列出三层默认值及来源（builtin/config/env）。"""
    return _defaults_mod.get_all()


@mcp.tool()
def list_model_routes(panel: list[str] | None = None) -> dict:
    """Show how model specs resolve to official providers or configured endpoints.

    This is a diagnostic tool only: it does not call models and never returns
    API key values. It helps distinguish bare model strings like
    ``claude-opus-4-8`` from endpoint refs like
    ``modelbridge_anthropic/claude-opus-4-8``.
    """
    all_defaults = _defaults_mod.get_all()
    defaults = {key: value["value"] for key, value in all_defaults.items()}
    panel_source = "explicit" if panel is not None else all_defaults.get("panel", {}).get("source", "unknown")
    return _describe_model_routes(panel, defaults, panel_source=panel_source)


@mcp.tool()
def suggest_panel(
    strategy: str = "balanced",
    task: str = "",
    panel: list[str] | None = None,
    max_models: int = 2,
    require_available_key: bool = True,
) -> dict:
    """Recommend a model panel from route/profile metadata without calling models.

    Strategies include balanced, cheap_fast, best_reasoning, sleep, awake, and
    structured_output. The returned selected_panel can be copied into tools
    such as plan_task or consult_problem when the user chooses to spend tokens.
    """
    all_defaults = _defaults_mod.get_all()
    defaults = {key: value["value"] for key, value in all_defaults.items()}
    return _suggest_panel(
        defaults=defaults,
        strategy=strategy,
        task=task,
        panel=panel,
        max_models=max_models,
        require_available_key=require_available_key,
    )


@mcp.tool()
def panel_stats() -> dict:
    """缓存统计：审查总数 + 缓存命中省掉的重复审查数。"""
    return {**reviews_db.stats(), **reviews_db.advice_feedback_stats()}


def main() -> None:
    """MCP server 入口（默认 stdio transport）。"""
    from . import __version__

    logger.info("brainregion %s starting (stdio)", __version__)
    mcp.run()


if __name__ == "__main__":
    main()
