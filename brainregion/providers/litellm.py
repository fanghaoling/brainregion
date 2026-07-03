"""LiteLLMBackend：默认且 v1 唯一内置的 ModelBackend。

litellm 1.89.x 内置 tenacity 重试（429/5xx/网络异常 + 尊重 Retry-After），所以**省掉
asset-generator-mcp 的 _http.py**。httpx 直连，不需单独装 openai/anthropic/google-genai SDK。

⚠️ 供应链安全：pyproject 已 pin `litellm>=1.83.0,<2.0`（1.82.7/1.82.8 被投毒）。

强制 JSON：统一 `response_format={"type":"json_object"}`（国产严格 json_schema 不可靠，
json_object + prompt 贴 schema 范例 + parsing 防御解析）。调用方可通过构造参数覆盖。
"""
from __future__ import annotations

import logging
import re

import litellm

from .base import ModelResponse

litellm.suppress_debug_info = True  # 抑制 litellm stdout banner（CLI/MCP stdout 要纯 JSON/JSON-RPC）

logger = logging.getLogger("brainregion.provider.litellm")


def _effort_kwargs(model: str, effort: str | None) -> dict:
    """把 effort（low/medium/high/xhigh/max）映射成 provider 特定参数。

    - Claude（4.6+）：effort 在 output_config；配 thinking adaptive 让思考生效（Opus 4.7/4.8 默认关思考）
    - OpenAI o 系列：reasoning_effort
    - 其余（gpt-4o/glm/deepseek 等非推理模型）：不传，litellm drop_params 也不会报错
    """
    if not effort:
        return {}
    short = model.split("/")[-1]
    if "claude" in model:
        return {
            "thinking": {"type": "adaptive"},
            "extra_body": {"output_config": {"effort": effort}},
        }
    if re.match(r"o[1-9]", short):  # o1/o3/o4/o5 系列
        return {"reasoning_effort": effort}
    return {}


class LiteLLMBackend:
    """基于 litellm 的 ModelBackend 实现。

    litellm 延迟 import（在 complete 内），避免 server 启动时加载重依赖、且让"不用 litellm
    的自定义 backend"场景不必装 litellm。
    """

    def __init__(
        self,
        *,
        num_retries: int = 4,
        timeout: float = 60.0,
        response_format: dict | None = None,
        endpoint_registry: dict | None = None,
    ) -> None:
        self.num_retries = num_retries
        self.timeout = timeout
        # 默认强制 JSON 输出（国产严格 schema 不可靠，用 json_object + 防御解析）
        self.response_format = (
            response_format if response_format is not None else {"type": "json_object"}
        )
        # v1.6：endpoint_id -> EndpointConfig{provider, base_url, api_key, headers, timeout}。
        # credential 只存活在 backend 边缘（调用时查 registry），不进 PipelineContext。
        self.endpoint_registry = endpoint_registry or {}

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.3,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        effort: str | None = None,
        endpoint_id: str | None = None,
    ) -> ModelResponse:
        import litellm  # 延迟 import

        # 自动丢弃 provider 不支持的参数（zai/volcengine/anthropic 兼容端不支持 response_format）。
        # 国产模型靠 prompt 强制 JSON + ParseStage 防御解析兜底。
        litellm.drop_params = True

        # v1.6：中转站/自定义 endpoint。endpoint_id 查 registry 取 credential（key 不进 pipeline）。
        # provider 决定 litellm model 前缀（openai/anthropic 是兼容网关协议）；endpoint_id=None 走官方 env。
        ep = self.endpoint_registry.get(endpoint_id) if endpoint_id else None
        litellm_model = model
        ep_kwargs: dict = {}
        if ep:
            provider = ep.get("provider")
            # 前缀守卫：model 已带本 provider 前缀（用户误写 openai/x）则不再拼，防 openai/openai/。
            # 用 startswith(<provider>/) 而非"含 /"判断——中转站模型名本身可含 /（如 SiliconFlow 的
            # Qwen/Qwen3-8B），旧启发式 "/" not in model 会误判这类合法 ID 为"已加前缀"而漏拼 → litellm 不认 provider。
            if provider in ("openai", "anthropic") and not model.startswith(f"{provider}/"):
                litellm_model = f"{provider}/{model}"
            if ep.get("base_url"):
                ep_kwargs["api_base"] = ep["base_url"]  # snake_case！勿用 base_url（有历史 bug）
            if ep.get("api_key"):
                ep_kwargs["api_key"] = ep["api_key"]
            if ep.get("headers"):
                ep_kwargs["extra_headers"] = ep["headers"]
        # endpoint timeout 覆盖全局（慢中转站）
        call_timeout = ep.get("timeout") if ep and ep.get("timeout") else self.timeout

        # OpenAI 推理模型（o 系列 + gpt-5 系列）不支持 temperature/top_p（只支持默认 1），传了报 400；
        # litellm drop_params 兜不住，按模型名显式跳过采样参数。
        short = litellm_model.split("/")[-1]
        is_reasoning = bool(re.match(r"(?:o[1-9]|gpt-5)", short))
        is_anthropic = litellm_model.startswith("anthropic/") or "claude" in short
        if is_reasoning:
            sampling = {}
        elif is_anthropic:
            sampling = {"temperature": 1 if effort else temperature}
        else:
            sampling = {"temperature": temperature, "top_p": top_p}

        try:
            resp = await litellm.acompletion(
                model=litellm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                num_retries=self.num_retries,
                timeout=call_timeout,
                max_tokens=max_tokens,
                response_format=self.response_format,
                **sampling,
                **_effort_kwargs(litellm_model, effort),
                **{k: v for k, v in ep_kwargs.items() if v is not None},
            )
            usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
            hp = getattr(resp, "_hidden_params", None) or {}
            content = resp.choices[0].message.content or ""
            return ModelResponse(
                model=model,
                content=content,
                usage=usage,
                cost_usd=hp.get("response_cost"),
            )
        except Exception as e:  # noqa: BLE001 — 失败隔离，不向上抛
            logger.warning("LiteLLMBackend 调用失败 model=%s: %s: %s", model, type(e).__name__, e)
            return ModelResponse(
                model=model,
                content="",
                error=f"{type(e).__name__}: {e}",
            )
