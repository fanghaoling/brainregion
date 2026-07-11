"""LiteLLMBackend：默认且 v1 唯一内置的 ModelBackend。

litellm 1.89.x 内置 tenacity 重试（429/5xx/网络异常 + 尊重 Retry-After），所以**省掉
asset-generator-mcp 的 _http.py**。httpx 直连，不需单独装 openai/anthropic/google-genai SDK。

⚠️ 供应链安全：pyproject 已 pin `litellm>=1.83.0,<2.0`（1.82.7/1.82.8 被投毒）。

强制 JSON：统一 `response_format={"type":"json_object"}`（国产严格 json_schema 不可靠，
json_object + prompt 贴 schema 范例 + parsing 防御解析）。调用方可通过构造参数覆盖。

`complete`（system+user 两段）服务 review/consult/capability 等单次调用；
`complete_messages`（完整 messages 列表）服务沙盒 agent loop——跨步携带对话历史 + tool-result。
两者共用 `_acomplete`（endpoint 解析 / 采样 / effort / 事件 / 失败隔离）。
"""
from __future__ import annotations

import logging
import re
import time

import litellm

from brainregion.runtime import emit_event
from brainregion.runtime.pricing import model_usage_payload

from .base import ModelResponse

litellm.suppress_debug_info = True  # 抑制 litellm stdout banner（CLI/MCP stdout 要纯 JSON/JSON-RPC）

logger = logging.getLogger("brainregion.provider.litellm")


def _effort_kwargs(model: str, effort: str | None, thinking: bool | None = None) -> dict:
    """把 effort + thinking 开关映射成 provider 特定参数。

    - **DeepSeek**(v4-flash/pro):思考模式开关 `extra_body={"thinking":{"type":"disabled|enabled"}}`
      (默认 enabled;关掉 = 便宜快的非推理模型,沙盒主脑用);思考开时 effort 走 `reasoning_effort`
      (low/medium→high、xhigh→max,见 DeepSeek 文档)。`thinking=None` 保持原行为(默认开,不显式传)。
    - Claude（4.6+）：effort 在 output_config；配 thinking adaptive 让思考生效（Opus 4.7/4.8 默认关思考）。
    - OpenAI o 系列：reasoning_effort。
    - 其余（gpt-4o/glm 等非推理模型）：不传,litellm drop_params 也不会报错。
    """
    short = model.split("/")[-1]
    if "deepseek" in model:
        if thinking is None:
            # 保持原契约:effort 对 deepseek no-op(§15.6:别用 effort 压 deepseek 思考,
            # 会抹 reasoning-cost 信号 + 制造 budget-starvation 假退化)。思考开关是独立的新能力。
            return {}
        if thinking is False:
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        kw: dict = {"extra_body": {"thinking": {"type": "enabled"}}}
        if effort:
            kw["reasoning_effort"] = effort
        return kw
    if "claude" in model:
        if not effort:
            return {}
        return {
            "thinking": {"type": "adaptive"},
            "extra_body": {"output_config": {"effort": effort}},
        }
    if re.match(r"o[1-9]", short):  # o1/o3/o4/o5 系列
        return {"reasoning_effort": effort} if effort else {}
    return {}


def _is_json_format_rejection(exc: Exception) -> bool:
    """provider 拒绝 response_format=json_object？命中则回退纯文本（靠 extract_json_object 解析）。"""
    msg = str(exc).lower()
    if "response_format" in msg:
        return True
    return "json" in msg and "format" in msg


class LiteLLMBackend:
    """基于 litellm 的 ModelBackend 实现。

    litellm 延迟 import（在 _acomplete 内），避免 server 启动时加载重依赖、且让"不用 litellm
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

    def _resolve_endpoint(self, model: str, endpoint_id: str | None) -> tuple[str, dict, float, str | None]:
        """返回 (litellm_model, ep_kwargs, call_timeout, provider_for_event)。"""
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
        call_timeout = ep.get("timeout") if ep and ep.get("timeout") else self.timeout
        provider_for_event = (ep or {}).get("provider")
        return litellm_model, ep_kwargs, call_timeout, provider_for_event

    @staticmethod
    def _sampling_for(litellm_model: str, temperature: float, top_p: float, effort: str | None, thinking: bool | None) -> dict:
        """OpenAI 推理模型（o 系列 + gpt-5 系列）不支持 temperature/top_p；anthropic effort 时 temp=1。

        DeepSeek 思考开(默认 enabled)也不支持 temp/top_p(文档:设了不报错但不生效);思考关 → 正常采样。
        """
        short = litellm_model.split("/")[-1]
        is_reasoning = bool(re.match(r"(?:o[1-9]|gpt-5)", short))
        is_anthropic = litellm_model.startswith("anthropic/") or "claude" in short
        # deepseek 思考【显式开】(thinking=True)→ 忽略 temp/top_p(文档);None=保持旧行为(发送),False=思考关支持采样。
        is_deepseek_thinking_on = "deepseek" in litellm_model and thinking is True
        if is_reasoning or is_deepseek_thinking_on:
            return {}
        if is_anthropic:
            return {"temperature": 1 if effort else temperature}
        return {"temperature": temperature, "top_p": top_p}

    async def _acompletion_with_fallback(
        self,
        messages: list[dict],
        *,
        litellm_model: str,
        sampling: dict,
        max_tokens: int,
        call_timeout: float,
        ep_kwargs: dict,
        effort: str | None,
        thinking: bool | None,
    ) -> object:
        """litellm.acompletion + json_object 回退：provider 拒 json_object 时去 response_format 重试。"""
        import litellm

        litellm.drop_params = True
        base_kwargs = dict(
            model=litellm_model,
            messages=messages,
            num_retries=self.num_retries,
            timeout=call_timeout,
            max_tokens=max_tokens,
            **sampling,
            **_effort_kwargs(litellm_model, effort, thinking),
            **{k: v for k, v in ep_kwargs.items() if v is not None},
        )
        try:
            return await litellm.acompletion(response_format=self.response_format, **base_kwargs)
        except Exception as e:  # noqa: BLE001
            if self.response_format and _is_json_format_rejection(e):
                logger.info("response_format 被拒，回退纯文本 model=%s", litellm_model)
                return await litellm.acompletion(**base_kwargs)
            raise

    async def _acomplete(
        self,
        *,
        messages: list[dict],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        effort: str | None,
        endpoint_id: str | None,
        thinking: bool | None = None,
    ) -> ModelResponse:
        litellm_model, ep_kwargs, call_timeout, provider_for_event = self._resolve_endpoint(model, endpoint_id)
        sampling = self._sampling_for(litellm_model, temperature, top_p, effort, thinking)

        emit_event(
            "model.call_started",
            model=model,
            provider=provider_for_event,
            payload={
                "resolved_model": litellm_model,
                "endpoint_id": endpoint_id,
                "effort": effort,
                "thinking": thinking,
                "max_tokens": max_tokens,
                "timeout": call_timeout,
            },
        )
        started = time.perf_counter()
        try:
            resp = await self._acompletion_with_fallback(
                messages,
                litellm_model=litellm_model,
                sampling=sampling,
                max_tokens=max_tokens,
                call_timeout=call_timeout,
                ep_kwargs=ep_kwargs,
                effort=effort,
                thinking=thinking,
            )
            usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
            hp = getattr(resp, "_hidden_params", None) or {}
            content = resp.choices[0].message.content or ""
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            cost_usd = hp.get("response_cost")
            usage_payload = model_usage_payload(
                provider=provider_for_event,
                model=model,
                resolved_model=litellm_model,
                endpoint_id=endpoint_id,
                usage=usage,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                status="ok",
            )
            emit_event(
                "model.call_finished",
                model=model,
                provider=provider_for_event,
                payload=usage_payload,
            )
            return ModelResponse(
                model=model,
                content=content,
                usage=usage,
                cost_usd=usage_payload["cost_usd"],
            )
        except Exception as e:  # noqa: BLE001 — 失败隔离，不向上抛
            logger.warning("LiteLLMBackend 调用失败 model=%s: %s: %s", model, type(e).__name__, e)
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            error = f"{type(e).__name__}: {e}"
            emit_event(
                "model.call_failed",
                model=model,
                provider=provider_for_event,
                payload=model_usage_payload(
                    provider=provider_for_event,
                    model=model,
                    resolved_model=litellm_model,
                    endpoint_id=endpoint_id,
                    usage={},
                    cost_usd=None,
                    latency_ms=latency_ms,
                    status="error",
                    error=error,
                ),
            )
            return ModelResponse(
                model=model,
                content="",
                error=error,
            )

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
        thinking: bool | None = None,
    ) -> ModelResponse:
        """单次调用（system+user 两段）。review/consult/capability 等用。"""
        return await self._acomplete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            effort=effort,
            endpoint_id=endpoint_id,
            thinking=thinking,
        )

    async def complete_messages(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float = 0.3,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        effort: str | None = None,
        endpoint_id: str | None = None,
        thinking: bool | None = None,
    ) -> ModelResponse:
        """带历史的完整 messages 列表调用。沙盒 agent loop 用（跨步携带对话 + tool-result）。"""
        return await self._acomplete(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            effort=effort,
            endpoint_id=endpoint_id,
            thinking=thinking,
        )
