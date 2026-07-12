"""ModelBackend 协议 + ModelResponse。

ModelBackend 抽象的是**调用层实现**（litellm vs 直连 SDK vs 自建网关），**不重复 litellm
的多 provider 能力**——换 OpenAI/Azure/OpenRouter/豆包只改 litellm model 字符串。协议仅给
"完全不用 litellm"的人留接入点。

失败隔离策略：backend.complete 内部 catch 所有异常，返回 ModelResponse(error=...)，
不向上抛。这样 ReviewStage 的 asyncio.gather 不会被单个模型失败打断。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ModelResponse:
    """一次 LLM 调用的结果（成功或失败隔离后）。

    Attributes:
        model: 模型字符串（litellm 约定）。
        content: 模型输出文本（失败时为空）。
        usage: token 用量 {prompt_tokens, completion_tokens, total_tokens}。
        cost_usd: provider 返回或本地价格表估算的 USD 成本。
        cost_source: 成本来源（provider/config/builtin/missing_price）。
        error: 失败原因（None=成功）。backend 内部已隔离，不向上抛。
    """

    model: str
    content: str = ""
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None
    cost_source: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@runtime_checkable
class ModelBackend(Protocol):
    """调用层实现协议。"""

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
    ) -> ModelResponse: ...
