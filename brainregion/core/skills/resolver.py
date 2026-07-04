"""Resolver —— body 解析策略(open-closed:加 kind = 注册新 Resolver,不改 dispatch)。

替代 ``if kind == ...`` elif 链(review/GPT#3)。镜像代码库既有模式:eval ``Family.apply`` 多态、
``KnowledgeProvider`` Protocol。本期仅 ``ProviderResolver``;加 ConsultantResolver/ToolResolver = 注册,
不改 ``resolve_skill_body``。

错误**结构化**(非裸 NotImplementedError):``UnsupportedSkillKind``(kind 无 resolver)、
``UnknownProvider``(provider ref 未注册)。
"""
from __future__ import annotations

from typing import Protocol

from ..context import ContextQuery, ProviderRegistry, RetrieveResult
from .manifest import SkillManifest


class ResolverError(Exception):
    """结构化 skill body 解析错误(带 skill_id/kind/ref 上下文,便于定位)。"""

    def __init__(
        self, kind: str, message: str, *,
        skill_id: str = "", ref: str = "", available: list[str] | None = None,
    ) -> None:
        self.kind = kind
        self.skill_id = skill_id
        self.ref = ref
        self.available = available or []
        super().__init__(message)


class UnsupportedSkillKind(ResolverError):
    """manifest.kind 无注册 Resolver(本期 consultant/tool/role = 仅可 manifest 注册、不可 resolve)。"""


class UnknownProvider(ResolverError):
    """kind=provider 的 ref 在 ProviderRegistry 中未注册。"""


class Resolver(Protocol):
    """body 解析策略。``resolve`` 把 manifest 的 ``ref`` 解析为实际 body 执行并返 RetrieveResult。"""

    kind: str

    def resolve(self, manifest: SkillManifest, query: ContextQuery) -> RetrieveResult: ...


class ProviderResolver:
    """kind=provider → ProviderRegistry.get(ref).retrieve(query)。"""

    kind = "provider"

    def __init__(self, provider_registry: ProviderRegistry) -> None:
        self._providers = provider_registry

    def resolve(self, manifest: SkillManifest, query: ContextQuery) -> RetrieveResult:
        provider = self._providers.get(manifest.ref)
        if provider is None:
            raise UnknownProvider(
                "provider",
                f"provider {manifest.ref!r} not registered (skill {manifest.id!r})",
                skill_id=manifest.id, ref=manifest.ref,
                available=self._providers.list_names(),
            )
        return provider.retrieve(query)


def setup_resolvers(*, provider_registry: ProviderRegistry) -> dict[str, Resolver]:
    """注册默认 resolvers(本期仅 provider)。返 kind→Resolver dict,供 resolve_skill_body + list_skills 持有。"""
    return {"provider": ProviderResolver(provider_registry)}


def resolve_skill_body(
    manifest: SkillManifest, query: ContextQuery, *, resolvers: dict[str, Resolver],
) -> RetrieveResult:
    """按 manifest.kind 查 resolver 并执行。无 resolver → ``UnsupportedSkillKind``(结构化)。"""
    resolver = resolvers.get(manifest.kind)
    if resolver is None:
        raise UnsupportedSkillKind(
            manifest.kind,
            f"no resolver for kind {manifest.kind!r} (skill {manifest.id!r}); "
            f"this phase only supports provider",
            skill_id=manifest.id,
        )
    return resolver.resolve(manifest, query)
