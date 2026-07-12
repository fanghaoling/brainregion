"""SkillManifest —— Skill 元数据(三级结构 Region→Skill→Parameter 的 Skill 层)。

manifest = **廉价路由元数据**(供 cross-region LLM router 读 description/tags),**separate from body**
(body = provider/consultant/tool 的实际执行,经 Resolver 按 kind 分发,见 resolver.py)。

设计(review_plan + GPT refine):
- 结构化核心字段 + ``metadata: dict`` **开放扩展**(未来 input_schema/cost/latency/capability 入此,防 breaking change)。
- ``status``(experimental|beta|stable)= 持久 lifecycle 字段(替短命 migration flag ``routed``)。
- ``to_public_dict`` sanitized(屏蔽 ``ref``)——MCP 输出 + ``manifests_for_router`` 共用,不泄内部 body 引用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from brainregion.core.activation import ActivationContract

SkillKind = Literal["provider", "consultant", "tool", "role"]
SkillStatus = Literal["experimental", "beta", "stable"]

_VALID_KINDS: tuple[str, ...] = ("provider", "consultant", "tool", "role")
_VALID_STATUS: tuple[str, ...] = ("experimental", "beta", "stable")


@dataclass(frozen=True)
class SkillManifest:
    """Skill 元数据(dumb data;body 解析在 Resolver,不在 manifest)。"""

    id: str
    name: str
    region: str                                       # 绑定 Region(= 三级结构的 Region 层)
    kind: SkillKind                                   # body 类型 → Resolver 分发
    description: str = ""                             # cross-region LLM router 读这
    tags: tuple[str, ...] = ()                        # 索引/匹配(frozen → tuple)
    version: dict[str, Any] = field(default_factory=dict)
    ref: str = ""                                     # body 引用(provider_name/consultant_id/tool_name);to_public_dict 屏蔽
    status: SkillStatus = "experimental"              # lifecycle(experimental=manifest-only 未接 production routing)
    metadata: dict[str, Any] = field(default_factory=dict)   # 开放扩展(防 breaking change)

    def activation_contract(self) -> ActivationContract | None:
        """Parse optional typed activation metadata; None keeps legacy manifests inert."""
        raw = self.metadata.get("activation")
        if raw is None:
            return None
        return ActivationContract.from_dict(skill_id=self.id, region=self.region, data=raw)

    def to_public_dict(self) -> dict[str, Any]:
        """sanitized:屏蔽 ``ref``(body 内部引用)。MCP 输出 + manifests_for_router 共用此 serializer。"""
        return {
            "id": self.id,
            "name": self.name,
            "region": self.region,
            "kind": self.kind,
            "description": self.description,
            "tags": list(self.tags),
            "version": dict(self.version),
            "status": self.status,
            "metadata": dict(self.metadata),
        }
