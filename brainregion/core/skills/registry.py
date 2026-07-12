"""SkillRegistry —— Skill manifest 内存索引。

startup 从 YAML 建(``load_skills``)+ programmatic ``register``(镜像 eval ``register_family`` 模式,
供 eval/测试注册合成 skill)。``register`` id 冲突 **raise**(不静默覆盖,防第二真相源)。
"""
from __future__ import annotations

from brainregion.core.activation import (
    ActivationContract,
    ActivationPlan,
    ActivationSignal,
    plan_activation,
)

from .manifest import SkillManifest


class SkillRegistry:
    """Skill manifest 内存索引(id → manifest)。"""

    def __init__(self) -> None:
        self._by_id: dict[str, SkillManifest] = {}

    def register(self, manifest: SkillManifest) -> None:
        if manifest.id in self._by_id:
            raise ValueError(f"duplicate skill id {manifest.id!r} (already registered)")
        self._by_id[manifest.id] = manifest

    def get(self, skill_id: str) -> SkillManifest | None:
        return self._by_id.get(skill_id)

    def has(self, skill_id: str) -> bool:
        return skill_id in self._by_id

    def by_region(self, region_id: str) -> list[SkillManifest]:
        return [m for m in self._by_id.values() if m.region == region_id]

    def all_manifests(self) -> list[SkillManifest]:
        return list(self._by_id.values())

    def manifests_for_router(self, regions: list[str] | None = None) -> list[dict]:
        """router 入参:**sanitized** manifests(``to_public_dict``,屏蔽 ref = body 内部引用)。
        ``regions`` 给定则只返这些 region 的 manifest(跨区域路由候选集)。"""
        wanted = set(regions) if regions is not None else None
        ms = [m for m in self._by_id.values() if wanted is None or m.region in wanted]
        return [m.to_public_dict() for m in ms]

    def activation_contracts(self, regions: list[str] | None = None) -> list[ActivationContract]:
        """Return typed contracts from this registry; manifests without one stay asleep."""
        wanted = set(regions) if regions is not None else None
        out = []
        for manifest in self._by_id.values():
            if wanted is not None and manifest.region not in wanted:
                continue
            contract = manifest.activation_contract()
            if contract is not None:
                out.append(contract)
        return out

    def plan_activation(
        self,
        signal: ActivationSignal,
        *,
        regions: list[str] | None = None,
        max_regions: int = 3,
        max_context_tokens: int = 4000,
    ) -> ActivationPlan:
        """Plan explicit activation over this registry without introducing another registry."""
        return plan_activation(
            self.activation_contracts(regions),
            signal,
            max_regions=max_regions,
            max_context_tokens=max_context_tokens,
        )

    def __len__(self) -> int:
        return len(self._by_id)
