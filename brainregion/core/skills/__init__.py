"""Skill/Region Manifest + Registry(三级结构 Region→Skill→Parameter 的 Skill 层地基;Phase 4)。

服务 §15.7:Skill 成一级实体(manifest = 廉价路由元数据,separate from body),为 Router API铺地基。
"""
from __future__ import annotations

from .loader import SKILLS_DIR, list_skills, load_skill, load_skills
from .manifest import SkillManifest, SkillKind, SkillStatus
from .registry import SkillRegistry
from .resolver import (
    ProviderResolver,
    Resolver,
    ResolverError,
    UnsupportedSkillKind,
    UnknownProvider,
    resolve_skill_body,
    setup_resolvers,
)

__all__ = [
    "SKILLS_DIR",
    "SkillManifest",
    "SkillKind",
    "SkillStatus",
    "SkillRegistry",
    "Resolver",
    "ProviderResolver",
    "ResolverError",
    "UnsupportedSkillKind",
    "UnknownProvider",
    "resolve_skill_body",
    "setup_resolvers",
    "list_skills",
    "load_skill",
    "load_skills",
]
