"""可视化包(Phase 1):BrainSnapshot 数据 + Renderer 协议 + HTML 渲染。

facade:``build_snapshot()`` 投影 Inspector → BrainSnapshot;``render_html()`` 渲染。
``BrainSnapshot.to_dict()/from_dict()`` 是序列化方法(snapshot 可落盘/复渲染)。
镜像 inspector/ 的 facade 结构(1 MCP + 1 CLI 都走这里)。
"""
from __future__ import annotations

from .diff import BrainDiff, ExperienceChange, build_diff
from .render import DiffHtmlRenderer, HtmlRenderer, Renderer, render, render_diff, render_html
from .snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    BrainSnapshot,
    Kpi,
    ManifestExperience,
    RegionSnapshot,
    build_snapshot,
)

__all__ = [
    "SNAPSHOT_SCHEMA_VERSION",
    "BrainSnapshot",
    "Kpi",
    "ManifestExperience",
    "RegionSnapshot",
    "BrainDiff",
    "ExperienceChange",
    "Renderer",
    "HtmlRenderer",
    "DiffHtmlRenderer",
    "build_snapshot",
    "build_diff",
    "render",
    "render_html",
    "render_diff",
]
