"""Renderer 协议 + 分发(可视化 Phase 1)。

`Renderer.render(snapshot) -> str`:把 BrainSnapshot 渲染成某种文本格式。Phase 1 唯一实现是
HtmlRenderer。**不包装 RenderResult**(mime/extension/body)——单 renderer 返回 str 足够;
多格式(SVG/PNG)是 Phase 2+,届时改返回类型是局部小改,不为未到的需求提前抽象(YAGNI)。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .diff import BrainDiff
from .html import DiffHtmlRenderer, HtmlRenderer
from .snapshot import BrainSnapshot


@runtime_checkable
class Renderer(Protocol):
    def render(self, snapshot: BrainSnapshot) -> str: ...


_RENDERERS = {"html": HtmlRenderer}
_DIFF_RENDERERS = {"html": DiffHtmlRenderer}


def render(snapshot: BrainSnapshot, fmt: str = "html") -> str:
    """按 fmt 选 renderer 渲染 snapshot → str。未知 fmt → ValueError。"""
    cls = _RENDERERS.get(fmt)
    if cls is None:
        raise ValueError(f"unknown render format: {fmt!r}; expected one of {list(_RENDERERS)}")
    return cls().render(snapshot)


def render_html(snapshot: BrainSnapshot) -> str:
    """便捷:HTML 渲染。"""
    return HtmlRenderer().render(snapshot)


def render_diff(diff: BrainDiff, fmt: str = "html") -> str:
    """按 fmt 选 renderer 渲染 BrainDiff → str。未知 fmt → ValueError。"""
    cls = _DIFF_RENDERERS.get(fmt)
    if cls is None:
        raise ValueError(f"unknown render format: {fmt!r}; expected one of {list(_DIFF_RENDERERS)}")
    return cls().render(diff)
