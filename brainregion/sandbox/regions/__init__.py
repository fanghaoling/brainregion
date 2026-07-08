"""沙盒脑区(Phase D+):region-as-tool —— recall_map 在 region 臂调专用 LLM(第二个 LLM 当脑区)。

env-loop 原本单主脑 + 工具;Phase D 加「脑区缝」:工具的 dispatch 可拦下转调一个 scoped 第二 LLM。
首个脑区 = 记忆脑区(MemoryRegion),A/B vs Phase C 被动 recall_map(干净变量隔离,见 loop._recall_via_region)。
"""
from .memory_region import MemoryRegion, build_memory_region_system_prompt

__all__ = ["MemoryRegion", "build_memory_region_system_prompt"]
