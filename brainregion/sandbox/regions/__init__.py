"""沙盒脑区(Phase D+):region-as-tool —— 工具 dispatch 可拦下转调 scoped 第二 LLM(脑区)。

env-loop 原本单主脑 + 工具;Phase D 加「脑区缝」。脑区:
- 记忆脑区(MemoryRegion,D.2 有状态自给 rough map):recall_map → memory。
- 策略脑区(StrategyRegion,D.3 无状态规划器):plan → strategy,读 memory 的 rough_map = 多脑区协同。
- EchoStrategy(Phase 4 控制臂):无 LLM · 复述主脑上一句,隔离「plan 工具存在」的行为改变混淆。
- 拓扑记忆脑区(TopologicalRegion,Phase 4.6 代码无 LLM):recall_topo → 解读 env 图成 Trémaux
  动作状态(frontier/dead-end/backtrack)。⚠️ 表征/scaffolding 实验,非架构价值实验。
"""
from .memory_region import MemoryRegion, build_memory_region_system_prompt
from .strategy_region import StrategyRegion, EchoStrategy, build_strategy_region_system_prompt
from .topo_region import TopologicalRegion
from .path_region import PathTraceRegion
from .navigation_region import NavigationRegion

__all__ = [
    "MemoryRegion",
    "build_memory_region_system_prompt",
    "StrategyRegion",
    "EchoStrategy",
    "build_strategy_region_system_prompt",
    "TopologicalRegion",
    "PathTraceRegion",
    "NavigationRegion",
]
