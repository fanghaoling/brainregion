"""Phase 4 formal A/B harness(env-regime):多 run 估 solve_rate 分布 + Echo 控制臂 + 过程指标
+ config 级 bootstrap CI/gate。

复用 ``run_agent`` + 统计原语(``bootstrap_statistic`` / ``evaluate_gate``),**不改 code-regime eval**
(``run_sandbox_eval``)。arms = feature-config(``EnvArm`` 脑区开/关 + strategy 模式)。

回答的问题(默认 ``memory-strategy`` 臂集):
- ``memory_strategy − memory_only``:加 Strategy 是否提升 solve_rate?
- ``memory_strategy − memory_echo``:提升来自 Strategy **规划内容**,还是「plan 工具存在」的行为改变?(控制臂)
- ``memory_echo − memory_only``:控制洁净度诊断(self-echo 本身是否改行为)。

review 双强(2026-07-08,gpt-5.5 + opus-4-8)硬化:
- cost-cap **matched-set 循环**(``for config: for repeat: for arm``)+ 按**实际 n_runs** 聚合 + 不完整矩阵 → gate INCONCLUSIVE;
- revisit_rate 除零 → null(零移动 run);coverage 分母 = ``size²``(cap 1.0);
- bootstrap 单位 = config(尊重 maze 级聚类,避 pseudo-replication);N(config)<2 → None CI → INCONCLUSIVE;
- 报告记生成配置(model/temperature/thinking/effort)→ R repeats 可复现语义明确。
"""
from __future__ import annotations

import csv
import json
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brainregion.eval import stats as eval_stats

from .eval import evaluate_gate
from .envs import GridWorld, build_env_system_prompt
from .isolation import cleanup_run_dir, make_run_dir
from .loop import _current_env, run_agent, scoped_env, scoped_memory_mode
from .regions import EchoStrategy, MemoryRegion, StrategyRegion
from .regions.strategy_region import _strip_to_thought
from .task import SandboxTask

logger = logging.getLogger("brainregion.sandbox.env_eval")

# 网格动作 → pose delta(镜像 gridworld._ACTION_DELTA / memory_region._ACTION_DELTA;网格约定 y 向下增)。
_ACTION_DELTA: dict[str, tuple[int, int]] = {
    "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
}


@dataclass(frozen=True)
class EnvConfig:
    """一个 maze = 一个对比单位(bootstrap 行)。"""

    size: int
    seed: int | None = None            # random_goal_seed
    wall_seed: int | None = None
    wall_density: float = 0.0
    visibility_radius: int = 2
    max_steps: int = 30

    @property
    def label(self) -> str:
        parts = [f"{self.size}x{self.size}"]
        if self.seed is not None:
            parts.append(f"seed{self.seed}")
        if self.wall_density:
            parts.append(f"w{self.wall_density}")
        return "_".join(parts)


@dataclass(frozen=True)
class EnvArm:
    """一个臂 = 一个 feature-config(脑区开/关 + strategy 模式 + metronome push)。

    加新脑区 = 加一个字段 + 一个 CLI flag,不动 harness 主体(feature-config 本位)。
    """

    name: str
    memory_tool: bool = False          # --memory 被动完美图工具(Phase C 基线下界)
    memory_region: bool = False        # --memory-region 有状态脑区
    strategy: str = "none"             # "none" | "real" | "echo" | "dummy"(echo=主脑自产无LLM;dummy=同源同成本固定模板)
    metronome: bool = False            # Phase 4.1 push 臂:无 pull 工具,节拍器每 N 步推脑区状态(清测内容价值)
    visual_ephemeral: bool = False     # Phase 4.2:剥历史视觉观察出 transcript(只留最新);逼主脑调脑区拿历史
    registry: str = "none"             # Phase 4.3 脑区注册表块:"none"|"cap"(仅能力)|"full"(能力+客观触发)


# 预设 = 常用比较的糖(CLI 也可 --arm 显式给 feature-config)
ARMS_MEMORY_STRATEGY: tuple[EnvArm, ...] = (
    EnvArm("memory_only", memory_region=True),
    EnvArm("memory_strategy", memory_region=True, strategy="real"),
    EnvArm("memory_echo", memory_region=True, strategy="echo"),       # 控制臂
)
ARMS_MEMORY_BASELINE: tuple[EnvArm, ...] = (
    EnvArm("memory_tool", memory_tool=True),
    EnvArm("memory_only", memory_region=True),
)
ARMS_METRONOME: tuple[EnvArm, ...] = (        # Phase 4.1 push:清测脑区内容价值(强制曝光)
    EnvArm("push_real",  memory_region=True, strategy="real",  metronome=True),
    EnvArm("push_dummy", memory_region=True, strategy="dummy", metronome=True),   # 主控制:同源同成本固定模板
    EnvArm("push_echo",  memory_region=True, strategy="echo",  metronome=True),   # 次要:self-reminder 分析
)
ARMS_EPHEMERAL: tuple[EnvArm, ...] = (        # Phase 4.2:剥视觉出 transcript,测脑区是否变必需(GPT#2:真记忆为主对照)
    EnvArm("eph_memregion", memory_region=True, visual_ephemeral=True),   # 主:真记忆(recall_map→rough_map 不完美)
    EnvArm("eph_noregion",  visual_ephemeral=True),                       # 基线:仅当前视野,无记忆
    EnvArm("eph_region",    memory_tool=True,  visual_ephemeral=True),    # 上界:recall_map→env.render() 完美 oracle
)
ARMS_REGISTRY: tuple[EnvArm, ...] = (         # Phase 4.3:脑区注册表块,测是否触发 consult(ephemeral 失败处)
    EnvArm("eph_memregion",        memory_region=True, visual_ephemeral=True),                # baseline(无 registry)
    EnvArm("eph_memregion_regcap", memory_region=True, visual_ephemeral=True, registry="cap"),   # 仅能力(显著性)
    EnvArm("eph_memregion_reg",    memory_region=True, visual_ephemeral=True, registry="full"),  # 能力+客观触发
)
ARM_PRESETS: dict[str, tuple[EnvArm, ...]] = {
    "memory-strategy": ARMS_MEMORY_STRATEGY,
    "memory-baseline": ARMS_MEMORY_BASELINE,
    "metronome": ARMS_METRONOME,
    "ephemeral": ARMS_EPHEMERAL,
    "registry": ARMS_REGISTRY,
    "all": ARMS_MEMORY_STRATEGY + ARMS_MEMORY_BASELINE + ARMS_METRONOME + ARMS_EPHEMERAL + ARMS_REGISTRY,
}


# ---------- env / region 装配 ----------


def build_env_for_config(cfg: EnvConfig) -> GridWorld:
    """从 EnvConfig 构造 GridWorld(eval 恒 strict_obs + fog → 部分可观,所有臂同底)。"""
    kw: dict[str, Any] = {
        "size": cfg.size, "start": (0, 0),
        "visibility_radius": cfg.visibility_radius, "strict_obs": True,
    }
    if cfg.seed is not None:
        kw["random_goal_seed"] = cfg.seed
    if cfg.wall_seed is not None:
        kw["random_walls_seed"] = cfg.wall_seed
        kw["wall_density"] = cfg.wall_density
    return GridWorld(**kw)


def build_regions_for_arm(
    arm: EnvArm, env: GridWorld, *, log_len: int = 32,
) -> tuple[Any, Any, bool]:
    """返 ``(memory_region, strategy_region, memory_mode)``。

    memory_mode=True → runner 侧包 scoped_memory_mode(recall_map 可用)。memory_tool 臂无 region 但
    开 memory_mode(Phase C 被动图);memory_region 臂 new MemoryRegion;strategy real→StrategyRegion,
    echo→EchoStrategy,none→None。
    """
    memory_mode = arm.memory_tool or arm.memory_region
    memory_region = MemoryRegion(start=env.start, log_len=log_len) if arm.memory_region else None
    if arm.strategy in ("real", "dummy"):
        strategy_region: Any = StrategyRegion()   # dummy 用真 StrategyRegion(injector 调它 match real 的 2 次调用成本,丢输出)
    elif arm.strategy == "echo":
        strategy_region = EchoStrategy()          # 占位(echo injector 不调 region,返主脑自产)
    else:
        strategy_region = None
    return memory_region, strategy_region, memory_mode


def _format_status(*, mem: str, strat: str) -> str:
    """两槽结构(push_real/dummy/echo 同结构,只内容差)—— 记忆脑区 + 策略脑区状态行。"""
    return f"记忆脑区:{mem}\n策略脑区:{strat}"


# review 双强 consensus HIGH:real(real 解读)vs dummy(模板)注入长度差 → transcript 体积差是 presence 混淆,
# 破「唯一差=内容」。固定字符预算:real 截断到预算,dummy **程序化填到恰好预算**(同结构标签 + 无信息中性填充)。
_STATUS_BUDGET = 300
_DUMMY_FILLER = (
    "记忆脑区:本周期无可报告的具体地图解读(占位状态,不含可用导航信息);"
    "策略脑区:本周期无可报告的方向意图(占位状态,不含可用规划信息);"
    "请依据你的局部视野与已走路径自行判断,本条不作为动作依据。"
)
_DUMMY_STATUS = (_DUMMY_FILLER * 8)[:_STATUS_BUDGET]   # 恰好 _STATUS_BUDGET 字符,等长对齐 real


def make_status_injector(
    arm: EnvArm, memory_region: MemoryRegion, strategy_region: Any,
    backend: Any, model: str, *, endpoint_id: str | None, thinking: bool | None, effort: str | None,
):
    """Phase 4.1 metronome injector:async (step, messages) -> (status_str|None, cost_usd)。

    - **real**:调 memory.reason + strategy.reason(同源)→ 喂回 real rough_map + intent。
    - **dummy**(主控制):**同样**调两脑区(同源同成本同延迟,real 工作)→ 但喂回**固定中性模板**(content-null)。
      → real vs dummy 隔离「内容质量」(GPT #1+#2:echo 不同源,解释不了 real≈echo)。
    - **echo**(次要):返主脑上一句 thought(不同源,无 LLM)→ 仅 self-reminder 分析。
    - real/dummy 都调 memory.reason + strategy.reason(**2 次同源同成本同延迟调用**);dummy 丢输出喂固定模板。
      → real vs dummy 隔离「内容质量」(GPT #1+#2:echo 不同源、call-count 不 match,解释不了 real≈echo)。
    """
    ep, th, ef = endpoint_id, thinking, effort

    async def inject(step: int, messages: list) -> tuple[str | None, float]:
        if arm.strategy == "echo":
            prev = next((m["content"] for m in reversed(messages) if m.get("role") == "assistant"), "")
            e = _strip_to_thought(prev) or "(无上一句推理)"
            return _format_status(mem=e, strat=e), 0.0
        env = _current_env.get()
        if env is None or memory_region is None:
            return None, 0.0
        # real 与 dummy 都调 memory.reason + strategy.reason(同源、同成本、同调用数);memory.reason 刷 rough_map
        m = await memory_region.reason(backend, model, env.relative_view(), endpoint_id=ep, thinking=th, effort=ef)
        s = await strategy_region.reason(
            backend, model, memory_rough_map=memory_region.rough_map, current_view=env.relative_view(),
            rough_position=memory_region.pose, prev_assistant=None,
            endpoint_id=ep, thinking=th, effort=ef,
        )
        cost = float(m.get("cost_usd", 0.0) or 0.0) + float(s.get("cost_usd", 0.0) or 0.0)
        if arm.strategy == "dummy":       # 同源同成本,喂回等长中性占位(content-null,长度对齐避混淆)
            return _DUMMY_STATUS, cost
        return _format_status(mem=memory_region.rough_map, strat=s["intent"])[:_STATUS_BUDGET], cost   # real 截断到预算

    return inject


# ---------- 过程指标 ----------


def _positions_from_traj(traj: Any, env: GridWorld) -> list[tuple[int, int]]:
    """从 trajectory 的 act steps 重放位置序列(确定性:动作=单位步,墙/越界挡)。

    用 act args + env.walls/grid 重放 → 与 env 实际位置一致(无需 run_agent per-step 钩子)。
    每 act step 追一个位置(成功移动则变,blocked/invalid 则同位)。
    """
    pos = tuple(env.start)
    positions: list[tuple[int, int]] = [pos]
    walls = env.walls
    size = env.size
    for s in traj.steps:
        if s.tool != "act":
            continue
        action = (s.args or {}).get("action", "")
        action = action.strip().lower() if isinstance(action, str) else ""
        delta = _ACTION_DELTA.get(action)
        if delta is None:
            positions.append(pos)
            continue
        dx, dy = delta
        nxt = (pos[0] + dx, pos[1] + dy)
        if 0 <= nxt[0] < size and 0 <= nxt[1] < size and nxt not in walls:
            pos = nxt
        positions.append(pos)
    return positions


def _revisit_rate(positions: list[tuple[int, int]]) -> float | None:
    """走重复格比例 = revisits / successful_moves(脑区帮建图 → 应降)。

    review 双强(除零):无成功移动(零移动 run:首步 give-up / 全程撞墙)→ **None**(聚合跳过 null)。
    """
    if not positions:
        return None
    visited = {positions[0]}
    successful = 0
    revisits = 0
    prev = positions[0]
    for p in positions[1:]:
        if p != prev:  # 成功移动
            successful += 1
            if p in visited:
                revisits += 1
            visited.add(p)
        prev = p
    if successful == 0:
        return None
    return revisits / successful


def _coverage(env: GridWorld) -> float | None:
    """网格探索覆盖率 = min(1.0, explored/size²)。分母 size²(恒 ≥4;review 双强统一口径,免可达争议)。

    `_explored` 可含隔墙 Chebyshev 可见格 → cap 1.0。denom≤0 → None(防御;GridWorld 构造器已保 free_cells>0)。
    """
    denom = env.size * env.size
    if denom <= 0:
        return None
    explored = len(getattr(env, "_explored", set()))
    return min(1.0, explored / denom)


# region_status 被主脑引用的标记(review 双强:收紧为 status 专属标签,去通用词「意图/脑区/地图理解」假阳)。
# real/dummy 注入串都含「记忆脑区/策略脑区」标签(_format_status/_DUMMY_STATUS),故两臂同标签基线;
# thought 引用这些 status 专属词 → 视为「读了注入的 status」。
_STATUS_MARKERS = ("记忆脑区", "策略脑区", "记忆显示", "根据记忆", "rough")


def _status_referenced(traj: Any, period: int) -> float | None:
    """post-push 步(注入在 step 顶部,该步 thought 即收 status 后的推理)thought 引用 region_status 的比例。

    review 双强:粗粒度,仅作诊断。负对照基线(无注入 run 的标记率)应另测 —— 若 ≈0 则本指标可信;
    若高则需收紧到 status 具体内容 token。real/dummy 同标签 → real 高 dummy 低 = engage real 内容。
    """
    if not period or period <= 0:
        return None
    push_steps = [s for s in traj.steps if s.index > 0 and s.index % period == 0]
    if not push_steps:
        return None
    hit = sum(1 for s in push_steps if any(m in (s.thought or "") for m in _STATUS_MARKERS))
    return hit / len(push_steps)


# ---------- 单 episode ----------


async def _run_one_episode(
    backend: Any, model: str, cfg: EnvConfig, arm: EnvArm, *,
    max_cost_usd: float, temperature: float, max_tokens: int,
    endpoint_id: str | None, thinking: bool | None, effort: str | None,
    log_len: int = 32, status_period: int = 3,
) -> dict:
    """单 episode:构造 env+regions → run_agent → per-run 摘要(含过程指标)。复用 run_env 装配模式。

    metronome(push)臂:无 pull 工具(memory=False prompt),节拍器每 status_period 步注入 region_status;
    strategy_region 传 None 给 run_agent(禁 plan-intercept),真对象由 injector 闭包持有。
    """
    env = build_env_for_config(cfg)
    memory_region, strategy_region, memory_mode = build_regions_for_arm(arm, env, log_len=log_len)
    # goal_text 臂感知:push 臂不提 pull 工具;无记忆臂(noregion)不提 recall_map;real/echo 提 plan。
    if arm.metronome:
        goal_text = "找到并到达藏在网格里的目标 G(observe 只看当前视野;每几步收到脑区背景状态作参考;先探索再过去)"
    else:
        has_memory = arm.memory_tool or arm.memory_region
        tools_hint = "recall_map 拿累积探索图/记忆理解" if has_memory else "靠当前视野探索"
        if arm.strategy in ("real", "echo"):
            tools_hint += ",plan 拿策略意图"
        goal_text = f"找到并到达藏在网格里的目标 G(observe 只看当前视野,{tools_hint};先探索拼图再过去)"
    task = SandboxTask(id=f"env-{cfg.label}", goal=goal_text)

    def verify(t, run_dir, *, python_exe=None):  # env-grounded,完整 verify shape
        return {
            "tests_green": bool(env.solved),
            "solve_status": "solved" if env.solved else "tests_fail",
            "pytest": None, "gold_diff": getattr(t, "gold_diff", ""),
        }

    # push 臂:injector 闭包持真 strategy_region;run_agent 收 strategy_region=None(禁 plan pull-intercept)
    injector = None
    run_strategy_region = strategy_region
    if arm.metronome:
        injector = make_status_injector(
            arm, memory_region, strategy_region, backend, model,
            endpoint_id=endpoint_id, thinking=thinking, effort=effort,
        )
        run_strategy_region = None

    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            with scoped_memory_mode() if memory_mode else nullcontext():
                traj = await run_agent(
                    backend, model, task, run_dir=run_dir, arm="none",
                    max_steps=cfg.max_steps, max_cost_usd=max_cost_usd,
                    temperature=temperature, max_tokens=max_tokens,
                    endpoint_id=endpoint_id, thinking=thinking, effort=effort,
                    system_prompt=build_env_system_prompt(
                        env, goal_text,
                        memory=(arm.memory_tool or arm.memory_region) and not arm.metronome,  # 有 recall_map 才 memory=True
                        strategy=(not arm.metronome and arm.strategy in ("real", "echo")),
                        metronome=arm.metronome,
                        # ephemeral 不改 prompt(GPT #5:不强调「你会忘」,现有 memory prompt 已说 observe=当前/recall_map=历史)
                        registry=arm.registry,   # Phase 4.3 脑区注册表块(none/cap/full)
                    ),
                    verify_fn=verify,
                    memory_region=memory_region, strategy_region=run_strategy_region,
                    status_injector=injector, status_period=status_period,
                    visual_ephemeral=arm.visual_ephemeral,
                )
    finally:
        cleanup_run_dir(run_dir)

    positions = _positions_from_traj(traj, env)
    return {
        "config": cfg.label, "arm": arm.name,
        "solved": bool(traj.tests_green), "steps": traj.n_steps,
        "cost": round(traj.total_main_cost_usd + traj.total_arm_cost_usd, 6),
        "termination": traj.termination_reason,
        "n_recall": sum(1 for s in traj.steps if s.tool == "recall_map"),
        "n_plan": sum(1 for s in traj.steps if s.tool == "plan"),
        "revisit_rate": _revisit_rate(positions),
        "coverage": _coverage(env),
        "status_referenced": _status_referenced(traj, status_period) if arm.metronome else None,
    }


# ---------- 聚合 + bootstrap ----------


def _agg_arm_runs(arm_runs: list[dict]) -> dict:
    """聚合一个 (config×arm 或 全 config×arm) 的 run 列表 → 指标 dict(按实际 n_runs,非名义 repeats)。"""
    n = len(arm_runs)
    n_solved = sum(1 for r in arm_runs if r["solved"])

    def _mean(key: str) -> float | None:
        vals = [r[key] for r in arm_runs if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n_runs": n, "n_solved": n_solved, "solve_rate": (n_solved / n) if n else 0.0,
        "mean_steps": _mean("steps"), "mean_cost": _mean("cost"),
        "mean_revisit_rate": _mean("revisit_rate"), "mean_coverage": _mean("coverage"),
        "mean_n_plan": _mean("n_plan"), "mean_n_recall": _mean("n_recall"),
        "mean_status_referenced": _mean("status_referenced"),   # Phase 4.1 push 臂(post-push 引用 status 比例)
    }


def _solve_rate_delta(rows: list[dict], control: str, treatment: str) -> float | None:
    """跨 config 均值 (solve_rate[t] − solve_rate[c])(bootstrap stat_fn;row 带 per-arm 聚合)。"""
    ds = []
    for r in rows:
        c = r[control]; t = r[treatment]
        if c["n_runs"] < 1 or t["n_runs"] < 1:
            continue  # 不完整 config 跳过
        ds.append(t["solve_rate"] - c["solve_rate"])
    return sum(ds) / len(ds) if ds else None


def _metric_delta(key: str):
    """per-config 均值的 delta 工厂(过程指标,容忍 null:跳过该 config)。"""
    def _fn(rows: list[dict], control: str, treatment: str) -> float | None:
        ds = []
        for r in rows:
            c = r[control]; t = r[treatment]
            cv = c.get(key); tv = t.get(key)
            if cv is None or tv is None:
                continue
            ds.append(tv - cv)
        return sum(ds) / len(ds) if ds else None
    return _fn


def _bootstrap_pair(rows: list[dict], control: str, treatment: str, run_id: str, pair_key: str) -> dict:
    """config 级 bootstrap(行=config,resample configs;stat_fn 重算跨 config 均值 delta)。每 metric 独立 seed 流。"""
    metrics = {
        "solve_rate_delta": _solve_rate_delta,
        "revisit_delta": _metric_delta("mean_revisit_rate"),
        "coverage_delta": _metric_delta("mean_coverage"),
        "steps_delta": _metric_delta("mean_steps"),
        "cost_delta": _metric_delta("mean_cost"),
    }
    out: dict[str, dict] = {}
    for metric, fn in metrics.items():
        out[metric] = eval_stats.bootstrap_statistic(
            rows, lambda rs: fn(rs, control, treatment),
            seed=eval_stats.seed_for(run_id, f"{pair_key}|{metric}"),
        )
    return out


def _pairs(arms: tuple[EnvArm, ...]) -> list[tuple[EnvArm, EnvArm]]:
    """所有 i<j 有序对比对(回答每对 treatment vs control)。3 臂 → 3 对。"""
    return [(a, b) for i, a in enumerate(arms) for j, b in enumerate(arms) if i < j]


def _aggregate(
    run_id: str, model: str, configs: list[EnvConfig], arms: tuple[EnvArm, ...],
    repeats: int, runs: list[dict], cost_total: float, cost_capped: bool, cost_capped_at: str | None,
    temperature: float, thinking: bool | None, effort: str | None, endpoint_id: str | None, max_tokens: int,
) -> dict:
    arm_names = [a.name for a in arms]

    # per-config × arm(实际 n_runs)
    per_config: dict[str, dict[str, dict]] = {}
    for cfg in configs:
        per_config[cfg.label] = {}
        for arm in arms:
            arm_runs = [r for r in runs if r["config"] == cfg.label and r["arm"] == arm.name]
            per_config[cfg.label][arm.name] = _agg_arm_runs(arm_runs)

    # per-arm 池化(over configs×repeats)
    per_arm: dict[str, dict] = {}
    for arm in arms:
        arm_runs = [r for r in runs if r["arm"] == arm.name]
        agg = _agg_arm_runs(arm_runs)
        agg["n"] = len(arm_runs)
        per_arm[arm.name] = agg

    # signal regime(难度待拧 flag)
    if runs:
        all_solve = all(r["solved"] for r in runs)
        all_fail = all(not r["solved"] for r in runs)
        signal_regime = "all_solve" if all_solve else ("all_fail" if all_fail else "ok")
    else:
        signal_regime = "no_runs"

    # rows for bootstrap:每 config 一行,带各 arm 聚合
    rows = [{arm.name: per_config[cfg.label][arm.name] for arm in arms} for cfg in configs]

    pairwise: dict[str, dict] = {}
    n_complete_configs = 0
    for c_arm, t_arm in _pairs(arms):
        pair_key = f"{c_arm.name}_vs_{t_arm.name}"
        complete = [r for r in rows if r[c_arm.name]["n_runs"] >= 1 and r[t_arm.name]["n_runs"] >= 1]
        if c_arm.name == arms[0].name:  # 用第一对的完整 config 数作报告口径(各对通常一致)
            n_complete_configs = len(complete)
        boot = _bootstrap_pair(complete, c_arm.name, t_arm.name, run_id, pair_key)
        gate = evaluate_gate({"solve_rate_delta": boot["solve_rate_delta"]}, n=len(complete))
        pairwise[pair_key] = {
            "solve_rate_delta": boot["solve_rate_delta"],
            "revisit_delta": boot["revisit_delta"],
            "coverage_delta": boot["coverage_delta"],
            "steps_delta": boot["steps_delta"],
            "cost_delta": boot["cost_delta"],
            "n_complete_configs": len(complete),
            "gate": gate,
        }

    # cost cap 截断 → 矩阵不完整 → 所有 gate 强制 INCONCLUSIVE(review 双强:防偏置结论)
    incomplete = bool(cost_capped) and (n_complete_configs < len(configs))
    if incomplete:
        for pk in pairwise:
            g = pairwise[pk]["gate"]
            if "INCONCLUSIVE" not in g.get("decision", ""):
                pairwise[pk]["gate"] = {
                    "decision": "INCONCLUSIVE", "primary": "solve_rate_delta",
                    "reason": f"cost_capped 截断({cost_capped_at}),对比矩阵不完整 → gate 作废",
                }

    return {
        "run_id": run_id, "model": model, "arms": arm_names, "repeats": repeats,
        "configs": [c.label for c in configs],
        "temperature": temperature, "thinking": thinking, "effort": effort,
        "endpoint_id": endpoint_id, "max_tokens": max_tokens,
        "per_arm": per_arm,
        "per_config": [{label: per_config[label]} for label in per_config],
        "pairwise": pairwise, "signal_regime": signal_regime,
        "n_complete_configs": n_complete_configs,
        "cost_total": round(cost_total, 6), "cost_capped": cost_capped,
        "cost_capped_at": cost_capped_at, "incomplete_pairs": incomplete,
        "runs": runs,
    }


# ---------- 主循环 ----------


async def run_env_eval(
    backend: Any, model: str, configs: list[EnvConfig], arms: tuple[EnvArm, ...], *,
    repeats: int = 3, max_cost_usd: float = 2.0,
    temperature: float = 0.0, max_tokens: int = 2048,
    endpoint_id: str | None = None, thinking: bool | None = None, effort: str | None = None,
    log_progress: bool = True, status_period: int = 3,
) -> dict:
    """formal A/B:configs × arms × repeats(matched-set 循环)→ 报告 dict。

    循环顺序 ``for config: for repeat: for arm``(review 双强 cost-cap 均衡:每 repeat 是完整跨臂 matched set,
    cost cap 整组丢,不系统性缺某臂)。全局 ``max_cost_usd`` 守护:超即停 + 标 cost_capped + 聚合按实际 n_runs。
    """
    run_id = f"env-eval-{int(time.time() * 1000)}"
    runs: list[dict] = []
    cost_total = 0.0
    cost_capped = False
    cost_capped_at: str | None = None

    for cfg in configs:
        if cost_capped:
            break
        for r in range(repeats):
            if cost_capped:
                break
            for arm in arms:
                if cost_total >= max_cost_usd:
                    cost_capped = True
                    cost_capped_at = f"{cfg.label} repeat{r} arm{arm.name}"
                    break
                summary = await _run_one_episode(
                    backend, model, cfg, arm,
                    max_cost_usd=max(0.01, max_cost_usd - cost_total),  # per-run 剩余预算(下限避 0)
                    temperature=temperature, max_tokens=max_tokens,
                    endpoint_id=endpoint_id, thinking=thinking, effort=effort,
                    status_period=status_period,
                )
                cost_total += summary["cost"]
                runs.append(summary)
                if log_progress:
                    logger.info(
                        "[env-eval] %s r%d %s: solved=%s steps=%d revisit=%s cov=%s n_plan=%d cost=%.4f total=%.4f/%.4f",
                        cfg.label, r, arm.name, summary["solved"], summary["steps"],
                        summary["revisit_rate"], summary["coverage"], summary["n_plan"],
                        summary["cost"], cost_total, max_cost_usd,
                    )

    return _aggregate(
        run_id, model, configs, arms, repeats, runs, cost_total, cost_capped, cost_capped_at,
        temperature, thinking, effort, endpoint_id, max_tokens,
    )


# ---------- 输出 ----------


def write_csv(report: dict, path: str | Path) -> Path:
    """每 run 一行(平铺)落 CSV(含生成配置列,review 双强可复现)。"""
    p = Path(path)
    cols = ["config", "arm", "solved", "steps", "cost", "termination",
            "n_recall", "n_plan", "revisit_rate", "coverage", "status_referenced",
            "model", "temperature", "thinking", "effort"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in report["runs"]:
            w.writerow([
                r["config"], r["arm"], r["solved"], r["steps"], r["cost"], r["termination"],
                r["n_recall"], r["n_plan"], r["revisit_rate"], r["coverage"], r["status_referenced"],
                report["model"], report["temperature"], report["thinking"], report["effort"],
            ])
    return p


def write_report(report: dict, out_dir: str | Path | None = None) -> tuple[Path, Path]:
    """落 JSON 报告 + CSV 到 artifact 目录(默认 .brain-region/sandbox/)。返 (json_path, csv_path)。"""
    out = Path(out_dir) if out_dir else Path(".brain-region") / "sandbox"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{report['run_id']}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = write_csv(report, out / f"{report['run_id']}.csv")
    return json_path, csv_path


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def render_env_eval_summary(report: dict) -> str:
    """人类可读总结(markdown)。头版 = per-arm 指标(含过程指标);次要 = pairwise gate(pilot + signal_regime 警告)。"""
    lines = [
        f"### env-eval {report['run_id']}",
        f"model={report['model']} | configs={len(report['configs'])}({report['configs']}) "
        f"| repeats={report['repeats']} | arms={report['arms']} | temp={report['temperature']}",
        f"cost_total=${report['cost_total']:.4f} (cap{'触发' if report['cost_capped'] else '未触'})",
        "",
    ]
    sr = report["signal_regime"]
    if sr == "all_solve":
        lines.append("⚠️ signal_regime=all_solve —— 全解出(太难信号反向:太易),solve_rate gate 无区分度,勿 over-read。")
    elif sr == "all_fail":
        lines.append("⚠️ signal_regime=all_fail —— 全未解(太难),solve_rate gate 无区分度,看过程指标(coverage/revisit)。")
    if report["cost_capped"]:
        lines.append(f"⚠️ cost_capped at {report['cost_capped_at']} —— 对比矩阵不完整,gate 已作废为 INCONCLUSIVE。")

    lines.append("\n**per-arm(池化 over configs×repeats):**")
    lines.append("| arm | n | solve_rate | revisit | coverage | n_plan | n_recall | stat_ref | mean_steps | mean_cost |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm, s in report["per_arm"].items():
        lines.append(
            f"| {arm} | {s['n']} | {s['solve_rate']:.2f} | {_fmt(s['mean_revisit_rate'])} | "
            f"{_fmt(s['mean_coverage'])} | {(s['mean_n_plan'] or 0):.1f} | {(s['mean_n_recall'] or 0):.1f} | "
            f"{_fmt(s.get('mean_status_referenced'))} | {(s['mean_steps'] or 0):.1f} | ${(s['mean_cost'] or 0):.4f} |"
        )

    lines.append("\n**pairwise(config 级 bootstrap,次要 —— env 在变,标 pilot):**")
    for pk, pv in report["pairwise"].items():
        d = pv["solve_rate_delta"]
        lines.append(
            f"- {pk}: solve_rate Δ point={_fmt(d['point'])} CI=[{_fmt(d['low'])},{_fmt(d['high'])}] "
            f"(n_configs={pv['n_complete_configs']}) → **{pv['gate']['decision']}**"
        )
        rd = pv["revisit_delta"]; cd = pv["coverage_delta"]
        if rd["point"] is not None or cd["point"] is not None:
            lines.append(f"    过程:revisit Δ={_fmt(rd['point'])} coverage Δ={_fmt(cd['point'])}")
    return "\n".join(lines)
