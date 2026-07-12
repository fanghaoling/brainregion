"""沙盒 A/B eval + gate:对比 brainregion 臂 vs none 臂的 solve_rate(主)/cost(诊断)/steps(诊断)。

matched-pair(同 task 跑两臂,各用 fresh 物化的 run_dir)+ temperature=0 降噪。
bootstrap CI on solve_rate_delta(复用 eval/stats.bootstrap_statistic);cost 不当 primary gate
(brainregion 臂因 wake/注入天然多一点开销,同 Phase1 cost_primary=False 教训)。
小 n / 零方差 → INCONCLUSIVE(复用 bootstrap 守卫)。

持久化:MVP 落 JSON 报告到 run artifact(trajectory 可序列化);专用 SQLite trajectory 表 defer
(能力答案不依赖 DB 持久化,保持 MVP lean)。
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from brainregion.eval import stats as eval_stats
from brainregion.runtime import merge_usage, normalize_usage

from . import SandboxTask
from .isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from .loop import Trajectory, run_agent

logger = logging.getLogger("brainregion.sandbox.eval")

FORMAL_MIN_N = 30  # n < 此 → pilot_ 前缀(不宣称可信闸门),同 outcome 习惯


def _solve_rate_delta(rows: list[dict]) -> float | None:
    """mean(t_solved) − mean(c_solved),control/treatment 约定 row["_control_"]/_treatment_。"""
    if not rows:
        return None
    c = sum(1 for r in rows if r["_control_"]["solved"]) / len(rows)
    t = sum(1 for r in rows if r["_treatment_"]["solved"]) / len(rows)
    return t - c


def _cost_delta(rows: list[dict]) -> float | None:
    if not rows:
        return None
    c = sum(r["_control_"]["cost"] for r in rows) / len(rows)
    t = sum(r["_treatment_"]["cost"] for r in rows) / len(rows)
    return t - c


def _steps_delta(rows: list[dict]) -> float | None:
    if not rows:
        return None
    c = sum(r["_control_"]["steps"] for r in rows) / len(rows)
    t = sum(r["_treatment_"]["steps"] for r in rows) / len(rows)
    return t - c


def _input_tokens_delta(rows: list[dict]) -> float | None:
    if not rows:
        return None
    c = sum(r["_control_"]["input_tokens"] for r in rows) / len(rows)
    t = sum(r["_treatment_"]["input_tokens"] for r in rows) / len(rows)
    return t - c


def evaluate_gate(deltas: dict[str, dict], n: int, formal_min_n: int = FORMAL_MIN_N) -> dict[str, Any]:
    """GO/NO_GO/INCONCLUSIVE on solve_rate_delta(主);cost/steps 仅 diagnostic。"""
    sr = deltas["solve_rate_delta"]
    prefix = "pilot_" if n < formal_min_n else ""
    if sr["point"] is None:
        return {"decision": f"{prefix}INCONCLUSIVE", "primary": "solve_rate_delta", "reason": "not estimable (n<2 / zero variance)"}
    low, high = sr["low"], sr["high"]
    if low is not None and low > 0:
        return {"decision": f"{prefix}GO", "primary": "solve_rate_delta", "reason": f"CI [{low:.3f},{high:.3f}] 整段 >0 = brainregion 臂 solve_rate 显著更高"}
    if high is not None and high < 0:
        return {"decision": f"{prefix}NO_GO", "primary": "solve_rate_delta", "reason": f"CI [{low:.3f},{high:.3f}] 整段 <0 = brainregion 臂 solve_rate 显著更低"}
    return {"decision": f"{prefix}INCONCLUSIVE", "primary": "solve_rate_delta", "reason": f"CI [{low:.3f},{high:.3f}] 跨 0;cost/steps 见 diagnostics(非 primary)"}


def _bootstrap(rows: list[dict], stat_fn, run_id: str, metric: str) -> dict:
    return eval_stats.bootstrap_statistic(
        rows, stat_fn, seed=eval_stats.seed_for(run_id, metric),
    )


def _per_arm_summary(trajectories: list[Trajectory]) -> dict[str, dict[str, float]]:
    by_arm: dict[str, list[Trajectory]] = {}
    for tr in trajectories:
        by_arm.setdefault(tr.arm, []).append(tr)
    summary = {}
    for arm, trajs in by_arm.items():
        n = len(trajs)
        arm_summary = {
            "n": n,
            "solve_rate": sum(1 for t in trajs if t.tests_green) / n if n else 0.0,
            "mean_steps": sum(t.n_steps for t in trajs) / n if n else 0.0,
            "mean_total_cost_usd": sum(t.total_main_cost_usd + t.total_arm_cost_usd for t in trajs) / n if n else 0.0,
            "mean_main_cost_usd": sum(t.total_main_cost_usd for t in trajs) / n if n else 0.0,
            "mean_arm_cost_usd": sum(t.total_arm_cost_usd for t in trajs) / n if n else 0.0,
        }
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens"):
            arm_summary[f"mean_main_{key}"] = (
                sum(normalize_usage(t.total_main_usage)[key] for t in trajs) / n if n else 0.0
            )
            arm_summary[f"mean_arm_{key}"] = (
                sum(normalize_usage(t.total_arm_usage)[key] for t in trajs) / n if n else 0.0
            )
            arm_summary[f"mean_total_{key}"] = (
                sum(merge_usage(t.total_main_usage, t.total_arm_usage)[key] for t in trajs) / n if n else 0.0
            )
        summary[arm] = arm_summary
    return summary


async def run_sandbox_eval(
    backend: Any,
    model: str,
    tasks: list[SandboxTask],
    *,
    control: str = "none",
    treatment: str = "brainregion",
    max_steps: int = 10,
    max_cost_usd: float = 0.5,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    keep_on_fail: bool = False,
    endpoint_id: str | None = None,
    thinking: bool | None = None,
    effort: str | None = None,
    **loop_kwargs: Any,
) -> dict[str, Any]:
    """matched-pair A/B:每 task 跑 control+treatment 两臂(各 fresh run_dir)。返报告 dict。"""
    run_id = f"sandbox-{int(time.time() * 1000)}"
    trajectories: list[Trajectory] = []
    rows: list[dict] = []

    for task in tasks:
        arm_results: dict[str, dict] = {}
        for arm in (control, treatment):
            run_dir = make_run_dir()
            materialize_fixture(task, Path(run_dir))
            try:
                traj = await run_agent(
                    backend, model, task, run_dir=run_dir, arm=arm,
                    max_steps=max_steps, max_cost_usd=max_cost_usd,
                    temperature=temperature, max_tokens=max_tokens,
                    endpoint_id=endpoint_id, thinking=thinking, effort=effort, **loop_kwargs,
                )
            finally:
                if not (keep_on_fail and not traj.tests_green):
                    cleanup_run_dir(run_dir)
                else:
                    logger.info("kept run_dir for failed task %s arm %s: %s", task.id, arm, run_dir)
            trajectories.append(traj)
            arm_results[arm] = {
                "solved": traj.tests_green,
                "cost": traj.total_main_cost_usd + traj.total_arm_cost_usd,
                "steps": traj.n_steps,
                "input_tokens": merge_usage(traj.total_main_usage, traj.total_arm_usage)["input_tokens"],
            }
        rows.append({"task_id": task.id, "_control_": arm_results[control], "_treatment_": arm_results[treatment]})

    n = len(rows)
    deltas = {
        "solve_rate_delta": _bootstrap(rows, _solve_rate_delta, run_id, "solve_rate_delta"),
        "cost_delta": _bootstrap(rows, _cost_delta, run_id, "cost_delta"),
        "steps_delta": _bootstrap(rows, _steps_delta, run_id, "steps_delta"),
        "input_tokens_delta": _bootstrap(rows, _input_tokens_delta, run_id, "input_tokens_delta"),
    }
    gate = evaluate_gate(deltas, n)
    return {
        "run_id": run_id,
        "model": model,
        "control": control,
        "treatment": treatment,
        "n": n,
        "per_arm": _per_arm_summary(trajectories),
        "deltas": deltas,
        "gate": gate,
        "rows": rows,
        "trajectories": [t.to_dict() for t in trajectories],
    }


def write_report(report: dict[str, Any], out_dir: str | os.PathLike[str] | None = None) -> Path:
    """把报告落 JSON 到 artifact 目录(默认 .brain-region/sandbox/)。"""
    out = Path(out_dir) if out_dir else Path(".brain-region") / "sandbox"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{report['run_id']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def render_summary(report: dict[str, Any]) -> str:
    """人类可读的 gate 总结(markdown-ish)。"""
    g = report["gate"]
    pa = report["per_arm"]
    sr = report["deltas"]["solve_rate_delta"]
    lines = [
        f"### sandbox eval {report['run_id']} (n={report['n']}, model={report['model']})",
        f"**gate: {g['decision']}** — {g['reason']}",
        "",
        f"solve_rate_delta: point={_fmt(sr['point'])} CI=[{_fmt(sr['low'])},{_fmt(sr['high'])}] (n={sr['n']}, B={sr['B']})",
    ]
    for arm, s in pa.items():
        lines.append(
            f"  {arm}: solve_rate={s['solve_rate']:.2f} mean_steps={s['mean_steps']:.1f} "
            f"mean_cost=${s['mean_total_cost_usd']:.4f} (main ${s['mean_main_cost_usd']:.4f} + arm ${s['mean_arm_cost_usd']:.4f})"
        )
    return "\n".join(lines)


def _fmt(x: float | None) -> str:
    return "None" if x is None else f"{x:.3f}"
