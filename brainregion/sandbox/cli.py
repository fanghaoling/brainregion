"""`brain-region sandbox` CLI:run(单跑)+ eval(A/B gate)。

backend 构建复用 eval 模式(_resolve_endpoints + _normalize_one);主脑模型经 --main-brain 传入
(或 sandbox_main_brain 配置),解析成 {model, endpoint_id} 喂 run_agent。
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from brainregion import defaults as _defaults_mod
from brainregion.providers.litellm import LiteLLMBackend
from brainregion.server import _normalize_one, _resolve_endpoints

from .eval import render_summary, run_sandbox_eval, write_report
from .fixtures import SANDBOX_FIXTURES, get_fixture, list_fixture_ids
from .isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from .loop import run_agent

logger = logging.getLogger("brainregion.sandbox.cli")


def _build_backend(dd: dict[str, Any]) -> tuple[LiteLLMBackend, dict[str, Any]]:
    registry = _resolve_endpoints(dd.get("endpoints") or {})
    backend = LiteLLMBackend(timeout=float(dd.get("timeout", 90)), endpoint_registry=registry)
    return backend, registry


def _resolve_main_brain(model_str: str, registry: dict, dd: dict[str, Any]) -> tuple[str, str | None]:
    entry = _normalize_one(model_str, set(registry.keys()), dd.get("endpoints"))
    return entry["model"], entry.get("endpoint_id")


def _thinking_arg(args: argparse.Namespace) -> bool | None:
    """--thinking off→False(便宜快非推理,默认),on→True(None=provider 默认,沙盒不用)。"""
    val = getattr(args, "thinking", "off")
    if val == "on":
        return True
    if val == "off":
        return False
    return None


def _resolve_tasks(args: argparse.Namespace) -> list:
    ids = list_fixture_ids()
    if getattr(args, "task", None):
        return [get_fixture(args.task)]
    if getattr(args, "tasks", None):
        wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in ids]
        if unknown:
            raise SystemExit(f"unknown fixture(s): {unknown}; available: {ids}")
        return [get_fixture(t) for t in wanted]
    return list(SANDBOX_FIXTURES)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    """`brain-region sandbox run`:单跑一个 fixture(默认 arm=none)。"""
    dd = _defaults_mod.apply()
    backend, registry = _build_backend(dd)
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain 必填(或配置 sandbox_main_brain)")
    model, endpoint_id = _resolve_main_brain(model_str, registry, dd)
    tasks = _resolve_tasks(args)
    if len(tasks) != 1:
        raise SystemExit(f"`run` 一次一个 task;got {len(tasks)}(用 `eval` 跑多 task A/B)")

    task = tasks[0]
    run_dir = make_run_dir()
    materialize_fixture(task, Path(run_dir))
    keep = bool(getattr(args, "keep", False))
    traj = None
    try:
        traj = await run_agent(
            backend, model, task, run_dir=run_dir, arm=args.arm,
            max_steps=int(args.max_steps or dd.get("sandbox_max_steps", 10)),
            max_cost_usd=float(args.max_cost_usd or dd.get("sandbox_max_cost_usd", 0.5)),
            temperature=float(dd.get("sandbox_temperature", 0.0)),
            max_tokens=int(args.max_tokens or 2048),
            consecutive_error_limit=int(dd.get("sandbox_consecutive_error_limit", 3)),
            transcript_token_cap=int(dd.get("sandbox_transcript_token_cap", 24000)),
            endpoint_id=endpoint_id,
            thinking=_thinking_arg(args), effort=args.effort,
        )
    finally:
        # 失败(run_agent raise 或 tests 没 green)且 --keep → 留检;否则清。
        if traj is None or not (keep and not traj.tests_green):
            cleanup_run_dir(run_dir)
        else:
            logger.info("kept run_dir (failed): %s", run_dir)

    result = {"trajectory": traj.to_dict(), "run_dir": run_dir if (keep and not traj.tests_green) else None}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    """`brain-region sandbox eval`:matched-pair A/B(none vs brainregion),bootstrap CI + gate。"""
    dd = _defaults_mod.apply()
    backend, registry = _build_backend(dd)
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain 必填(或配置 sandbox_main_brain)")
    model, endpoint_id = _resolve_main_brain(model_str, registry, dd)
    tasks = _resolve_tasks(args)
    if not tasks:
        raise SystemExit("no tasks selected")

    report = await run_sandbox_eval(
        backend, model, tasks,
        control="none", treatment="brainregion",
        max_steps=int(args.max_steps or dd.get("sandbox_max_steps", 10)),
        max_cost_usd=float(args.max_cost_usd or dd.get("sandbox_max_cost_usd", 0.5)),
        temperature=float(dd.get("sandbox_temperature", 0.0)),
        max_tokens=int(args.max_tokens or 2048),
        keep_on_fail=bool(getattr(args, "keep", False)),
        consecutive_error_limit=int(dd.get("sandbox_consecutive_error_limit", 3)),
        transcript_token_cap=int(dd.get("sandbox_transcript_token_cap", 24000)),
        endpoint_id=endpoint_id,
        thinking=_thinking_arg(args), effort=args.effort,
    )
    path = write_report(report, getattr(args, "out", None))
    print(render_summary(report))
    print(f"\n报告: {path}")
    return {"report": report, "path": str(path)}
