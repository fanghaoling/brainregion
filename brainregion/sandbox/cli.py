"""`brain-region sandbox` CLI:run(单跑)+ eval(A/B gate)。

backend 构建复用 eval 模式(_resolve_endpoints + _normalize_one);主脑模型经 --main-brain 传入
(或 sandbox_main_brain 配置),解析成 {model, endpoint_id} 喂 run_agent。
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from brainregion import defaults as _defaults_mod
from brainregion.providers.litellm import LiteLLMBackend
from brainregion.server import _normalize_one, _resolve_endpoints

from .eval import render_summary, run_sandbox_eval, write_report
from .fixtures import SANDBOX_FIXTURES, get_fixture, list_fixture_ids
from .isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from .loop import run_agent
from .brain_verify import TraceResult, composite_verify, extract_final_patch, forced_trace
from .task import WorktreeTask
from .worktree import (
    bootstrap_worktree,
    capture_worktree_diff,
    detect_venv_python,
    worktree,
    write_worktree_run,
)

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
    """`brain-region sandbox run`:单跑(fixture 模式 默认;``--worktree`` 真实仓库模式)。"""
    dd = _defaults_mod.apply()
    backend, registry = _build_backend(dd)
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain 必填(或配置 sandbox_main_brain)")
    model, endpoint_id = _resolve_main_brain(model_str, registry, dd)
    if getattr(args, "worktree", False):
        return await _run_worktree(args, dd, backend, model, endpoint_id)
    return await _run_fixture(args, dd, backend, model, endpoint_id)


def _agent_kwargs(args: argparse.Namespace, dd: dict[str, Any], endpoint_id: str | None) -> dict[str, Any]:
    """run_agent 的公共 kwargs(fixture / worktree 两模式都用)。"""
    return {
        "arm": args.arm,
        "max_steps": int(args.max_steps or dd.get("sandbox_max_steps", 10)),
        "max_cost_usd": float(args.max_cost_usd or dd.get("sandbox_max_cost_usd", 0.5)),
        "temperature": float(dd.get("sandbox_temperature", 0.0)),
        "max_tokens": int(args.max_tokens or 2048),
        "consecutive_error_limit": int(dd.get("sandbox_consecutive_error_limit", 3)),
        "transcript_token_cap": int(dd.get("sandbox_transcript_token_cap", 24000)),
        "endpoint_id": endpoint_id,
        "thinking": _thinking_arg(args),
        "effort": args.effort,
        "brain_verify": bool(getattr(args, "brain_verify", False)),
    }


async def _run_fixture(args, dd, backend, model, endpoint_id) -> dict[str, Any]:
    """fixture 模式:物化 synthetic fixture 到 tmp_dir,跑 agent(默认 arm=none)。"""
    tasks = _resolve_tasks(args)
    if len(tasks) != 1:
        raise SystemExit(f"`run` 一次一个 task;got {len(tasks)}(用 `eval` 跑多 task A/B)")

    task = tasks[0]
    run_dir = make_run_dir()
    materialize_fixture(task, Path(run_dir))
    keep = bool(getattr(args, "keep", False))
    traj = None
    try:
        traj = await run_agent(backend, model, task, run_dir=run_dir, **_agent_kwargs(args, dd, endpoint_id))
    finally:
        # 失败(run_agent raise 或 tests 没 green)且 --keep → 留检;否则清。
        if traj is None or not (keep and not traj.tests_green):
            cleanup_run_dir(run_dir)
        else:
            logger.info("kept run_dir (failed): %s", run_dir)

    result = {"trajectory": traj.to_dict(), "run_dir": run_dir if (keep and not traj.tests_green) else None}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _resolve_bootstrap(args: argparse.Namespace) -> list[list[str]] | None:
    """``--no-bootstrap``→``[]``;``--bootstrap "cmd"``→``[[shlex.split]]``;否则 ``None``(自动探测)。"""
    if getattr(args, "no_bootstrap", False):
        return []
    raw = getattr(args, "bootstrap", None)
    if raw:
        return [shlex.split(raw)]
    return None


def _load_worktree_task(path: str) -> WorktreeTask:
    """从 JSON spec 文件加载 WorktreeTask(必填 id/goal/repo_path)。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in ("id", "goal", "repo_path") if not data.get(k)]
    if missing:
        raise SystemExit(f"task-spec 缺必填字段: {missing}")
    return WorktreeTask(
        id=data["id"],
        goal=data["goal"],
        repo_path=data["repo_path"],
        base_ref=data.get("base_ref", "HEAD"),
        test_args=data.get("test_args", ["-q"]),
        bootstrap_commands=data.get("bootstrap_commands"),
        seed_memory=data.get("seed_memory", []),
        gold_diff=data.get("gold_diff", ""),
        notes=data.get("notes", ""),
    )


def _build_worktree_task(args: argparse.Namespace) -> WorktreeTask:
    """从 ``--task-spec <json>`` 或内联(``--repo``/``--goal``/[``--test-args``]/[``--bootstrap``])构造 WorktreeTask。"""
    spec = getattr(args, "task_spec", None)
    if spec:
        task = _load_worktree_task(spec)
        # CLI 显式 bootstrap 覆盖 spec(--no-bootstrap / --bootstrap)
        if getattr(args, "no_bootstrap", False) or getattr(args, "bootstrap", None):
            task = replace(task, bootstrap_commands=_resolve_bootstrap(args))
        return task
    repo = getattr(args, "repo", None)
    goal = getattr(args, "goal", None)
    if not repo or not goal:
        raise SystemExit("worktree 模式需 --task-spec <json> 或 (--repo + --goal)")
    test_args = shlex.split(args.test_args) if getattr(args, "test_args", None) else ["-q"]
    return WorktreeTask(
        id=getattr(args, "task_id", None) or f"worktree-{int(time.time() * 1000)}",
        goal=goal,
        repo_path=repo,
        base_ref=getattr(args, "base", None) or "HEAD",
        test_args=test_args,
        bootstrap_commands=_resolve_bootstrap(args),
    )


async def _run_worktree(args, dd, backend, model, endpoint_id) -> dict[str, Any]:
    """worktree 模式:create worktree → bootstrap env → run_agent → 抓 diff + 落 run.json → 清理(RAII)。

    流程见 plan §5。run.json = replay-dataset 种子(task→patch→outcome→trace);agent 改动**不
    commit/push**,清理前抓 git diff 入 artifact,worktree 随后丢弃(``--keep`` 保留检视)。
    """
    task = _build_worktree_task(args)
    keep = bool(getattr(args, "keep", False))
    run_id = f"worktree-{int(time.time() * 1000)}"
    thinking = _thinking_arg(args)
    traj = None
    artifact_path: Path | None = None
    bootstrap_results: list[dict[str, Any]] = []
    with worktree(task.repo_path, task.base_ref, autoremove=not keep) as h:
        bootstrap_results = bootstrap_worktree(h, task.bootstrap_commands)
        py = getattr(args, "python", None) or detect_venv_python(h.path) or sys.executable
        traj = await run_agent(
            backend, model, task, run_dir=h.path, python_exe=py,
            **_agent_kwargs(args, dd, endpoint_id),
        )
        # 清理前抓 agent 产物 diff + 落 run.json(CM 退出即 remove,除非 --keep)
        diff = capture_worktree_diff(h)
        artifact_path = write_worktree_run({
            "run_id": run_id,
            "mode": "worktree",
            "task": {"id": task.id, "goal": task.goal, "repo_path": task.repo_path, "base_ref": task.base_ref},
            "branch": h.branch,
            "worktree_path": h.path,
            "kept": keep,
            "model": model,
            "python_exe": py,
            "arm": args.arm,
            "thinking": thinking,
            "effort": args.effort,
            "test_args": task.test_args,
            "bootstrap_commands": task.bootstrap_commands,
            "bootstrap_results": bootstrap_results,
            "trajectory": traj.to_dict(),
            **diff,
        })

    result = {
        "trajectory": traj.to_dict(),
        "run_id": run_id,
        "artifact": str(artifact_path) if artifact_path else None,
        # keep 才有意义的 worktree 元组(否则已清)
        "worktree_path": h.path if keep else None,
        "branch": h.branch if keep else None,
    }
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


async def verify_brain(args: argparse.Namespace) -> dict[str, Any]:
    """`brain-region sandbox verify-brain`:§15.8 trace-first + test-backstop 落地。

    对 run.json 里专家(沙盒)的补丁跑一次 forced-trace(廉价 LLM 层),对照该 run **已存的**
    客观 ``tests_green`` → composite(agree / 弱测试信号 / trace 漏检)。不重跑 pytest(用 run 时的
    客观结果),故能在历史 run 上离线复盘。
    """
    dd = _defaults_mod.apply()
    backend, registry = _build_backend(dd)
    model_str = args.main_brain or dd.get("sandbox_main_brain") or "deepseek-v4-flash"
    model, endpoint_id = _resolve_main_brain(model_str, registry, dd)

    run = json.load(open(args.run, encoding="utf-8"))
    traj = run.get("trajectory") or {}
    goal = (run.get("task") or {}).get("goal", "")
    test_green = traj.get("tests_green")  # run 时跑出的客观结果(最强 grounded check)
    test_req = args.test_req or goal
    patch = extract_final_patch(traj)

    if patch is None:
        tr = TraceResult(verdict=None, error="no apply_text_patch in run trajectory")
        res = composite_verify(tr, test_green)
        res.notes.insert(0, "run trajectory 无 apply_text_patch → 跳过 trace,仅客观测试")
    else:
        tr = await forced_trace(
            backend, model=model, endpoint_id=endpoint_id,
            goal=goal, test_req=test_req, patch=patch,
        )
        res = composite_verify(tr, test_green)

    out = {
        "run": Path(args.run).name,
        "model": model,
        "trace_verdict": res.trace_verdict,
        "test_green": res.test_green,
        "final_verdict": res.verdict,
        "agree": res.agree,
        "weak_test_signal": res.weak_test_signal,
        "trace_missed": res.trace_missed,
        "trace": tr.trace,
        "check": tr.check,
        "notes": res.notes,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
