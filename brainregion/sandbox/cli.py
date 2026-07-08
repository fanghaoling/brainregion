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
from contextlib import nullcontext as _nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from brainregion import defaults as _defaults_mod
from brainregion.providers.litellm import LiteLLMBackend
from brainregion.server import _normalize_one, _resolve_endpoints

from .eval import render_summary, run_sandbox_eval, write_report
from .envs import GridWorld, build_env_system_prompt, write_replay_html
from .fixtures import SANDBOX_FIXTURES, get_fixture, list_fixture_ids
from .isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from .loop import run_agent, run_cognitive_loop, scoped_env, scoped_memory_mode
from .brain_verify import TraceResult, composite_verify, extract_final_patch, forced_trace
from .task import SandboxTask, WorktreeTask
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


def _resolve_orthogonal(
    args: argparse.Namespace, dd: dict[str, Any], main_endpoint_id: str | None,
) -> tuple[str | None, str | None]:
    """``--orthogonal-brain``(或 ``sandbox_orthogonal_brain`` 配置)→ ``{model, endpoint_id}``;未设 → ``(None, None)``。

    仅 ``--brain-loop`` 下有意义(escalate 重跑语义只在 loop)。同 main 家族/endpoint → warn(退化同模型二跑,
    非正交);建议换家族(如 main=deepseek → orthogonal=glm)。
    """
    orthogonal_str = getattr(args, "orthogonal_brain", None) or dd.get("sandbox_orthogonal_brain") or ""
    if not orthogonal_str:
        return None, None
    registry = _resolve_endpoints(dd.get("endpoints") or {})
    entry = _normalize_one(orthogonal_str, set(registry.keys()), dd.get("endpoints"))
    model, endpoint_id = entry["model"], entry.get("endpoint_id")
    if endpoint_id is not None and endpoint_id == main_endpoint_id:
        logger.warning(
            "orthogonal-brain(%s) 与 main-brain 同 endpoint %s → 退化同模型二跑(非正交);建议换家族",
            model, endpoint_id,
        )
    return model, endpoint_id


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
        "brain_delegate": bool(getattr(args, "brain_delegate", False)),
    }


async def _run_expert(
    args: argparse.Namespace, backend, model, task, run_dir: str, dd: dict[str, Any],
    endpoint_id: str | None, *, python_exe: str | None = None,
):
    """单遍 run_agent 或外环 run_cognitive_loop(--brain-loop 分支)。

    run_cognitive_loop 强制 brain_verify+brain_delegate(内部 True),故 pop 这两 kw;
    且不吃 max_iterations 之外的 loop 专参。python_exe 仅 worktree 模式传。
    """
    kwargs = _agent_kwargs(args, dd, endpoint_id)
    if python_exe is not None:
        kwargs["python_exe"] = python_exe
    if bool(getattr(args, "brain_loop", False)):
        kwargs.pop("brain_verify", None)
        kwargs.pop("brain_delegate", None)
        ortho_model, ortho_ep = _resolve_orthogonal(args, dd, endpoint_id)
        if ortho_model is not None:
            kwargs["orthogonal_model"] = ortho_model
            kwargs["orthogonal_endpoint_id"] = ortho_ep
        return await run_cognitive_loop(
            backend, model, task, run_dir=run_dir,
            max_iterations=int(getattr(args, "max_iterations", 3)), **kwargs,
        )
    return await run_agent(backend, model, task, run_dir=run_dir, **kwargs)


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
        traj = await _run_expert(args, backend, model, task, run_dir, dd, endpoint_id)
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
        traj = await _run_expert(args, backend, model, task, h.path, dd, endpoint_id, python_exe=py)
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


async def run_env(args: argparse.Namespace) -> dict[str, Any]:
    """`brain-region sandbox env`(Phase A):主脑玩 GridWorld,observe/act 作 tool 复用 run_agent。

    env-grounded verify(tests_green := env.solved);0/1 reward。--debug 开后台调试窗(SSE 实时看
    env.step 事件);replay HTML 落 .brain-region/sandbox/。成功标准 = loop 干净终止 + 事件流 + replay 写出
    (solved 是信号非闸:0/1 稀疏 → solved=False 常态,Phase A 不要求解出)。
    """
    dd = _defaults_mod.apply()
    backend, registry = _build_backend(dd)
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain 必填(或配置 sandbox_main_brain)")
    model, endpoint_id = _resolve_main_brain(model_str, registry, dd)

    # 构造 env + 边界校验(constructor 校验 size/visibility_radius/goal/walls;非法 → 干净退出)
    size = int(args.size)
    memory = bool(getattr(args, "memory", False))  # Phase C:严格部分可观 + recall_map
    fog = bool(getattr(args, "fog", False)) or memory  # --memory 自动启用 fog(strict_obs 需要半径)
    vis_radius = getattr(args, "visibility_radius", None)
    if vis_radius is None and fog:
        vis_radius = 2  # --fog/--memory 默认半径 2
    goal_kw: dict[str, Any] = {"visibility_radius": vis_radius, "strict_obs": memory}
    if bool(getattr(args, "random_goal", False)):
        goal_kw["random_goal_seed"] = getattr(args, "seed", None) if getattr(args, "seed", None) is not None else 0
    elif getattr(args, "goal_x", None) is not None and getattr(args, "goal_y", None) is not None:
        goal_kw["goal"] = (int(args.goal_x), int(args.goal_y))
    wall_seed = getattr(args, "wall_seed", None)
    if wall_seed is not None:
        goal_kw["random_walls_seed"] = wall_seed
        goal_kw["wall_density"] = float(getattr(args, "wall_density", None) or 0.2)  # 默认密度 0.2
    try:
        env = GridWorld(size=size, start=(0, 0), **goal_kw)
    except ValueError as exc:
        raise SystemExit(f"env 构造非法: {exc}")
    goal_text = args.goal_text or (
        "找到并到达藏在网格里的目标 G(observe 只看当前视野,recall_map 拿累积探索图;先探索拼图再过去)"
        if memory else
        ("找到并到达藏在网格里的目标 G(你只看得到周围,`?` 是未探索区,先探索再过去)" if fog
         else "到达目标 G(从 @ 出发,避开墙 #,走到 G)")
    )
    max_steps = int(args.max_steps or dd.get("sandbox_max_steps", 10))
    if max_steps < 1:
        raise SystemExit("--max-steps 须为正整数")

    # --debug:后台开调试窗(用户实时看 env.step 事件进 SSE 时间线)
    debug_port = int(getattr(args, "debug_port", 8765))
    if getattr(args, "debug", False):
        import threading
        from brainregion.viz import DebugDashboardOptions, serve_debug_dashboard
        opts = DebugDashboardOptions(
            goal=goal_text, problem=goal_text, refresh_ms=1000, port=debug_port,
        )
        threading.Thread(
            target=serve_debug_dashboard, args=(opts,),
            kwargs={"open_browser": True, "open_path": "/scene"}, daemon=True,
        ).start()
        print(f"\n>>> 调试窗已开:场景查看 http://127.0.0.1:{debug_port}/scene (网格实时渲染 + 可回看)")
        print(f">>>          BrainRegion 面板 http://127.0.0.1:{debug_port}/ (脑区/模型调用)\n")

    task = SandboxTask(id=f"env-{env.size}x{env.size}", goal=goal_text)

    def verify(t, run_dir, *, python_exe=None):  # env-grounded,返完整 verify_solution shape
        return {
            "tests_green": bool(env.solved),
            "solve_status": "solved" if env.solved else "tests_fail",
            "pytest": None,
            "gold_diff": getattr(t, "gold_diff", ""),
        }

    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            # --memory:激活记忆脑区(recall_map 可用);scoped_memory_mode 包 run_agent
            with scoped_memory_mode() if memory else _nullcontext():
                traj = await run_agent(
                    backend, model, task, run_dir=run_dir, arm=args.arm,
                    max_steps=max_steps,
                    max_cost_usd=float(args.max_cost_usd or dd.get("sandbox_max_cost_usd", 0.5)),
                    temperature=float(dd.get("sandbox_temperature", 0.0)),
                    max_tokens=int(args.max_tokens or 2048),
                    consecutive_error_limit=int(dd.get("sandbox_consecutive_error_limit", 3)),
                    transcript_token_cap=int(dd.get("sandbox_transcript_token_cap", 24000)),
                    endpoint_id=endpoint_id, thinking=_thinking_arg(args), effort=args.effort,
                    system_prompt=build_env_system_prompt(env, goal_text, memory=memory), verify_fn=verify,
                )
    finally:
        cleanup_run_dir(run_dir)

    run_id = f"env-{int(time.time() * 1000)}"
    meta = {
        "model": model, "size": env.size, "goal": goal_text,
        "solved": env.solved, "total_reward": env.total_reward,
        "n_steps": traj.n_steps, "termination": traj.termination_reason,
        "visibility_radius": env.visibility_radius, "goal_pos": tuple(env.goal),
        "n_walls": len(env.walls),
    }
    out_dir = Path(".brain-region") / "sandbox"
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_path = write_replay_html(out_dir / f"{run_id}.html", env.frames, meta)  # 显式 utf-8

    result = {
        "run_id": run_id, "model": model, "solved": env.solved,
        "total_reward": env.total_reward, "n_steps": traj.n_steps,
        "termination": traj.termination_reason, "tests_green": traj.tests_green,
        "cost_usd": round(traj.total_main_cost_usd + traj.total_arm_cost_usd, 6),
        "replay": str(replay_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if getattr(args, "debug", False):
        # --debug:跑完后保持调试窗 300s 供回看(/scene 已缓存所有帧,可拖动/播放)。Ctrl+C 提前退。
        # 用 bounded sleep 而非 input() —— input()/isatty() 在后台/非交互语境脆(EOF 即退),sleep 稳。
        print(f"\n>>> run 结束。场景回看 http://127.0.0.1:{debug_port}/scene "
              f"(或静态 replay: {replay_path})")
        print(">>> 调试窗保持 300 秒供回看(Ctrl+C 提前退出)...")
        try:
            for _ in range(600):
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    return result
