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
from brainregion.runtime import merge_usage, normalize_usage
from brainregion.server import _normalize_one, _resolve_endpoints

from .delegation_eval import (
    SandboxExpertSpec,
    render_fixture_delegation_summary,
    run_fixture_delegation_eval,
)
from .delegation_trigger import DelegationTriggerPolicy
from .effort_routing import disabled_effort_shadow_metrics
from .delegation_shadow import (
    render_shadow_gate_summary,
    replay_shadow_report,
)
from .cognitive_eval import render_cognitive_eval_summary, run_cognitive_scaffold_eval
from .eval import render_summary, run_sandbox_eval, write_report
from .functional_region_eval import (
    render_functional_region_eval_summary,
    run_functional_region_eval,
)
from .phase_effort_eval import (
    render_phase_effort_eval_summary,
    run_phase_effort_eval,
)
from .envs import (
    ArcAgiEnv,
    GridWorld,
    UrbanDeliveryEnv,
    build_env_system_prompt,
    generate_urban_delivery_scenario,
    write_replay_html,
)
from .fixtures import SANDBOX_FIXTURES, get_fixture, list_fixture_ids
from .isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from .loop import run_agent, run_cognitive_loop, scoped_env, scoped_memory_mode
from .brain_verify import TraceResult, composite_verify, extract_final_patch, forced_trace
from .task import SandboxTask, WorktreeTask
from .tool_result_eval import render_tool_result_eval_summary, run_tool_result_eval
from .worktree import (
    bootstrap_worktree,
    capture_worktree_diff,
    detect_venv_python,
    worktree,
    write_worktree_run,
)

logger = logging.getLogger("brainregion.sandbox.cli")


def _endpoint_ids_for_refs(dd: dict[str, Any], refs: list[str]) -> set[str]:
    """从 endpoint_id/model 引用提取本次实际需要的 endpoint；裸 provider 模型不命中。"""
    configured = set((dd.get("endpoints") or {}).keys())
    return {
        ref.split("/", 1)[0]
        for ref in refs
        if isinstance(ref, str) and "/" in ref and ref.split("/", 1)[0] in configured
    }


def _build_backend(
    dd: dict[str, Any], *, endpoint_ids: set[str] | None = None,
) -> tuple[LiteLLMBackend, dict[str, Any]]:
    endpoints_cfg = dd.get("endpoints") or {}
    if endpoint_ids is not None:
        endpoints_cfg = {eid: endpoints_cfg[eid] for eid in endpoint_ids if eid in endpoints_cfg}
    registry = _resolve_endpoints(endpoints_cfg)
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
    endpoint_ids = _endpoint_ids_for_refs(dd, [orthogonal_str])
    endpoints_cfg = dd.get("endpoints") or {}
    registry = _resolve_endpoints({eid: endpoints_cfg[eid] for eid in endpoint_ids})
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
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain 必填(或配置 sandbox_main_brain)")
    backend, registry = _build_backend(
        dd, endpoint_ids=_endpoint_ids_for_refs(dd, [model_str]),
    )
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
        "effort_routing_shadow": bool(
            getattr(args, "effort_routing_shadow", False)
        ),
        "effort_routing_active": bool(
            getattr(args, "effort_routing_active", False)
        ),
        "effort_routing_policy": str(
            getattr(args, "effort_routing_policy", "phase")
        ),
        "brain_verify": bool(getattr(args, "brain_verify", False)),
        "brain_delegate": bool(getattr(args, "brain_delegate", False)),
        "cognitive_scaffold": bool(getattr(args, "cognitive_scaffold", False)),
        "cognitive_scaffold_mode": str(
            getattr(args, "cognitive_mode", "runtime_checkpoint")
        ),
        "cognitive_checkpoint_period": int(getattr(args, "checkpoint_period", 3)),
        "tool_result_lifecycle": str(getattr(args, "tool_result_lifecycle", "full")),
        "tool_result_live_reads": int(getattr(args, "tool_result_live_reads", 3)),
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
        if bool(getattr(args, "verification_region", False)) or bool(
            getattr(args, "evidence_region", False)
        ):
            raise SystemExit(
                "--evidence-region/--verification-region 暂不与 --brain-loop 组合；"
                "先在单遍 expert 中验证"
            )
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
    if bool(getattr(args, "verification_region", False)):
        from .regions import VerificationOptionRegion
        kwargs["option_region"] = VerificationOptionRegion()
        kwargs["option_continuous"] = True
        kwargs["max_option_activations"] = max(1, int(kwargs["max_steps"]))
    if bool(getattr(args, "evidence_region", False)):
        from .regions import EvidenceRegion
        kwargs["evidence_region"] = EvidenceRegion()
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
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain 必填(或配置 sandbox_main_brain)")
    backend, registry = _build_backend(
        dd, endpoint_ids=_endpoint_ids_for_refs(dd, [model_str]),
    )
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


def _parse_delegation_experts(
    raw_specs: list[str], registry: dict[str, Any], dd: dict[str, Any]
) -> list[SandboxExpertSpec]:
    if not raw_specs:
        raise SystemExit("at least one --expert REGION=MODEL is required")
    experts: list[SandboxExpertSpec] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise SystemExit(f"invalid --expert {raw!r}; expected REGION=MODEL")
        identity, model_ref = (part.strip() for part in raw.split("=", 1))
        if ":" in identity:
            assignment_id, region = (part.strip() for part in identity.split(":", 1))
        else:
            assignment_id = region = identity
        if not assignment_id or not region or not model_ref:
            raise SystemExit(
                f"invalid --expert {raw!r}; expected REGION=MODEL or ASSIGNMENT:REGION=MODEL"
            )
        entry = _normalize_one(model_ref, set(registry), dd.get("endpoints"))
        experts.append(
            SandboxExpertSpec(
                assignment_id=assignment_id.casefold(),
                region=region,
                question=(
                    f"Analyze the failure from the {region} perspective. Identify the root cause, "
                    "a concrete minimal fix, and verification risks."
                ),
                model=entry["model"],
                endpoint_id=entry.get("endpoint_id"),
            )
        )
    return experts


async def run_delegation_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Run fixture-backed eager and triggered expert delegation experiments."""
    dd = _defaults_mod.apply()
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain is required (or configure sandbox_main_brain)")
    raw_experts = list(args.expert or [])
    expert_refs = [raw.split("=", 1)[1].strip() for raw in raw_experts if "=" in raw]
    refs = [model_str, *expert_refs]
    backend, registry = _build_backend(
        dd,
        endpoint_ids=_endpoint_ids_for_refs(dd, refs),
    )
    main_model, main_endpoint_id = _resolve_main_brain(model_str, registry, dd)
    experts = _parse_delegation_experts(raw_experts, registry, dd)
    tasks = _resolve_tasks(args)
    selected_arms = [arm.strip() for arm in str(args.arms or "").split(",") if arm.strip()] or None
    report = await run_fixture_delegation_eval(
        backend,
        main_model,
        tasks,
        experts,
        main_endpoint_id=main_endpoint_id,
        repeats=int(args.repeats),
        arms=selected_arms,
        max_steps=int(args.max_steps or dd.get("sandbox_max_steps", 10)),
        max_cost_usd=float(args.max_cost_usd or dd.get("sandbox_max_cost_usd", 0.5)),
        temperature=float(dd.get("sandbox_temperature", 0.0)),
        max_tokens=int(args.max_tokens or 2048),
        transcript_token_cap=int(dd.get("sandbox_transcript_token_cap", 24000)),
        consecutive_error_limit=int(dd.get("sandbox_consecutive_error_limit", 3)),
        thinking=_thinking_arg(args),
        effort=args.effort,
        expert_max_context_tokens=int(args.expert_max_context_tokens),
        expert_max_tokens=int(args.expert_max_tokens),
        expert_temperature=float(args.expert_temperature),
        trigger_policy=DelegationTriggerPolicy(
            min_steps_without_effect=int(args.trigger_after_steps),
            min_remaining_steps=int(args.trigger_min_remaining_steps),
        ),
        keep_on_fail=bool(args.keep),
        bootstrap_samples=args.bootstrap_samples,
    )
    path = write_report(report, args.out)
    print(render_fixture_delegation_summary(report))
    print(f"\nReport: {path}")
    return {"report": report, "path": str(path)}


async def run_cognitive_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Run the native-thinking x external-scaffold matched fixture matrix."""
    dd = _defaults_mod.apply()
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain is required (or configure sandbox_main_brain)")
    backend, registry = _build_backend(
        dd,
        endpoint_ids=_endpoint_ids_for_refs(dd, [model_str]),
    )
    model, endpoint_id = _resolve_main_brain(model_str, registry, dd)
    tasks = _resolve_tasks(args)
    selected_arms = [arm.strip() for arm in str(args.arms or "").split(",") if arm.strip()] or None
    try:
        report = await run_cognitive_scaffold_eval(
            backend,
            model,
            tasks,
            endpoint_id=endpoint_id,
            repeats=int(args.repeats),
            arms=selected_arms,
            max_steps=int(args.max_steps or dd.get("sandbox_max_steps", 10)),
            max_cost_usd=float(args.max_cost_usd or dd.get("sandbox_max_cost_usd", 0.5)),
            temperature=float(dd.get("sandbox_temperature", 0.0)),
            max_tokens=int(args.max_tokens or 2048),
            transcript_token_cap=int(dd.get("sandbox_transcript_token_cap", 24000)),
            consecutive_error_limit=int(dd.get("sandbox_consecutive_error_limit", 3)),
            effort=args.effort,
            scaffold_mode=args.scaffold_mode,
            checkpoint_period=int(args.checkpoint_period),
            tool_result_lifecycle=args.tool_result_lifecycle,
            tool_result_live_reads=int(args.tool_result_live_reads),
            bootstrap_samples=args.bootstrap_samples,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    path = write_report(report, args.out)
    print(render_cognitive_eval_summary(report))
    print(f"\nReport: {path}")
    return {"report": report, "path": str(path)}


async def run_phase_effort_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixed-off versus phase-active matched fixture evaluation."""
    dd = _defaults_mod.apply()
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain is required (or configure sandbox_main_brain)")
    backend, registry = _build_backend(
        dd,
        endpoint_ids=_endpoint_ids_for_refs(dd, [model_str]),
    )
    model, endpoint_id = _resolve_main_brain(model_str, registry, dd)
    tasks = _resolve_tasks(args)
    try:
        report = await run_phase_effort_eval(
            backend,
            model,
            tasks,
            endpoint_id=endpoint_id,
            repeats=int(args.repeats),
            max_steps=int(args.max_steps or dd.get("sandbox_max_steps", 10)),
            max_cost_usd=float(args.max_cost_usd or dd.get("sandbox_max_cost_usd", 0.5)),
            max_total_cost_usd=(
                float(args.max_total_cost_usd)
                if args.max_total_cost_usd is not None
                else None
            ),
            temperature=float(dd.get("sandbox_temperature", 0.0)),
            max_tokens=int(args.max_tokens or 2048),
            transcript_token_cap=int(dd.get("sandbox_transcript_token_cap", 24000)),
            consecutive_error_limit=int(dd.get("sandbox_consecutive_error_limit", 3)),
            tool_result_lifecycle=args.tool_result_lifecycle,
            tool_result_live_reads=int(args.tool_result_live_reads),
            active_policy=args.active_policy,
            bootstrap_samples=args.bootstrap_samples,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    path = write_report(report, args.out)
    print(render_phase_effort_eval_summary(report))
    print(f"\nReport: {path}")
    return {"report": report, "path": str(path)}


async def run_functional_regions_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Run the matched passive-context and functional-Region fixture matrix."""
    dd = _defaults_mod.apply()
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain is required (or configure sandbox_main_brain)")
    backend, registry = _build_backend(
        dd,
        endpoint_ids=_endpoint_ids_for_refs(dd, [model_str]),
    )
    model, endpoint_id = _resolve_main_brain(model_str, registry, dd)
    tasks = _resolve_tasks(args)
    selected_arms = [arm.strip() for arm in str(args.arms or "").split(",") if arm.strip()]
    try:
        report = await run_functional_region_eval(
            backend,
            model,
            tasks,
            endpoint_id=endpoint_id,
            repeats=int(args.repeats),
            arms=selected_arms or None,
            max_steps=int(args.max_steps or dd.get("sandbox_max_steps", 10)),
            max_cost_usd=float(args.max_cost_usd or dd.get("sandbox_max_cost_usd", 0.5)),
            temperature=float(dd.get("sandbox_temperature", 0.0)),
            max_tokens=int(args.max_tokens or 2048),
            transcript_token_cap=int(dd.get("sandbox_transcript_token_cap", 24000)),
            consecutive_error_limit=int(dd.get("sandbox_consecutive_error_limit", 3)),
            thinking=_thinking_arg(args) is True,
            effort=args.effort,
            tool_result_lifecycle=args.tool_result_lifecycle,
            tool_result_live_reads=int(args.tool_result_live_reads),
            bootstrap_samples=args.bootstrap_samples,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    path = write_report(report, args.out)
    print(render_functional_region_eval_summary(report))
    print(f"\nReport: {path}")
    return {"report": report, "path": str(path)}


async def run_tool_result_lifecycle_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Run a matched full/compact tool-result lifecycle fixture matrix."""
    dd = _defaults_mod.apply()
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain is required (or configure sandbox_main_brain)")
    backend, registry = _build_backend(
        dd,
        endpoint_ids=_endpoint_ids_for_refs(dd, [model_str]),
    )
    model, endpoint_id = _resolve_main_brain(model_str, registry, dd)
    tasks = _resolve_tasks(args)
    selected_arms = [arm.strip() for arm in str(args.arms or "").split(",") if arm.strip()]
    try:
        report = await run_tool_result_eval(
            backend,
            model,
            tasks,
            endpoint_id=endpoint_id,
            repeats=int(args.repeats),
            arms=selected_arms or None,
            max_steps=int(args.max_steps or dd.get("sandbox_max_steps", 10)),
            max_cost_usd=float(args.max_cost_usd or dd.get("sandbox_max_cost_usd", 0.5)),
            temperature=float(dd.get("sandbox_temperature", 0.0)),
            max_tokens=int(args.max_tokens or 2048),
            transcript_token_cap=int(dd.get("sandbox_transcript_token_cap", 24000)),
            consecutive_error_limit=int(dd.get("sandbox_consecutive_error_limit", 3)),
            thinking=_thinking_arg(args) is True,
            effort=args.effort,
            cognitive_scaffold=bool(args.cognitive_scaffold),
            scaffold_mode=args.scaffold_mode,
            checkpoint_period=int(args.checkpoint_period),
            tool_result_live_reads=int(args.tool_result_live_reads),
            shared_prefix_turns=int(args.shared_prefix_turns),
            bootstrap_samples=args.bootstrap_samples,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    path = write_report(report, args.out)
    print(render_tool_result_eval_summary(report))
    print(f"\nReport: {path}")
    return {"report": report, "path": str(path)}


def run_delegation_shadow(args: argparse.Namespace) -> dict[str, Any]:
    """Replay content-free gate candidates from a saved delegation report."""
    try:
        summary = replay_shadow_report(args.report, max_steps=args.max_steps)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(render_shadow_gate_summary(summary))
    return summary


async def verify_brain(args: argparse.Namespace) -> dict[str, Any]:
    """`brain-region sandbox verify-brain`:§15.8 trace-first + test-backstop 落地。

    对 run.json 里专家(沙盒)的补丁跑一次 forced-trace(廉价 LLM 层),对照该 run **已存的**
    客观 ``tests_green`` → composite(agree / 弱测试信号 / trace 漏检)。不重跑 pytest(用 run 时的
    客观结果),故能在历史 run 上离线复盘。
    """
    dd = _defaults_mod.apply()
    model_str = args.main_brain or dd.get("sandbox_main_brain") or "deepseek-v4-flash"
    backend, registry = _build_backend(
        dd, endpoint_ids=_endpoint_ids_for_refs(dd, [model_str]),
    )
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
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain 必填(或配置 sandbox_main_brain)")
    backend, registry = _build_backend(
        dd, endpoint_ids=_endpoint_ids_for_refs(dd, [model_str]),
    )
    model, endpoint_id = _resolve_main_brain(model_str, registry, dd)

    # 构造 env + 边界校验(constructor 校验 size/visibility_radius/goal/walls;非法 → 干净退出)
    env_name = str(getattr(args, "env", "gridworld") or "gridworld")
    size = int(args.size or (13 if env_name == "urban-delivery" else 5))
    strategy_region_on = bool(getattr(args, "strategy_region", False))  # Phase D.3:策略脑区(多脑区协同)
    memory_region_on = bool(getattr(args, "memory_region", False)) or strategy_region_on  # strategy 隐含 memory
    memory = bool(getattr(args, "memory", False)) or memory_region_on  # 严格部分可观 + recall_map(--memory-region 隐含)
    fog = bool(getattr(args, "fog", False)) or memory  # --memory 自动启用 fog(strict_obs 需要半径)
    vis_radius = getattr(args, "visibility_radius", None)
    if vis_radius is None and fog:
        vis_radius = 2  # --fog/--memory 默认半径 2
    if env_name == "urban-delivery":
        unsupported = {
            "fog": bool(getattr(args, "fog", False)),
            "memory": memory,
            "maze": bool(getattr(args, "maze", False)),
            "ego_actions": bool(getattr(args, "ego_actions", False)),
            "random_goal": bool(getattr(args, "random_goal", False)),
            "explicit_goal": getattr(args, "goal_x", None) is not None or getattr(args, "goal_y", None) is not None,
            "random_walls": getattr(args, "wall_seed", None) is not None,
            "registry": getattr(args, "registry", "none") != "none",
            "visual_ephemeral": bool(getattr(args, "visual_ephemeral", False)),
            "memory_dummy": bool(getattr(args, "memory_dummy", False)),
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise SystemExit("urban-delivery 首版尚未接入这些 GridWorld 开关: " + ", ".join(enabled))
        try:
            scenario = generate_urban_delivery_scenario(
                seed=int(getattr(args, "seed", None) or 0),
                width=size,
                height=size,
                order_count=int(getattr(args, "orders", 3)),
                vehicle_count=int(getattr(args, "vehicles", 2)),
            )
            env = UrbanDeliveryEnv(scenario, visibility_radius=int(vis_radius if vis_radius is not None else 1))
        except ValueError as exc:
            raise SystemExit(f"env 构造非法: {exc}")
        goal_text = args.goal_text or "按顺序完成全部配送订单，并在最后一单后返回商铺 S"
        recommended_steps = int(env.oracle.optimal_total_time * 2) + 20
        max_steps = int(args.max_steps or max(int(dd.get("sandbox_max_steps", 10)), recommended_steps))
    else:
        goal_kw: dict[str, Any] = {"visibility_radius": vis_radius, "strict_obs": memory}
        if bool(getattr(args, "random_goal", False)):
            goal_kw["random_goal_seed"] = getattr(args, "seed", None) if getattr(args, "seed", None) is not None else 0
        elif getattr(args, "goal_x", None) is not None and getattr(args, "goal_y", None) is not None:
            goal_kw["goal"] = (int(args.goal_x), int(args.goal_y))
        wall_seed = getattr(args, "wall_seed", None)
        if wall_seed is not None:
            goal_kw["random_walls_seed"] = wall_seed
            goal_kw["wall_density"] = float(getattr(args, "wall_density", None) or 0.2)  # 默认密度 0.2
        # Phase 4.5 迷宫地形:maze_seed 用 --seed;覆盖 random_walls。fog 由 --memory-region/--fog 控制
        if bool(getattr(args, "maze", False)):
            goal_kw["maze_seed"] = getattr(args, "seed", None) if getattr(args, "seed", None) is not None else 0
            goal_kw["maze_braid"] = float(getattr(args, "maze_braid", 0.2) or 0.2)
        if bool(getattr(args, "ego_actions", False)):  # Phase 4.8 ego-relative action
            goal_kw["ego_actions"] = True
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

    task = SandboxTask(id=f"env-{env_name}-{size}x{size}", goal=goal_text)

    def verify(t, run_dir, *, python_exe=None):  # env-grounded,返完整 verify_solution shape
        return {
            "tests_green": bool(env.solved),
            "solve_status": "solved" if env.solved else "tests_fail",
            "pytest": None,
            "gold_diff": getattr(t, "gold_diff", ""),
        }

    memory_region = None
    if memory_region_on:  # Phase D.2:有状态记忆脑区(代码 dead-reckon + LLM rough_map)
        from brainregion.sandbox.regions import MemoryRegion
        memory_region = MemoryRegion(
            start=env.start, log_len=int(getattr(args, "memory_log_len", 32) or 32),
            dummy=bool(getattr(args, "memory_dummy", False)),   # Phase 4.4:matched-source dummy 内容控制
        )
    strategy_region = None
    if strategy_region_on:  # Phase D.3:策略脑区(读 memory.rough_map 规划,多脑区协同)
        from brainregion.sandbox.regions import StrategyRegion
        strategy_region = StrategyRegion()
    # Phase 4.2/4.3 旋钮(单 episode 复刻 env-eval 臂配置,供 --debug 可视化)
    visual_ephemeral = bool(getattr(args, "visual_ephemeral", False))
    registry_mode = getattr(args, "registry", "none") or "none"

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
                    effort_routing_shadow=bool(
                        getattr(args, "effort_routing_shadow", False)
                    ),
                    effort_routing_active=bool(
                        getattr(args, "effort_routing_active", False)
                    ),
                    effort_routing_policy=str(
                        getattr(args, "effort_routing_policy", "phase")
                    ),
                    system_prompt=build_env_system_prompt(
                        env, goal_text, memory=memory, strategy=strategy_region_on,
                        registry=registry_mode,   # Phase 4.3 脑区注册表块(none/cap/full)
                    ), verify_fn=verify,
                    memory_region=memory_region, strategy_region=strategy_region,
                    visual_ephemeral=visual_ephemeral,   # Phase 4.2:剥历史视觉(只留最新 <visual>)
                )
    finally:
        cleanup_run_dir(run_dir)

    run_id = f"env-{int(time.time() * 1000)}"
    meta: dict[str, Any] = {
        "model": model, "env": env_name, "size": size, "goal": goal_text,
        "solved": env.solved, "total_reward": env.total_reward,
        "n_steps": traj.n_steps, "termination": traj.termination_reason,
        "visibility_radius": env.visibility_radius,
        "memory_region": memory_region_on, "strategy_region": strategy_region_on,
        "visual_ephemeral": visual_ephemeral, "registry": registry_mode,
        "memory_dummy": bool(getattr(args, "memory_dummy", False)),
    }
    if isinstance(env, UrbanDeliveryEnv):
        meta.update({
            "orders": len(env.scenario.orders),
            "vehicles": len(env.scenario.vehicles),
            "delivery_metrics": env.metrics(),
        })
    else:
        meta.update({"goal_pos": tuple(env.goal), "n_walls": len(env.walls)})
    out_dir = Path(".brain-region") / "sandbox"
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_path = write_replay_html(out_dir / f"{run_id}.html", env.frames, meta)  # 显式 utf-8

    result = {
        "run_id": run_id, "model": model, "solved": env.solved,
        "total_reward": env.total_reward, "n_steps": traj.n_steps,
        "termination": traj.termination_reason, "tests_green": traj.tests_green,
        "cost_usd": round(traj.total_main_cost_usd + traj.total_arm_cost_usd, 6),
        "main_usage": normalize_usage(traj.total_main_usage),
        "region_usage": normalize_usage(traj.total_arm_usage),
        "total_usage": merge_usage(traj.total_main_usage, traj.total_arm_usage),
        "main_cost_sources": list(traj.main_cost_sources),
        "region_cost_sources": list(traj.arm_cost_sources),
        "phase_control": (
            traj.phase_controller.snapshot() if traj.phase_controller else {"enabled": False}
        ),
        "effort_routing_shadow": (
            traj.effort_routing_shadow.snapshot()
            if traj.effort_routing_shadow
            else disabled_effort_shadow_metrics()
        ),
        "replay": str(replay_path),
    }
    if isinstance(env, UrbanDeliveryEnv):
        result["delivery_metrics"] = env.metrics()
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


async def run_arc_env(args: argparse.Namespace) -> dict[str, Any]:
    """Run a content-neutral main-brain baseline on a public ARC-AGI-3 game."""

    dd = _defaults_mod.apply()
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain is required (or configure sandbox_main_brain)")
    if int(args.max_steps) < 1:
        raise SystemExit("--max-steps must be positive")
    if float(args.max_cost_usd) <= 0:
        raise SystemExit("--max-cost-usd must be positive")

    backend, registry = _build_backend(
        dd,
        endpoint_ids=_endpoint_ids_for_refs(dd, [model_str]),
    )
    model, endpoint_id = _resolve_main_brain(model_str, registry, dd)
    goal = args.goal_text or (
        "Explore the unfamiliar environment, infer useful goals and action effects from observations, "
        "and complete as many levels as possible efficiently."
    )
    try:
        env = ArcAgiEnv.create(str(args.game), seed=int(args.seed))
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    task = SandboxTask(id=f"arc-agi-3-{args.game}", goal=goal)

    def verify(t, run_dir, *, python_exe=None):
        del t, run_dir, python_exe
        return {
            "tests_green": env.solved,
            "solve_status": "solved" if env.solved else "tests_fail",
            "pytest": None,
            "gold_diff": "",
        }

    run_dir = make_run_dir(prefix="brainregion-arc-agi-")
    try:
        with scoped_env(env):
            trajectory = await run_agent(
                backend,
                model,
                task,
                run_dir=run_dir,
                max_steps=int(args.max_steps),
                max_env_actions=int(args.max_steps),
                max_cost_usd=float(args.max_cost_usd),
                temperature=float(dd.get("sandbox_temperature", 0.0)),
                max_tokens=int(args.max_tokens),
                transcript_token_cap=int(dd.get("sandbox_transcript_token_cap", 24000)),
                consecutive_error_limit=int(dd.get("sandbox_consecutive_error_limit", 3)),
                endpoint_id=endpoint_id,
                thinking=_thinking_arg(args),
                effort=args.effort,
                system_prompt=env.build_system_prompt(goal),
                verify_fn=verify,
                visual_ephemeral=True,
                tool_result_lifecycle=args.tool_result_lifecycle,
                tool_result_live_reads=int(args.tool_result_live_reads),
                initial_observation=env.observation(),
            )
        snapshot = env.snapshot()
        run_id = f"arc-agi-3-{int(time.time() * 1000)}"
        progress_trace = trajectory.progress_trace
        operation_counts: dict[str, int] = {}
        for progress in progress_trace:
            operation = str(progress.get("operation") or "model_turn")
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
        environment_actions = operation_counts.get("act", 0)
        error_steps = sum(bool(progress.get("error")) for progress in progress_trace)
        action_denominator = max(1, environment_actions)
        normalized_usage = normalize_usage(trajectory.total_main_usage)
        result = {
            "run_id": run_id,
            "mode": "arc_agi_3_public_baseline",
            "game_id": snapshot.get("game_id"),
            "model": model,
            "endpoint_id": endpoint_id,
            "solved": env.solved,
            "state": snapshot.get("state"),
            "levels_completed": snapshot.get("levels_completed"),
            "win_levels": snapshot.get("win_levels"),
            "environment_actions": environment_actions,
            "model_steps": trajectory.n_steps,
            "operation_counts": operation_counts,
            "error_steps": error_steps,
            "environment_action_rate": round(
                environment_actions / max(1, trajectory.n_steps), 4
            ),
            "tokens_per_environment_action": round(
                int(normalized_usage.get("total_tokens", 0)) / action_denominator,
                2,
            ),
            "cost_per_environment_action_usd": round(
                trajectory.total_main_cost_usd / action_denominator, 6
            ),
            "termination": trajectory.termination_reason,
            "cost_usd": round(trajectory.total_main_cost_usd, 6),
            "usage": normalized_usage,
            "input_attribution": trajectory.main_input_attribution,
            "tool_result_lifecycle": trajectory.tool_result_lifecycle,
            "workspace_effects": trajectory.workspace_effects,
            "interaction_trace": list(env.action_trace),
            "contains_reasoning": False,
            "contains_frame_content": False,
        }
        report_path = write_report(result, Path(".brain-region") / "arc-agi" / "runs")
        result["report"] = str(report_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        env.close()
        cleanup_run_dir(run_dir)


def _parse_arm_spec(spec: str):
    """``--arm mem=region,strat=real`` → EnvArm(feature-config;review 双强 feature-flag 表层)。

    可选 ``dummy=1``(Phase 4.4 matched-source dummy 记忆)/ ``registry=cap|full`` / ``eph=1`` /
    ``metronome=1`` / ``nav=oracle|grounded``。
    """
    from .env_eval import EnvArm
    parts: dict[str, str] = {}
    for chunk in spec.split(","):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            parts[k.strip()] = v.strip()
    mem = parts.get("mem", "").lower()
    strat = parts.get("strat", "none").lower()
    if strat not in ("none", "real", "echo", "dummy"):
        raise SystemExit(f"--arm strat 非法 {strat!r};合法:none/real/echo/dummy")
    registry = parts.get("registry", "none").lower()
    if registry not in ("none", "cap", "full"):
        raise SystemExit(f"--arm registry 非法 {registry!r};合法:none/cap/full")
    name = parts.get("name") or f"mem={mem or 'none'},strat={strat}"
    return EnvArm(
        name=name,
        memory_tool=(mem == "tool"),
        memory_region=mem in ("region", "true", "1"),
        strategy=strat,
        metronome=parts.get("metronome", "").lower() in ("1", "true", "yes"),
        visual_ephemeral=parts.get("eph", "").lower() in ("1", "true", "yes"),
        registry=registry,
        memory_dummy=parts.get("dummy", "").lower() in ("1", "true", "yes"),
        navigation_delegate=parts.get("nav", "").lower() in ("delegate", "oracle", "1", "true", "yes"),
        navigation_grounded=parts.get("nav", "").lower() == "grounded",
    )


async def run_env_eval(args: argparse.Namespace) -> dict[str, Any]:
    """`brain-region sandbox env-eval`(Phase 4):多 run × arms(Echo 控制臂)+ 过程指标 + config 级 bootstrap。

    非交互(不开 --debug)。产 JSON 报告 + CSV + markdown 总结。判据 = harness 跑通(三臂都产数据、过程指标非零、
    pairwise delta+CI 有值、cost 不超 cap)—— **非**「Strategy 显著」结论(N 小,pilot_ 前缀,signal_regime 看 regime)。
    """
    from .env_eval import (
        ARM_PRESETS, EnvConfig, run_env_eval as run_env_eval_harness,
        render_env_eval_summary, write_report,
    )

    dd = _defaults_mod.apply()
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain 必填(或配置 sandbox_main_brain)")
    backend, registry = _build_backend(
        dd, endpoint_ids=_endpoint_ids_for_refs(dd, [model_str]),
    )
    model, endpoint_id = _resolve_main_brain(model_str, registry, dd)

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    vis_radius = int(args.visibility_radius) if args.visibility_radius is not None else 2
    maze_on = bool(getattr(args, "maze", False))
    maze_braid = float(getattr(args, "maze_braid", 0.2) or 0.2)
    action_budget = int(args.max_steps or dd.get("sandbox_max_steps", 30))
    main_turn_cap = getattr(args, "max_main_turns", None)
    main_turn_cap = int(main_turn_cap) if main_turn_cap is not None else None
    if action_budget < 1:
        raise SystemExit("--max-steps 须为正整数")
    if main_turn_cap is not None and main_turn_cap < 1:
        raise SystemExit("--max-main-turns 须为正整数")
    configs = [
        EnvConfig(
            size=sz, seed=sd,
            wall_seed=getattr(args, "wall_seed", None) if not maze_on else None,
            wall_density=float(args.wall_density or 0.0) if not maze_on else 0.0,
            visibility_radius=vis_radius,
            max_steps=action_budget, max_main_turns=main_turn_cap,
            maze=maze_on, maze_braid=maze_braid,
            ego_actions=bool(getattr(args, "ego_actions", False)),  # Phase 4.8
        )
        for sz in sizes for sd in seeds
    ]
    if len(configs) < 2:
        logger.warning("configs<2(sizes×seeds=%d)→ bootstrap 退化为 None,pairwise gate INCONCLUSIVE", len(configs))

    if getattr(args, "arm", None):  # 显式 feature-config(覆盖预设)
        arms = tuple(_parse_arm_spec(s) for s in args.arm)
    else:
        preset = args.arms or "memory-strategy"
        if preset not in ARM_PRESETS:
            raise SystemExit(f"--arms 非法 {preset!r};合法:{list(ARM_PRESETS)}")
        arms = ARM_PRESETS[preset]

    logger.info("env-eval: %d configs × %d arms × %d repeats = ≤%d runs",
                len(configs), len(arms), args.repeats, len(configs) * len(arms) * args.repeats)

    status_period = int(getattr(args, "metronome_period", 3) or 3)
    if status_period <= 0:  # review 双强:period 须正(避 ZeroDivisionError + 语义明确)
        raise SystemExit("--metronome-period 须为正整数")

    report = await run_env_eval_harness(
        backend, model, configs, arms,
        repeats=int(args.repeats),
        max_cost_usd=float(args.max_cost_usd or dd.get("sandbox_max_cost_usd", 2.0)),
        temperature=float(dd.get("sandbox_temperature", 0.0)),
        max_tokens=int(args.max_tokens or 2048),
        endpoint_id=endpoint_id, thinking=_thinking_arg(args), effort=args.effort,
        status_period=status_period,
    )
    json_path, csv_path = write_report(report, getattr(args, "out", None))
    print(render_env_eval_summary(report))
    print(f"\n报告 JSON: {json_path}\nper-run CSV: {csv_path}")
    return {"report": report, "json": str(json_path), "csv": str(csv_path)}


async def run_delivery_eval(args: argparse.Namespace) -> dict[str, Any]:
    """`brain-region sandbox delivery-eval`:配送主脑/导航执行脑区成对评测。"""
    from .urban_delivery_eval import (
        DeliveryEvalConfig,
        build_delivery_env,
        render_delivery_summary,
        run_delivery_eval as run_harness,
        write_delivery_report,
    )

    dd = _defaults_mod.apply()
    model_str = args.main_brain or dd.get("sandbox_main_brain") or ""
    if not model_str:
        raise SystemExit("--main-brain 必填(或配置 sandbox_main_brain)")
    backend, registry = _build_backend(
        dd, endpoint_ids=_endpoint_ids_for_refs(dd, [model_str]),
    )
    model, endpoint_id = _resolve_main_brain(model_str, registry, dd)

    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not sizes or not seeds:
        raise SystemExit("--sizes 和 --seeds 至少各包含一个值")
    configs = [
        DeliveryEvalConfig(
            size=size,
            seed=seed,
            orders=int(args.orders),
            vehicles=int(args.vehicles),
            visibility_radius=int(args.visibility_radius),
            max_env_actions=int(args.max_env_actions),
            max_main_turns=(int(args.max_main_turns) if args.max_main_turns is not None else None),
        )
        for size in sizes
        for seed in seeds
    ]
    try:
        for config in configs:
            build_delivery_env(config)
    except ValueError as exc:
        raise SystemExit(f"delivery-eval 配置非法: {exc}")
    if int(args.repeats) < 1:
        raise SystemExit("--repeats 须为正整数")
    if int(args.max_env_actions) < 1:
        raise SystemExit("--max-env-actions 须为正整数")
    if args.max_main_turns is not None and int(args.max_main_turns) < 1:
        raise SystemExit("--max-main-turns 须为正整数")
    if not (1 <= int(args.option_actions) <= 16):
        raise SystemExit("--option-actions 须在 1..16")
    if len(configs) < 2:
        logger.warning("delivery configs<2 → bootstrap CI 退化；仅作 smoke/pilot")

    resume_report = None
    if args.resume_report:
        resume_path = Path(args.resume_report).expanduser().resolve()
        if not resume_path.is_file():
            raise SystemExit(f"--resume-report 文件不存在: {resume_path}")
        try:
            resume_report = json.loads(resume_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"--resume-report 读取失败: {exc}") from exc
        if not isinstance(resume_report, dict):
            raise SystemExit("--resume-report 须为 JSON object")

    try:
        report = await run_harness(
            backend,
            model,
            configs,
            repeats=int(args.repeats),
            max_cost_usd=float(args.max_cost_usd or dd.get("sandbox_max_cost_usd", 2.0)),
            temperature=float(dd.get("sandbox_temperature", 0.0)),
            max_tokens=int(args.max_tokens or 2048),
            endpoint_id=endpoint_id,
            thinking=_thinking_arg(args),
            effort=args.effort,
            option_actions=int(args.option_actions),
            resume_report=resume_report,
        )
    except ValueError as exc:
        if resume_report is None:
            raise
        raise SystemExit(f"--resume-report 不兼容: {exc}") from exc
    json_path, csv_path = write_delivery_report(report, args.out)
    print(render_delivery_summary(report))
    print(f"\n报告 JSON: {json_path}\nper-run CSV: {csv_path}")
    return {"report": report, "json": str(json_path), "csv": str(csv_path)}
