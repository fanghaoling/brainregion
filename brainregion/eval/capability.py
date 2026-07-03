"""NP 能力基准(Phase 1):程序化 3-SAT 能力阶梯 + 客观验证 harness。

换**确定性客观验证**(跑验证器、不调 judge)拿干净信号(脱离主观 judge 的 INCONCLUSIVE);换
**hard-solve regime**(容量吃紧的 3-SAT)测**注入 context 对 solve-rate 的影响**。

⚠️ 声明范围:注入的 memory 是**策略陈述/指令式**文本 → 测 **instruction interference**(嵌入 context 的
算法指引对 hard 推理的影响),**非** diffuse context pollution。措辞用「嵌入 context 的策略指引影响
hard SAT 推理」。3-SAT 相变 α≈4.26 作难度旋钮(均匀随机 CNF + rejection-sampling 留 SAT,故 α 有效)。

复用:LiteLLMBackend(complete)/ _normalize_one(solver 解析)/ store ledger(无新表)/ bootstrap_statistic。
不重用 consult/judge/calibration。**无新依赖**(DPLL 自写)。

分层指标隔离 format-fighting:parse_ok / valid_output(complete) / solved / call_failed(excluded);
主 gap = solve_rate_given_valid。matched-pair(同 instance 跨 arm)+ 配对 bootstrap + effect size
(risk difference / odds ratio)。output_tokens × arm 作搜索成本副信号。
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from . import store
from .metadata import defaults_hash, git_sha
from .schema import EvalCaseRecord, EvalLedgerEntry
from .stats import bootstrap_statistic, seed_for

logger = logging.getLogger("brainregion.eval.capability")

# literal 约定(SAT 标准):正整数 +i = x_i;负整数 -i = ¬x_i。变量 i ∈ [1, n_vars]。
# clause = 3 个**不同变量**的 literal 析取。assignment: dict[int, bool](全变量)。

_ARM_NAMES = ("baseline", "relevant", "neutral", "distractor")


# ── DPLL + 生成 + 验证(纯函数)──────────────────────────────────────────────────

@dataclass(frozen=True)
class SatInstance:
    """一个可满足 3-SAT 实例(rejection-sampled,带 DPLL 真 witness + 难度计数器)。"""

    n_vars: int
    clauses: tuple[tuple[int, ...], ...]    # 每子句 3 signed-int literal(不同变量)
    alpha: float                            # n_clauses / n_vars
    witness: tuple[tuple[int, bool], ...]   # 一个满足赋值(var → bool),全变量
    conflicts: int                          # DPLL 冲突数(难度 proxy)
    decisions: int
    propagations: int
    max_depth: int
    seed: int

    @property
    def task_id(self) -> str:
        return f"sat-n{self.n_vars}-a{self.alpha}-s{self.seed}"


def _random_3cnf(n_vars: int, n_clauses: int, rng: random.Random) -> tuple[tuple[int, ...], ...]:
    """均匀随机 3-CNF:n_clauses 子句,每子句 3 个不同变量 + 随机极性。"""
    clauses: list[tuple[int, ...]] = []
    for _ in range(n_clauses):
        vs = rng.sample(range(1, n_vars + 1), 3)
        clauses.append(tuple(v if rng.random() < 0.5 else -v for v in vs))
    return tuple(clauses)


def dpll(clauses, n_vars: int, *, max_decisions: int = 50000):
    """DPLL + unit propagation。返回 (sat, witness|None, conflicts, decisions, propagations, max_depth)。

    sat=False = 证明 UNSAT 或超 max_decisions(生成时都视作「该候选不要」)。witness 为全变量赋值。
    计数器(conflicts/decisions/propagations/max_depth)作 instance-hardness proxy(未来 hardness 回归)。
    """
    stats = {"conflicts": 0, "decisions": 0, "propagations": 0, "max_depth": 0}
    clause_list = [tuple(c) for c in clauses]

    def _prop(assign: dict) -> bool:
        """unit propagation(原地改 assign)。遇空子句 → False(conflict)。"""
        changed = True
        while changed:
            changed = False
            for cl in clause_list:
                unassigned: list[int] = []
                sat = False
                for lit in cl:
                    v = abs(lit)
                    if v in assign:
                        if assign[v] == (lit > 0):
                            sat = True
                            break
                    else:
                        unassigned.append(lit)
                if sat:
                    continue
                if not unassigned:
                    return False          # 空子句(conflict)
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    assign[abs(lit)] = lit > 0
                    stats["propagations"] += 1
                    changed = True
        return True

    def _all_sat(assign: dict) -> bool:
        for cl in clause_list:
            if not any((lit > 0 and assign.get(lit) is True) or
                       (lit < 0 and assign.get(-lit) is False) for lit in cl):
                return False
        return True

    def _recurse(assign: dict, depth: int):
        stats["max_depth"] = max(stats["max_depth"], depth)
        a = dict(assign)
        if not _prop(a):
            stats["conflicts"] += 1
            return None
        if _all_sat(a):
            return a
        chosen = next((abs(lit) for cl in clause_list for lit in cl if abs(lit) not in a), None)
        if chosen is None:
            return a if _all_sat(a) else None
        if stats["decisions"] >= max_decisions:
            stats["conflicts"] += 1
            return None
        stats["decisions"] += 1
        for val in (True, False):
            a2 = dict(a)
            a2[chosen] = val
            res = _recurse(a2, depth + 1)
            if res is not None:
                return res
        stats["conflicts"] += 1
        return None

    witness = _recurse({}, 0)
    if witness is None:
        return False, None, stats["conflicts"], stats["decisions"], stats["propagations"], stats["max_depth"]
    full = {v: witness.get(v, False) for v in range(1, n_vars + 1)}    # 补全未 forced 变量
    return True, full, stats["conflicts"], stats["decisions"], stats["propagations"], stats["max_depth"]


def gen_3sat(n_vars: int, alpha: float, seed: int, *, max_attempts: int = 200) -> SatInstance:
    """均匀随机 3-CNF → DPLL → rejection-sample 留 SAT(带真 witness)。

    α≈4.26 相变对**均匀随机**分布成立(rejection-sampling 不引入 planted 偏差)。max_attempts 超限 raise
    (高 α SAT 实例稀少)。确定性:同 seed → 同实例序列 → 同首个 SAT 实例。
    """
    if n_vars < 3:
        raise ValueError(f"n_vars>=3 需要(3-SAT 每子句 3 不同变量),got {n_vars}")
    n_clauses = max(1, round(alpha * n_vars))
    rng = random.Random(seed)
    for _ in range(max_attempts):
        clauses = _random_3cnf(n_vars, n_clauses, rng)
        sat, witness, conflicts, decisions, propagations, max_depth = dpll(clauses, n_vars)
        if sat and witness is not None:
            return SatInstance(
                n_vars=n_vars, clauses=clauses, alpha=float(alpha),
                witness=tuple(sorted(witness.items())),
                conflicts=conflicts, decisions=decisions, propagations=propagations,
                max_depth=max_depth, seed=seed,
            )
    raise RuntimeError(
        f"gen_3sat: {max_attempts} 次未生成 SAT 实例(n_vars={n_vars}, α={alpha});该 α 可能 SAT 过稀"
    )


def verify_sat(assignment: dict, clauses) -> bool:
    """客观验证器:每子句被满足 → True。assignment: dict[int,bool](caller 保证全变量 + bool 值)。

    缺变量/非布尔 → 该 literal 不满足;若致子句全不满足 → False(partial/invalid 自然判否)。
    """
    for cl in clauses:
        ok = False
        for lit in cl:
            v = abs(lit)
            val = assignment.get(v)
            if val is None:
                continue
            if (lit > 0 and val is True) or (lit < 0 and val is False):
                ok = True
                break
        if not ok:
            return False
    return True


# ── solve prompt + schema 解析 + manipulation-check ─────────────────────────────

def _var_names(n_vars: int) -> list[str]:
    return [f"x{i}" for i in range(1, n_vars + 1)]


def render_solve_prompt(instance: SatInstance, memory_blocks: list[dict]) -> tuple[str, str]:
    """→ (system, user)。memory_blocks: [{label, text}],注入为带唯一标签的参考 note。

    「本实例可满足,给一个满足赋值」绑定 SAT-only 族(非全局硬禁 UNSAT)。严格 JSON schema 覆盖全变量。
    """
    clauses_str = json.dumps([[int(lit) for lit in c] for c in instance.clauses])
    var_list = ", ".join(_var_names(instance.n_vars))
    notes = ""
    if memory_blocks:
        lines = ["参考笔记(可考虑应用;方括号为唯一标签):"]
        for b in memory_blocks:
            lines.append(f"[{b['label']}] {b['text']}")
        notes = "\n".join(lines) + "\n\n"
    system = (
        "你是 SAT 求解器。给定 3-CNF 公式,找一个满足赋值。**该实例保证可满足**(禁止答 UNSAT)。\n"
        "signed int:+i = x_i 为真,-i = ¬x_i(x_i 为假)。\n"
        f"只输出 JSON:{{\"assign\":{{\"x1\":true,\"x2\":false,...}}}},覆盖全部 {instance.n_vars} 个变量,"
        "值必须为布尔。不要解释、不要 markdown。"
    )
    user = (
        f"{notes}变量: {var_list}\n"
        f"子句(CNF,signed int): {clauses_str}\n"
        f"输出 JSON {{\"assign\":{{...}}}}(覆盖全部 {instance.n_vars} 个变量)。"
    )
    return system, user


def parse_assignment(content: str, n_vars: int) -> dict[int, bool] | None:
    """程序化 schema 校验(评审):抽 JSON → 顶层 assign → 键 == {x1..xn} 精确 → 值 bool → 转 int。

    malformed/缺变量/余变量/非布尔/越界 → None(valid_output=0)。容 markdown fence。
    """
    if not content:
        return None
    text = content.strip()
    # 去 markdown fence
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # 抽首个 {...}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or "assign" not in obj:
        return None
    assign = obj["assign"]
    if not isinstance(assign, dict):
        return None
    expected = set(_var_names(n_vars))
    if set(assign.keys()) != expected:
        return None                       # 缺/余变量
    out: dict[int, bool] = {}
    for k, v in assign.items():
        # 仅接受真布尔(True/False),拒 "true"/1/0 字符串
        if isinstance(v, bool):
            out[int(k[1:])] = v
        else:
            return None
    return out


def parse_claimed_notes(content: str, injected_labels: list[str]) -> list[str]:
    """manipulation-check:从「你考虑了哪些策略(按标签)?」回答里抽被 mention 的标签。

    诚实命名 claimed(模型*声称*用,非因果 use)。容自由文本:扫标签字面。
    """
    if not content:
        return []
    mentioned = []
    for label in injected_labels:
        if label in content:
            mentioned.append(label)
    return mentioned


# ── 记忆臂 + seed 解析 ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CapabilityVariant:
    """4 臂之一。memory_role 决定注入哪些 seed block。"""
    name: str
    memory_role: Literal["baseline", "relevant", "neutral", "distractor"]


def build_capability_variants(arms: list[str] | None = None) -> list[CapabilityVariant]:
    """arms 子集 → CapabilityVariant 列表。None=全 4 臂;空 list/全非法 → ValueError(fail-fast)。"""
    wanted = list(_ARM_NAMES) if arms is None else arms
    out: list[CapabilityVariant] = []
    for a in _ARM_NAMES:
        if a in wanted:
            out.append(CapabilityVariant(name=f"memory_{a}" if a != "baseline" else "baseline",
                                         memory_role=a))
    if not out:
        raise ValueError(f"arms 为空或全非法(合法: {_ARM_NAMES})")
    return out


def load_memory_seeds(path: str) -> dict:
    """memory_seeds.yaml → {relevant:[...], neutral:[...], distractor_candidates:[...]}。

    schema 校验 fail-fast(评审):缺 key / block 缺 label|text / 候选池空 → ValueError,不静默退化成 baseline。
    """
    import yaml
    from pathlib import Path
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key in ("relevant", "neutral", "distractor_candidates"):
        if key not in data:
            raise ValueError(f"memory_seeds 缺 key: {key}")
        if not isinstance(data[key], list):
            raise ValueError(f"memory_seeds.{key} 必须是 list")
    if not data["distractor_candidates"]:
        raise ValueError("memory_seeds.distractor_candidates 空(pilot 无候选可筛)")
    for key in ("relevant", "neutral", "distractor_candidates"):
        for i, b in enumerate(data[key]):
            if not isinstance(b, dict) or not b.get("label") or not b.get("text"):
                raise ValueError(f"memory_seeds.{key}[{i}] 缺 label/text")
    return data


def resolve_memory_blocks(
    role: str, seeds: dict, distractor_label: str | None,
) -> list[dict]:
    """臂 → 注入的 memory block 列表。

    - baseline:[]
    - relevant:relevant seeds
    - neutral:relevant + neutral(length-matched benign;控 token)
    - distractor:relevant + 选定 distractor 候选(adversarial 错向)
    """
    if role == "baseline":
        return []
    rel = list(seeds["relevant"])
    if role == "relevant":
        return rel
    if role == "neutral":
        return rel + list(seeds["neutral"])
    if role == "distractor":
        cand = next((c for c in seeds["distractor_candidates"]
                     if distractor_label is None or c["label"] == distractor_label),
                    seeds["distractor_candidates"][0])
        return rel + [cand]
    raise ValueError(f"未知 memory_role: {role}")


def injected_labels(blocks: list[dict]) -> list[str]:
    return [b["label"] for b in blocks]


# ── runner ──────────────────────────────────────────────────────────────────────

@dataclass
class CapabilityCase:
    """单 (instance, arm, solver) 的产出(喂 store 走 EvalCaseRecord adapter)。"""
    run_id: str
    task_id: str
    arm: str
    solver: str
    n_vars: int
    alpha: float
    memory_role: str
    blocks_injected: int
    parse_ok: int = 0          # 抽到 JSON
    valid_output: int = 0      # complete(全变量 + bool)
    solved: int = 0            # valid 且 verify_sat 过
    call_failed: int = 0       # backend error / 空 content(excluded,不计 solve 分母)
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    claimed_notes: list = field(default_factory=list)   # manip-check 标签
    error: str = ""

    def to_case_record(self) -> EvalCaseRecord:
        return EvalCaseRecord(
            run_id=self.run_id, task_id=self.task_id, variant=self.arm,
            report_summary={
                "solver": self.solver, "n_vars": self.n_vars, "alpha": self.alpha,
                "memory_role": self.memory_role, "blocks_injected": self.blocks_injected,
                "parse_ok": self.parse_ok, "valid_output": self.valid_output,
                "solved": self.solved, "call_failed": self.call_failed,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "claimed_notes": self.claimed_notes,
            },
            retrieved_case_ids=[],
            cost={"inference_usd": self.cost_usd, "input_tokens": self.input_tokens,
                  "output_tokens": self.output_tokens},
            latency_ms=0.0,
            outputs_json=json.dumps({"arm": self.arm, "solver": self.solver}, ensure_ascii=False),
            error=self.error,
        )


def _token_counts(usage: dict) -> tuple[int, int, int]:
    """→ (prompt_tokens, completion_tokens, reasoning_tokens)。reasoning 取自 completion_tokens_details
    (GPT④:推理预算是搜索成本副信号——accuracy 不变但 reasoning 翻倍 / 被 reasoning 吃光 = 另一种干扰)。"""
    if not isinstance(usage, dict):
        return 0, 0, 0
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    reasoning = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
    return prompt, completion, reasoning


async def _solve_one(
    backend, solver_entry: dict, instance: SatInstance, blocks: list[dict],
    *, manipulation_check: bool, max_tokens: int, effort,
) -> CapabilityCase:
    """单 instance × arm × solver 的一次求解(含 manip-check 探针)。outcomes 分类记录。"""
    rec = CapabilityCase(
        run_id="", task_id=instance.task_id, arm="", solver=solver_entry["model"],
        n_vars=instance.n_vars, alpha=instance.alpha, memory_role="",
        blocks_injected=len(blocks),
    )
    system, user = render_solve_prompt(instance, blocks)
    try:
        resp = await backend.complete(
            model=solver_entry["model"], system=system, user=user,
            temperature=0.0, max_tokens=max_tokens, effort=effort,
            endpoint_id=solver_entry.get("endpoint_id"),
        )
    except Exception as e:  # noqa: BLE001 — backend 已内部隔离,双保险
        rec.call_failed = 1
        rec.error = f"{type(e).__name__}: {e}"
        return rec
    rec.input_tokens, rec.output_tokens, rec.reasoning_tokens = _token_counts(getattr(resp, "usage", {}) or {})
    rec.cost_usd = float(getattr(resp, "cost_usd", None) or 0.0)
    if getattr(resp, "error", None) or not getattr(resp, "content", ""):
        rec.call_failed = 1                      # excluded(不计 solve 分母)
        rec.error = getattr(resp, "error", "") or "empty content"
        return rec
    rec.parse_ok = 1
    assign = parse_assignment(resp.content, instance.n_vars)
    if assign is None:
        return rec                                # valid_output=0(format 失败)
    rec.valid_output = 1
    rec.solved = 1 if verify_sat(assign, instance.clauses) else 0
    if manipulation_check and blocks:
        rec.claimed_notes = await _manip_probe(backend, solver_entry, blocks, max_tokens, effort)
    return rec


async def _manip_probe(backend, solver_entry: dict, blocks: list[dict], max_tokens, effort) -> list[str]:
    """事后追问「考虑了哪些策略(按标签)?」→ 抽被 mention 标签(self-report,mention ≠ 因果 use)。"""
    labels = injected_labels(blocks)
    system = "如实回答。只列标签或 'none'。"
    user = "你刚才解题时考虑了哪些参考笔记?按方括号标签列出(如 " + ", ".join(labels) + "),或 'none'。"
    try:
        resp = await backend.complete(
            model=solver_entry["model"], system=system, user=user,
            temperature=0.0, max_tokens=max_tokens, effort=effort,
            endpoint_id=solver_entry.get("endpoint_id"),
        )
        return parse_claimed_notes(getattr(resp, "content", "") or "", labels)
    except Exception:  # noqa: BLE001
        return []


async def run_capability_eval(
    *,
    n_vars: int, alphas: list[float], n_instances: int, base_seed: int,
    variants: list[CapabilityVariant], solver_entries: list[dict], backend,
    seeds: dict, run_id: str,
    distractor_label: str | None = None,
    max_cost_usd: float = 5.0, effort=None, manipulation_check: bool = False,
    max_tokens: int = 2048, confidence: float = 0.95, max_attempts_gen: int = 200,
) -> tuple[list[CapabilityCase], EvalLedgerEntry]:
    """主编排:生成 instances(matched-pair:同批跨 arm)→ 每 (solver,α,arm) 原子块求解 → 聚合。

    预算按 (solver,α,arm) 原子块:超限 stop + 标 incomplete + 列 dropped cells(避免半块数据)。
    返回 (cases, ledger_entry)。cases 按 (solver, instance, arm)。
    """
    spent = 0.0
    dropped: list[str] = []
    cases: list[CapabilityCase] = []

    # 预生成所有 instances(matched-pair 基础:所有 arm 共享同一批 instance)
    instances: list[SatInstance] = []
    for alpha in alphas:
        for i in range(n_instances):
            inst = gen_3sat(n_vars, alpha, base_seed + i, max_attempts=max_attempts_gen)
            instances.append(inst)

    for solver_entry in solver_entries:
        for v in variants:
            blocks = resolve_memory_blocks(v.memory_role, seeds, distractor_label)
            for inst in instances:
                # 原子块预算闸:块开始前已超限 → 整块 drop
                # (粒度=instance 级,因 matched-pair 需每 instance 全 arm;此处按 instance 计费停)
                if spent >= max_cost_usd:
                    dropped.append(f"{solver_entry['model']}/{inst.task_id}/{v.name}")
                    continue
                case = await _solve_one(
                    backend, solver_entry, inst, blocks,
                    manipulation_check=manipulation_check, max_tokens=max_tokens, effort=effort,
                )
                case.run_id = run_id
                case.arm = v.name
                case.memory_role = v.memory_role
                spent += case.cost_usd
                cases.append(case)
                store.record_case(case.to_case_record())

    summary = aggregate_capability(cases, variants, solver_entries, alphas,
                                   confidence=confidence, run_id=run_id)
    summary["budget"] = {"spent_usd": round(spent, 6), "max_usd": max_cost_usd,
                         "incomplete": bool(dropped), "dropped_cells": dropped}
    summary["manipulation_check"] = manipulation_check
    summary["distractor_label"] = distractor_label or (seeds["distractor_candidates"][0]["label"]
                                                        if seeds.get("distractor_candidates") else None)
    summary["n_vars"] = n_vars
    summary["alphas"] = list(alphas)
    summary["base_seed"] = base_seed

    entry = EvalLedgerEntry(
        run_id=run_id,
        date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=git_sha(),
        variants=[v.name for v in variants],
        judge_models=[se["model"] for se in solver_entries],   # 复用字段记 solver
        rubric_hash="", knowledge_hash="", reviewer_hash="",
        defaults_hash=defaults_hash({}),                       # capability 不读 dd
        n_tasks=len(instances),
        summary=summary,
    )
    store.record_run(entry)
    return cases, entry


# ── 分层指标 + 配对 bootstrap + effect size + 交互 ──────────────────────────────

def _rate(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def _cell_metrics(cell_cases: list[CapabilityCase]) -> dict:
    n = len(cell_cases)
    parse_ok = sum(c.parse_ok for c in cell_cases)
    valid = sum(c.valid_output for c in cell_cases)
    solved = sum(c.solved for c in cell_cases)
    failed = sum(c.call_failed for c in cell_cases)
    out_tok = [c.output_tokens for c in cell_cases if c.valid_output]
    rea_tok = [c.reasoning_tokens for c in cell_cases if c.valid_output]
    return {
        "n": n,
        "parse_rate": _rate(parse_ok, n),
        "valid_output_rate": _rate(valid, n),
        "solve_rate_given_valid": _rate(solved, valid),     # 主指标(隔离 format-fighting)
        "overall_solve_rate": _rate(solved, n),
        "call_fail_rate": _rate(failed, n),
        "output_tokens_given_valid": round(sum(out_tok) / len(out_tok), 1) if out_tok else None,
        "reasoning_tokens_given_valid": round(sum(rea_tok) / len(rea_tok), 1) if rea_tok else None,
        "solved_total": solved, "valid_total": valid, "failed_total": failed,
    }


def aggregate_capability(
    cases: list[CapabilityCase], variants: list[CapabilityVariant],
    solver_entries: list[dict], alphas: list[float], *, confidence: float, run_id: str,
) -> dict:
    """per-cell 指标 + 配对 bootstrap gap + effect size + 交互。

    matched-pair:rows = per-instance {arm: {solved, valid}}(同 instance 跨 arm);bootstrap 重采 instance。
    主 gap = distractor-vs-neutral(solve_rate_given_valid,长度控)。effect size = risk diff + odds ratio。
    """
    # per-cell(solver × α × arm)点指标
    per_cell: dict[str, dict] = {}
    for se in solver_entries:
        for alpha in alphas:
            for v in variants:
                cell = [c for c in cases if c.solver == se["model"]
                        and abs(c.alpha - alpha) < 1e-9 and c.arm == v.name]
                per_cell[f"{se['model']}|a{alpha}|{v.name}"] = _cell_metrics(cell)

    # 配对 bootstrap gap(主比较 distractor-vs-neutral + 辅)distractor-vs-relevant / relevant-vs-baseline
    gaps: dict[str, dict] = {}
    var_by_role = {v.memory_role: v.name for v in variants}
    pairs = [
        ("distractor_vs_neutral", "distractor", "neutral"),         # 主(长度控)
        ("distractor_vs_relevant", "distractor", "relevant"),
        ("relevant_vs_baseline", "relevant", "baseline"),
    ]
    for label, role_a, role_b in pairs:
        if role_a not in var_by_role or role_b not in var_by_role:
            continue
        arm_a, arm_b = var_by_role[role_a], var_by_role[role_b]
        gaps[label] = _paired_gap(cases, solver_entries, alphas, arm_a, arm_b,
                                  confidence=confidence, run_id=run_id, metric_key=label)

    # 交互:Δ_interaction = gap(hard α) − gap(easy α)(distractor-vs-neutral)
    interaction = _interaction(cases, solver_entries, var_by_role, alphas,
                               confidence=confidence, run_id=run_id)

    # claimed_note_usage(manip-check 时)
    claimed: dict[str, float | None] = {}
    if any(c.claimed_notes for c in cases):
        for v in variants:
            vc = [c for c in cases if c.arm == v.name and c.blocks_injected > 0]
            if vc:
                ratios = [len(c.claimed_notes) / c.blocks_injected for c in vc if c.blocks_injected]
                claimed[v.name] = round(sum(ratios) / len(ratios), 3) if ratios else None

    return {
        "claim_scope": "instruction interference(指令式策略 context),非 diffuse context pollution",
        "per_cell": per_cell,
        "gaps": gaps,
        "primary_gap": "distractor_vs_neutral(CI 整段<0 = 错向策略伤 solve-rate,长度已控)",
        "interaction": interaction,
        "claimed_note_usage": claimed or None,
    }


def _paired_rows(cases, solver_model, alpha, arm_a, arm_b) -> list[dict]:
    """同 instance × {arm_a, arm_b} 的配对行(solved/valid)。matched-pair 基础。"""
    by_inst: dict[str, dict] = {}
    for c in cases:
        if c.solver != solver_model or abs(c.alpha - alpha) >= 1e-9:
            continue
        by_inst.setdefault(c.task_id, {})[c.arm] = {"solved": c.solved, "valid": c.valid_output}
    rows = []
    for d in by_inst.values():
        if arm_a in d and arm_b in d:
            rows.append({arm_a: d[arm_a], arm_b: d[arm_b]})
    return rows


def _rate_from_rows(rows, arm):
    v = sum(r[arm]["valid"] for r in rows)
    s = sum(r[arm]["solved"] for r in rows)
    return (s / v) if v else None


def _paired_gap(cases, solver_entries, alphas, arm_a, arm_b, *, confidence, run_id, metric_key):
    """跨 α 的 solve_rate_given_valid(arm_a) − (arm_b):每 α 配对 bootstrap,再聚合展示。"""
    per_alpha: dict[str, dict] = {}
    for se in solver_entries:
        for alpha in alphas:
            rows = _paired_rows(cases, se["model"], alpha, arm_a, arm_b)
            if len(rows) < 2:
                per_alpha[f"{se['model']}|a{alpha}"] = {"point": None, "low": None, "high": None,
                                                          "n": len(rows)}
                continue

            def _rd(rs):
                ra, rb = _rate_from_rows(rs, arm_a), _rate_from_rows(rs, arm_b)
                return (ra - rb) if (ra is not None and rb is not None) else None

            def _or(rs):
                ra, rb = _rate_from_rows(rs, arm_a), _rate_from_rows(rs, arm_b)
                if ra in (None, 0, 1) or rb in (None, 0, 1):
                    return None            # odds ratio 边界 degenerate
                return (ra / (1 - ra)) / (rb / (1 - rb))

            rd = bootstrap_statistic(rows, _rd, confidence=confidence,
                                     seed=seed_for(run_id, f"{metric_key}-rd-a{alpha}"))
            orr = bootstrap_statistic(rows, _or, confidence=confidence,
                                      seed=seed_for(run_id, f"{metric_key}-or-a{alpha}"))
            per_alpha[f"{se['model']}|a{alpha}"] = {
                "risk_difference": {k: rd.get(k) for k in ("point", "low", "high", "n")},
                "odds_ratio": {k: orr.get(k) for k in ("point", "low", "high")},
            }
    return per_alpha


def _interaction(cases, solver_entries, var_by_role, alphas, *, confidence, run_id):
    """Δ_interaction = gap_distractor-vs-neutral(hard α) − gap(easy α)。

    easy/hard 取**本次 run 的 α 极值**(min/max),非硬编码理论相变(后者对校准后的网格不对)。
    不同 α = 不同 instance 池 → 无法跨 α 配对;给 per-α gap + 点 Δ,CI 注明「近似」(诚实限制)。
    """
    if "distractor" not in var_by_role or "neutral" not in var_by_role:
        return {"note": "缺 distractor/neutral 臂"}
    if not alphas:
        return {"note": "无 α"}
    easy_a, hard_a = min(alphas), max(alphas)
    arm_a, arm_b = var_by_role["distractor"], var_by_role["neutral"]
    points: dict[str, float | None] = {}
    for se in solver_entries:
        for tag, alpha in (("easy", easy_a), ("hard", hard_a)):
            rows = _paired_rows(cases, se["model"], alpha, arm_a, arm_b)
            ra, rb = _rate_from_rows(rows, arm_a), _rate_from_rows(rows, arm_b)
            points[f"{se['model']}|{tag}"] = (ra - rb) if (ra is not None and rb is not None) else None
    d = {}
    for se in solver_entries:
        e, h = points.get(f"{se['model']}|easy"), points.get(f"{se['model']}|hard")
        d[se["model"]] = {"gap_easy": e, "gap_hard": h,
                          "delta_interaction": (h - e) if (e is not None and h is not None) else None}
    d["note"] = "不同 α 不同 instance 池 → Δ_interaction CI 未单独 bootstrap(近似;以 per-α gap CI 为准)"
    d["easy_alpha"] = easy_a
    d["hard_alpha"] = hard_a
    return d
