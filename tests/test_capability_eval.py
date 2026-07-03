"""NP 能力基准(3-SAT 能力阶梯 + 客观验证)测试。

测纯函数(dpll/gen/verify/prompt/parse/arms/seeds)+ runner(block 注入/call 失败 excluded/预算 drop/
token 记录)+ 配对 bootstrap gap/effect size。声明范围 = instruction interference。
"""
from __future__ import annotations

import json

import pytest

from brainregion.eval import capability as cap
from brainregion.eval.capability import (
    CapabilityCase,
    build_capability_variants,
    dpll,
    gen_3sat,
    load_memory_seeds,
    parse_assignment,
    parse_claimed_notes,
    render_solve_prompt,
    resolve_memory_blocks,
    run_capability_eval,
    verify_sat,
)
from brainregion.providers.base import ModelResponse

SEEDS = "brainregion/eval/capability_fixtures/memory_seeds.yaml"


# ── DPLL + 生成 + 验证 ──────────────────────────────────────────────────────

def test_dpll_witness_satisfies_and_unsat_detected():
    inst = gen_3sat(8, 3.0, 42)
    witness = dict(inst.witness)
    assert verify_sat(witness, inst.clauses) is True          # 真 witness 必满足
    assert inst.conflicts >= 0 and inst.decisions >= 0         # 计数器合法
    assert inst.propagations >= 0 and inst.max_depth >= 0
    # UNSAT:x1 ∧ ¬x1(两单元子句)→ 必冲突
    sat, witness_u, *_ = dpll([(1,), (-1,)], 1)
    assert sat is False and witness_u is None


def test_gen_3sat_satisfiable_deterministic_and_alpha():
    a = gen_3sat(10, 4.0, 7)
    b = gen_3sat(10, 4.0, 7)
    assert a.clauses == b.clauses                              # 同 seed 确定性
    assert verify_sat(dict(a.witness), a.clauses) is True      # rejection-sample 必 SAT
    assert len(a.clauses) == round(4.0 * 10)                   # α × n_vars 子句数
    assert all(len(c) == 3 for c in a.clauses)                 # 3-SAT


def test_gen_3sat_raises_when_unsat_too_dense():
    # α=200(3 变量 600 子句)→ 必 UNSAT → max_attempts 超限 raise
    with pytest.raises(RuntimeError, match="未生成 SAT"):
        gen_3sat(3, 200.0, 1, max_attempts=30)


def test_gen_3sat_rejects_few_vars():
    with pytest.raises(ValueError):
        gen_3sat(2, 4.0, 1)


def test_verify_sat_partial_invalid_and_wrong():
    inst = gen_3sat(8, 3.0, 42)
    assert verify_sat(dict(inst.witness), inst.clauses) is True
    # partial(缺变量)→ 不满足(至少那些依赖缺变量的子句)
    partial = {v: b for v, b in inst.witness if v <= 4}
    assert verify_sat(partial, inst.clauses) is False
    # 翻转一个变量后大概率不满足(随便翻一个)
    flipped = dict(inst.witness)
    flipped[1] = not flipped[1]
    # 翻一个未必总破坏,但 w 是某满足赋值;此处只验 verify 跑通返回 bool
    assert isinstance(verify_sat(flipped, inst.clauses), bool)


# ── solve prompt + schema 解析 + manipulation-check ─────────────────────────

def test_parse_assignment_valid_fence_malformed():
    n = 6
    valid = '{"assign": {"x1": true, "x2": false, "x3": true, "x4": true, "x5": false, "x6": true}}'
    got = parse_assignment(valid, n)
    assert got is not None and got[1] is True and got[2] is False and len(got) == n
    # markdown fence 包裹仍可抽
    fenced = f"```json\n{valid}\n```"
    assert parse_assignment(fenced, n) is not None
    # 缺变量 → None
    short = '{"assign": {"x1": true}}'
    assert parse_assignment(short, n) is None
    # 余变量 → None
    extra = '{"assign": {"x1": true, "x2": false, "x3": true, "x4": true, "x5": false, "x6": true, "x7": true}}'
    assert parse_assignment(extra, n) is None
    # 非布尔(字符串 "true")→ None
    strbool = '{"assign": {"x1": "true", "x2": false, "x3": true, "x4": true, "x5": false, "x6": true}}'
    assert parse_assignment(strbool, n) is None
    # 完全 malformed → None
    assert parse_assignment("not json at all", n) is None
    assert parse_assignment("", n) is None


def test_render_solve_prompt_includes_labeled_memory():
    inst = gen_3sat(8, 3.0, 1)
    blocks = [{"label": "STRAT-R1", "text": "unit propagation"}, {"label": "STRAT-D2", "text": "全部取 True"}]
    system, user = render_solve_prompt(inst, blocks)
    assert "STRAT-R1" in user and "STRAT-D2" in user
    assert "assign" in system                                   # schema 提示
    # baseline 无 blocks → user 不含参考笔记
    s0, u0 = render_solve_prompt(inst, [])
    assert "参考笔记" not in u0


def test_parse_claimed_notes_labels():
    labels = ["STRAT-R1", "STRAT-D2"]
    assert parse_claimed_notes("我用了 STRAT-R1", labels) == ["STRAT-R1"]
    assert parse_claimed_notes("none", labels) == []
    assert parse_claimed_notes("STRAT-R1 和 STRAT-D2", labels) == ["STRAT-R1", "STRAT-D2"]


# ── arms + seeds ─────────────────────────────────────────────────────────────

def test_build_capability_variants_default_and_subset():
    vs = build_capability_variants()
    assert [v.memory_role for v in vs] == ["baseline", "relevant", "neutral", "distractor"]
    assert [v.name for v in vs] == ["baseline", "memory_relevant", "memory_neutral", "memory_distractor"]
    sub = build_capability_variants(["baseline", "distractor"])
    assert [v.memory_role for v in sub] == ["baseline", "distractor"]
    with pytest.raises(ValueError):
        build_capability_variants([])


def test_load_memory_seeds_packaged_and_failfast():
    seeds = load_memory_seeds(SEEDS)
    assert len(seeds["relevant"]) >= 1
    assert len(seeds["neutral"]) >= 1
    assert len(seeds["distractor_candidates"]) >= 10          # 候选池(pilot 筛)
    # 各 block 带 label/text
    for key in ("relevant", "neutral", "distractor_candidates"):
        for b in seeds[key]:
            assert b.get("label") and b.get("text")


def test_resolve_memory_blocks_per_arm():
    seeds = load_memory_seeds(SEEDS)
    n_rel = len(seeds["relevant"])
    n_neu = len(seeds["neutral"])
    assert len(resolve_memory_blocks("baseline", seeds, None)) == 0
    assert len(resolve_memory_blocks("relevant", seeds, None)) == n_rel
    assert len(resolve_memory_blocks("neutral", seeds, None)) == n_rel + n_neu
    d_blocks = resolve_memory_blocks("distractor", seeds, None)
    assert len(d_blocks) == n_rel + 1                           # relevant + 1 候选
    # distractor-label 选定特定候选
    label = seeds["distractor_candidates"][1]["label"]
    d_named = resolve_memory_blocks("distractor", seeds, label)
    assert d_named[-1]["label"] == label


# ── runner(fake backend)──────────────────────────────────────────────────────

class _FakeBackend:
    """记录调用;按 mode 返回 content/error。"""

    def __init__(self, content: str = "", *, error: str | None = None, cost: float = 0.002):
        self.content = content
        self.error = error
        self.cost = cost
        self.calls: list[dict] = []

    async def complete(self, *, model, system, user, temperature=0.0, max_tokens=1024,
                       effort=None, endpoint_id=None):
        self.calls.append({"model": model, "system": system, "user": user})
        return ModelResponse(
            model=model, content=self.content, error=self.error,
            usage={"prompt_tokens": 50, "completion_tokens": 20}, cost_usd=self.cost,
        )


def _witness_json(inst) -> str:
    return json.dumps({"assign": {f"x{v}": b for v, b in inst.witness}})


@pytest.mark.asyncio
async def test_runner_matched_pair_blocks_and_solve(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    inst = gen_3sat(8, 3.0, 42)
    monkeypatch.setattr(cap, "gen_3sat", lambda *a, **k: inst)   # 固定 instance(matched-pair)
    backend = _FakeBackend(_witness_json(inst))
    seeds = load_memory_seeds(SEEDS)
    variants = build_capability_variants()
    solver_entries = [{"model": "fake-solver", "endpoint_id": None, "label": "s"}]

    cases, entry = await run_capability_eval(
        n_vars=8, alphas=[3.0], n_instances=2, base_seed=0,
        variants=variants, solver_entries=solver_entries, backend=backend, seeds=seeds,
        run_id="run-cap-1", max_cost_usd=5.0,
    )
    n_rel, n_neu = len(seeds["relevant"]), len(seeds["neutral"])
    by_arm = {c.arm: c for c in cases if c.task_id == inst.task_id and c.solver == "fake-solver"}
    # 4 臂 × 2 instance = 8 cases;每臂 block 数对
    assert len(cases) == 8
    assert by_arm["baseline"].blocks_injected == 0
    assert by_arm["memory_relevant"].blocks_injected == n_rel
    assert by_arm["memory_neutral"].blocks_injected == n_rel + n_neu
    assert by_arm["memory_distractor"].blocks_injected == n_rel + 1
    # witness 全对 → solve_rate_given_valid = 1.0 / valid_output_rate = 1.0 / token 记录
    cell = entry.summary["per_cell"]["fake-solver|a3.0|baseline"]
    assert cell["solve_rate_given_valid"] == 1.0
    assert cell["valid_output_rate"] == 1.0
    assert cell["output_tokens_given_valid"] == 20
    assert entry.summary["budget"]["incomplete"] is False
    assert backend.calls and "STRAT-D" not in backend.calls[0]["user"]   # baseline 无 memory


@pytest.mark.asyncio
async def test_runner_real_gen_path(monkeypatch, tmp_path):
    """不 monkeypatch gen_3sat:验 runner 真 gen 调用签名对(防 kwarg 漂移)+ 真 instance 验证。"""
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    inst = gen_3sat(8, 3.0, 42)                                  # base_seed=42 → 同实例
    witness = json.dumps({"assign": {f"x{v}": b for v, b in inst.witness}})
    backend = _FakeBackend(witness)
    seeds = load_memory_seeds(SEEDS)
    cases, entry = await run_capability_eval(
        n_vars=8, alphas=[3.0], n_instances=1, base_seed=42,
        variants=build_capability_variants(["baseline"]),
        solver_entries=[{"model": "s", "endpoint_id": None, "label": "s"}],
        backend=backend, seeds=seeds, run_id="run-real-gen",
    )
    assert len(cases) == 1
    assert cases[0].task_id.startswith("sat-n8-a3.0-s")          # 真 instance task_id
    assert cases[0].solved == 1                                  # 真 witness → 真 instance solved


@pytest.mark.asyncio
async def test_runner_call_failure_excluded_from_denominator(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    inst = gen_3sat(8, 3.0, 42)
    monkeypatch.setattr(cap, "gen_3sat", lambda *a, **k: inst)
    backend = _FakeBackend("", error="boom")                     # call 失败
    seeds = load_memory_seeds(SEEDS)
    variants = build_capability_variants(["baseline"])
    solver_entries = [{"model": "fake-solver", "endpoint_id": None, "label": "s"}]

    cases, entry = await run_capability_eval(
        n_vars=8, alphas=[3.0], n_instances=2, base_seed=0,
        variants=variants, solver_entries=solver_entries, backend=backend, seeds=seeds,
        run_id="run-cap-fail", max_cost_usd=5.0,
    )
    cell = entry.summary["per_cell"]["fake-solver|a3.0|baseline"]
    assert cell["call_fail_rate"] == 1.0                          # 全 call 失败
    assert cell["valid_total"] == 0
    assert cell["solve_rate_given_valid"] is None                 # excluded → 分母 0 → None(不计 solve)


@pytest.mark.asyncio
async def test_runner_budget_drops_cells_atomic(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    inst = gen_3sat(8, 3.0, 42)
    monkeypatch.setattr(cap, "gen_3sat", lambda *a, **k: inst)
    backend = _FakeBackend(_witness_json(inst), cost=0.002)
    seeds = load_memory_seeds(SEEDS)
    variants = build_capability_variants()
    solver_entries = [{"model": "fake-solver", "endpoint_id": None, "label": "s"}]

    cases, entry = await run_capability_eval(
        n_vars=8, alphas=[3.0], n_instances=10, base_seed=0,
        variants=variants, solver_entries=solver_entries, backend=backend, seeds=seeds,
        run_id="run-cap-budget", max_cost_usd=0.005,            # 只够 ~2 次 → 余下 drop
    )
    assert entry.summary["budget"]["incomplete"] is True
    assert entry.summary["budget"]["dropped_cells"]              # 非空
    assert len(cases) < 40                                       # 没跑满 10×4


# ── 配对 bootstrap gap + effect size ─────────────────────────────────────────

def _case(task_id, arm, solver, alpha, solved, valid, output_tokens=20):
    return CapabilityCase(
        run_id="r", task_id=task_id, arm=arm, solver=solver, n_vars=8, alpha=alpha,
        memory_role=arm, blocks_injected=0, parse_ok=valid, valid_output=valid,
        solved=solved, output_tokens=output_tokens, cost_usd=0.0,
    )


def test_paired_gap_risk_difference_point():
    """distractor 解 2/4、neutral 解 4/4 → risk_difference(distractor−neutral) = 0.5−1.0 = −0.5。"""
    variants = build_capability_variants(["neutral", "distractor"])
    cases = []
    for tid in ("t1", "t2", "t3", "t4"):
        cases.append(_case(tid, "memory_neutral", "s", 4.26, solved=1, valid=1))
    # distractor 解对 t1,t3;错 t2,t4
    cases += [_case("t1", "memory_distractor", "s", 4.26, 1, 1),
              _case("t2", "memory_distractor", "s", 4.26, 0, 1),
              _case("t3", "memory_distractor", "s", 4.26, 1, 1),
              _case("t4", "memory_distractor", "s", 4.26, 0, 1)]
    solvers = [{"model": "s"}]
    summary = cap.aggregate_capability(cases, variants, solvers, [4.26], confidence=0.95, run_id="r")
    gap = summary["gaps"]["distractor_vs_neutral"]["s|a4.26"]["risk_difference"]
    assert gap["point"] == pytest.approx(-0.5, abs=1e-9)
    assert gap["n"] == 4                                         # matched-pair 4 instance
    assert summary["claim_scope"].startswith("instruction interference")
