"""Phase 3D-MultiFamily:skill-inventory bloat × region-scoping 多家族 benchmark 测试。

Family(逻辑层)+ Skill(数据层)+ registry;3 家族 decode/sort/filter。覆盖:registry、Skill 委托、
每家族 gen(determinism/necessity/distractor 不变量/容量)、render、parse、_solve_sb_one、
SkillBloatCase 字段、aggregate(per_family + overall macro/generation)、runner(families loop/cost gate)。
核心 gen/render/solve 测试用 pytest.mark.parametrize 跨家族。
"""
from __future__ import annotations

import pytest

from brainregion.eval import capability as cap
from brainregion.eval.capability import (
    SkillBloatCase,
    _classify_family_by_shapes,
    _sb_cell_name,
    _sb_gen_correct,
    _sb_gen_distractors,
    _sb_gen_task,
    _sb_parse_output,
    _shannon_entropy,
    aggregate_capability_mixed_router,
    aggregate_capability_skill_bloat,
    gen_mixed_pool,
    gen_skill_pool,
    get_family,
    render_skill_bloat_prompt,
    run_capability_eval_mixed_router,
    run_capability_eval_skill_bloat,
)
from brainregion.providers.base import ModelResponse

FAMILIES = ("decode", "sort", "filter")


# ── registry + Skill 委托 ──────────────────────────────────────────────────────

def test_family_registry():
    for f in FAMILIES:
        assert get_family(f).name == f
    with pytest.raises(ValueError):
        get_family("bogus")


@pytest.mark.parametrize("fam", FAMILIES)
def test_skill_delegates_to_family(fam):
    s = _sb_gen_correct(7, family=fam, table_size=10)
    seq = s.symbols()[:4]
    assert s.apply(seq) == get_family(fam).apply(s.parameter, seq)     # 委托
    assert s.doc_text() and s.name in s.doc_text()
    assert s.parameter_size == len(s.symbols()) == 10
    assert s.family == fam


# ── gen(跨家族 parametrize)─────────────────────────────────────────────────────

@pytest.mark.parametrize("fam", FAMILIES)
def test_gen_correct_seeded_deterministic(fam):
    a = _sb_gen_correct(5, family=fam, table_size=10)
    b = _sb_gen_correct(5, family=fam, table_size=10)
    assert a.parameter == b.parameter                               # seeded 确定性
    assert _sb_gen_correct(6, family=fam, table_size=10).parameter != a.parameter


@pytest.mark.parametrize("fam", FAMILIES)
def test_gen_task_necessity_and_gold(fam):
    correct = _sb_gen_correct(3, family=fam, table_size=12)
    task = _sb_gen_task(99, correct, n_examples=2)
    assert list(task.gold) == correct.apply(list(task.test_input))   # gold = correct 重跑
    assert task.family == fam
    uncovered = [s for s in correct.symbols() if s not in task.covered]
    assert any(s in uncovered for s in task.test_input)              # 必要性:测试含未覆盖
    assert task.covered <= set(correct.symbols())


@pytest.mark.parametrize("fam", FAMILIES)
def test_distractor_invariants(fam):
    correct = _sb_gen_correct(5, family=fam, table_size=12)
    task = _sb_gen_task(55, correct, n_examples=2)
    ds = _sb_gen_distractors(77, 20, correct, task)
    ex_inputs = [inp for inp, _ in task.examples]
    seen = {tuple(task.gold)}
    for d in ds:
        assert d.family == fam                                       # 同族
        # ① 与 correct 在 ≥1 示例输出上不同(→ 不一致 → correct 唯一可定)
        assert any(tuple(d.apply(ei)) != tuple(correct.apply(ei)) for ei in ex_inputs)
        g = tuple(d.apply(task.test_input))
        assert g != tuple(task.gold) and g not in seen               # ② gold 互异
        seen.add(g)


def test_distractor_capacity_raises_when_output_space_small():
    """filter + 2 不同符号测试 → gold 空间 2^2=4 远不足 50 → raise(不静默缩水)。"""
    correct = _sb_gen_correct(1, family="filter", table_size=8)
    # 手构造 2-symbol 测试(逼小 gold 空间)
    syms = correct.symbols()[:2]
    task = cap.SkillBloatTask(correct=correct, examples=(((syms[0],), tuple(correct.apply([syms[0]]))),),
                              test_input=tuple(syms), gold=tuple(correct.apply(syms)),
                              covered=frozenset({syms[0]}), seed=0, family="filter")
    with pytest.raises(ValueError):
        _sb_gen_distractors(3, 50, correct, task)


@pytest.mark.parametrize("fam", FAMILIES)
def test_gen_pool_seeded(fam):
    t1, p1 = gen_skill_pool(7, family=fam, n_skills=32, table_size=12, n_examples=2)
    t2, p2 = gen_skill_pool(7, family=fam, n_skills=32, table_size=12, n_examples=2)
    assert len(p1) == 32 and p1[0].family == fam
    assert t1.test_input == t2.test_input and t1.gold == t2.gold     # 确定性


# ── render ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fam", FAMILIES)
def test_render_arms(fam):
    task, pool = gen_skill_pool(11, family=fam, n_skills=64, table_size=12, n_examples=2)
    so, uo, co, _ = render_skill_bloat_prompt(task, "oracle", pool=pool, k=1, seed=1)
    assert len(co) == 1 and co[0].name == task.correct.name
    assert task.correct.name not in uo                               # user 不含 skill 名
    sp, up, cp, ip = render_skill_bloat_prompt(task, "plausible", pool=pool, k=8, seed=2)
    assert len(cp) == 8 and task.correct in cp
    sg, ug, cg, ig = render_skill_bloat_prompt(task, "garbage", pool=pool, k=8, seed=3)
    assert abs(ip - ig) / ip <= 0.15                                 # garbage ≈ plausible token
    sr, ur, cr, _ = render_skill_bloat_prompt(task, "random_subset", pool=pool, k=8, seed=4)
    assert len(cr) == 8


def test_render_rejects_unknown_arm():
    task, pool = gen_skill_pool(1, family="decode", n_skills=16, table_size=12, n_examples=2)
    with pytest.raises(ValueError):
        render_skill_bloat_prompt(task, "bogus", pool=pool, k=4, seed=1)


# ── parse ──────────────────────────────────────────────────────────────────────

def test_parse_output_variants():
    assert _sb_parse_output('{"result":["A","B"]}') == ["A", "B"]
    assert _sb_parse_output('```json\n{"result":["M"]}\n```') == ["M"]
    assert _sb_parse_output('x {"out":["Y"]} y') == ["Y"]
    assert _sb_parse_output("") is None
    assert _sb_parse_output("not json") is None


# ── fake backend + _solve_sb_one ───────────────────────────────────────────────

class _FakeBackend:
    def __init__(self, responses, *, cost_usd=0.01):
        self.responses = list(responses)
        self.cost_usd = cost_usd

    async def complete(self, *, model, system, user, temperature=0.0, max_tokens=1024,
                       effort=None, endpoint_id=None):
        if not self.responses:
            return ModelResponse(model=model, content="", error="no more responses")
        content, error = self.responses.pop(0)
        return ModelResponse(model=model, content=content, error=error,
                             usage={"prompt_tokens": 60, "completion_tokens": 30},
                             cost_usd=self.cost_usd)


def _entry():
    return {"model": "s", "endpoint_id": None}


@pytest.mark.asyncio
async def test_solve_sb_one_solved_and_wrong_selection(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    task, pool = gen_skill_pool(2, family="decode", n_skills=32, table_size=12, n_examples=2)
    gold_content = '{"result":' + str(list(task.gold)).replace("'", '"') + '}'
    # oracle:喂 gold → solved
    rec_o = await cap._solve_sb_one(_FakeBackend([(gold_content, None)]), _entry(), task,
                                    arm="oracle", k=0, pool=pool, max_tokens=4096, effort=None, seed=5)
    assert rec_o.outcome == "solved" and rec_o.family == "decode" and rec_o.cost_usd == 0.01
    # plausible:先 render 拿 chosen,挑一个 distractor 喂其输出 → wrong_selection;同 seed 同 chosen
    _, _, chosen, _ = render_skill_bloat_prompt(task, "plausible", pool=pool, k=8, seed=6)
    distractor = next(c for c in chosen if c.name != task.correct.name)
    wrong = '{"result":' + str(list(distractor.apply(task.test_input))).replace("'", '"') + '}'
    rec_p = await cap._solve_sb_one(_FakeBackend([(wrong, None), (wrong, None)]), _entry(), task,
                                    arm="plausible", k=8, pool=pool, max_tokens=4096, effort=None, seed=6)
    assert rec_p.wrong_selection == 1 and rec_p.matched_distractor == distractor.name
    # garbage 臂喂同样输出 → wrong_selection=0(仅 plausible)
    rec_g = await cap._solve_sb_one(_FakeBackend([(wrong, None)]), _entry(), task,
                                    arm="garbage", k=8, pool=pool, max_tokens=4096, effort=None, seed=7)
    assert rec_g.wrong_selection == 0


@pytest.mark.asyncio
async def test_solve_sb_one_parse_fail_and_failed(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    task, pool = gen_skill_pool(3, family="sort", n_skills=16, table_size=12, n_examples=2)
    rec_pf = await cap._solve_sb_one(_FakeBackend([("```,./not json", None)]), _entry(), task,
                                     arm="plausible", k=4, pool=pool, max_tokens=4096, effort=None, seed=8)
    assert rec_pf.outcome == "parse_fail" and rec_pf.parse_ok == 1
    rec_fl = await cap._solve_sb_one(_FakeBackend([("", "boom")]), _entry(), task,
                                     arm="plausible", k=4, pool=pool, max_tokens=4096, effort=None, seed=9)
    assert rec_fl.outcome == "failed" and rec_fl.call_failed == 1


def test_to_case_record_family_and_difficulty():
    c = SkillBloatCase(run_id="r", task_id="t", cell="plausible_k8", solver="s", arm="plausible",
                      k=8, n_skills=128, table_size=12, family="sort", difficulty=12.0)
    rec = c.to_case_record()
    assert rec.variant == "plausible_k8"
    assert rec.report_summary["family"] == "sort" and rec.report_summary["difficulty"] == 12.0


# ── aggregate(per_family + overall)─────────────────────────────────────────────

def _sbc(task_id, fam, arm, k, solved, *, wrong=False, matched="", solver="s", rea=0):
    c = SkillBloatCase(run_id="r", task_id=task_id, cell=_sb_cell_name(arm, k), solver=solver,
                      arm=arm, k=k, n_skills=128, table_size=12, family=fam, difficulty=12.0,
                      valid_output=1, inventory_tokens=100, input_tokens=200, reasoning_tokens=rea,
                      outcome=("solved" if solved else "unsolved"), solved=1 if solved else 0)
    if wrong:
        c.wrong_selection = 1
        c.matched_distractor = matched
    return c


def test_aggregate_per_family_and_overall_generalization():
    """两家族 decode/sort,都 oracle 全解 / plausible 解 1/3 → per_family degradation>0 +
    overall generalizes(两族 CI 都排 0)。"""
    cases = []
    for fam in ("decode", "sort"):
        for i in range(3):
            cases.append(_sbc(f"{fam}-t{i}", fam, "oracle", 0, True))
            cases.append(_sbc(f"{fam}-t{i}", fam, "plausible", 8, False))    # 全败 → diff=1.0 恒定 → CI 紧排 0
            cases.append(_sbc(f"{fam}-t{i}", fam, "garbage", 8, True))
            cases.append(_sbc(f"{fam}-t{i}", fam, "random_subset", 8, False))
    summ = aggregate_capability_skill_bloat(cases, [{"model": "s"}], [8],
                                            confidence=0.95, run_id="r", families=["decode", "sort"])
    # per_family
    assert set(summ["per_family"]) == {"decode", "sort"}
    d = summ["per_family"]["decode"]["contrasts"]["degradation_at_k8"]["s"]["risk_difference"]
    assert d["point"] is not None and d["point"] > 0
    # overall macro + generalization
    assert summ["overall"]["macro_mean"]["degradation_at_k8"]["s"] > 0
    g = summ["overall"]["generalization"]["degradation_at_k8"]["s"]
    assert g["n_families"] == 2 and g["generalizes"] is True           # 两族都 CI 排 0


def test_shannon_entropy_empty_and_uniform():
    assert _shannon_entropy([]) is None
    assert _shannon_entropy(["a", "a"]) == 0.0
    assert _shannon_entropy(["a", "b"]) == 1.0


# ── runner(families loop + cost gate)────────────────────────────────────────────

@pytest.mark.asyncio
async def test_runner_families_matched_pair_and_store(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    task0, _ = gen_skill_pool(101, family="decode", n_skills=32, table_size=12, n_examples=2)
    gold = '{"result":' + str(list(task0.gold)).replace("'", '"') + '}'

    class _B:
        async def complete(self, **kw):
            return ModelResponse(model=kw["model"], content=gold, usage={"prompt_tokens": 50}, cost_usd=0.0)

    recorded = []
    monkeypatch.setattr(cap.store, "record_case", lambda rec: recorded.append(rec))
    cases, entry = await run_capability_eval_skill_bloat(
        families=["decode", "sort"], table_size=12, n_examples=2, n_instances=2, base_seed=101,
        ks=[4, 8], arms=["oracle", "plausible", "garbage", "random_subset"],
        solver_entries=[{"model": "s", "endpoint_id": None}], backend=_B(),
        run_id="run-sb-mf", n_skills=32, max_cost_usd=5.0)
    fams_seen = {c.family for c in cases}
    assert fams_seen == {"decode", "sort"}                            # 两家族都跑
    assert len({c.task_id for c in cases}) == 4                       # 2 家族 × 2 instance
    assert len(recorded) == len(cases)


@pytest.mark.asyncio
async def test_runner_real_cost_gate_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))

    class _B:
        async def complete(self, **kw):
            return ModelResponse(model=kw["model"], content='{"result":["X"]}',
                                 usage={"prompt_tokens": 50}, cost_usd=0.01)

    cases, entry = await run_capability_eval_skill_bloat(
        families=["decode"], table_size=12, n_examples=2, n_instances=2, base_seed=200,
        ks=[4, 8], arms=["oracle", "plausible", "garbage", "random_subset"],
        solver_entries=[{"model": "s", "endpoint_id": None}], backend=_B(),
        run_id="run-sb-mf2", n_skills=32, max_cost_usd=0.05)
    assert entry.summary["budget"]["incomplete"] is True
    assert all(c.cost_usd == 0.01 for c in cases)


@pytest.mark.asyncio
async def test_runner_validates_table_size_and_pool():
    with pytest.raises(ValueError):
        await run_capability_eval_skill_bloat(
            families=["decode"], table_size=6, n_examples=2, n_instances=1, base_seed=1, ks=[4],
            arms=["oracle"], solver_entries=[{"model": "s", "endpoint_id": None}],
            backend=None, run_id="r", n_skills=32)
    with pytest.raises(ValueError):
        await run_capability_eval_skill_bloat(
            families=["decode"], table_size=12, n_examples=2, n_instances=1, base_seed=1, ks=[64],
            arms=["oracle"], solver_entries=[{"model": "s", "endpoint_id": None}],
            backend=None, run_id="r", n_skills=32)


# ── renderer(防 list.append 回归 + 多家族结构)──────────────────────────────────

def test_skill_bloat_markdown_renders_multifamily():
    from brainregion.cli import _capability_skill_bloat_markdown
    result = {
        "run_id": "r1", "mode": "skill_bloat", "n_instances": 2, "solvers": ["s"],
        "claim_scope": "test", "summary": {
            "families": ["decode", "sort"], "ks": [8],
            "arms": ["oracle", "plausible"], "table_size": 12, "n_examples": 2,
            "n_skills": 128, "base_seed": 700, "max_tokens": 4096,
            "per_family": {
                "decode": {"k_curve": {"s": {"oracle": 1.0, "plausible": {"8": 0.4}}},
                           "contrasts": {"degradation_at_k8": {"s": {"risk_difference": {"point": 0.6, "low": 0.2, "high": 0.9}, "n": 5}}},
                           "per_cell": {"s|plausible_k8": {"wrong_selection_rate": 0.1, "top_distractor": "decode-3", "selection_entropy": 1.2, "inventory_tokens_mean": 900}}},
                "sort": {"k_curve": {"s": {"oracle": 0.95, "plausible": {"8": 0.5}}},
                         "contrasts": {"degradation_at_k8": {"s": {"risk_difference": {"point": 0.45, "low": 0.1, "high": 0.8}, "n": 5}}},
                         "per_cell": {}},
            },
            "overall": {"macro_mean": {"degradation_at_k8": {"s": 0.525}},
                        "generalization": {"degradation_at_k8": {"s": {"n_families": 2, "n_ci_excludes_0": 2, "generalizes": True}}},
                        },
            "budget": {"incomplete": False}, "note": "test note",
        },
    }
    md = _capability_skill_bloat_markdown(result)
    assert "skill-bloat,多家族" in md and "Overall" in md
    assert "degradation_at_k8" in md and "✅" in md              # generalizes 标记
    assert "family=decode" in md and "family=sort" in md


# ── Phase 3E:mixed-pool + 跨区域 router ───────────────────────────────────────

def _mixed_pool(fam="decode", families=("decode", "filter"), n_within=2, n_cross=2):
    return gen_mixed_pool(11, correct_family=fam, families=list(families), n_within=n_within,
                          n_cross_per_family=n_cross, table_size=12, n_examples=2)


def test_gen_mixed_pool_structure():
    task, fp = _mixed_pool("decode", ("decode", "filter"), n_within=3, n_cross=4)
    assert task.family == "decode"
    assert fp["decode"][0].name == task.correct.name          # correct 在 correct 家族首位
    assert len(fp["decode"]) == 1 + 3                          # correct + n_within
    assert len(fp["filter"]) == 4                              # 每别家族 n_cross
    for s in fp["filter"]:
        assert s.family == "filter"                           # cross 来自别族
    assert sum(len(v) for v in fp.values()) == 1 + 3 + 4


def test_gen_mixed_pool_rejects_bad_inputs():
    with pytest.raises(ValueError):
        gen_mixed_pool(1, correct_family="decode", families=["decode"], n_within=2, n_cross_per_family=2,
                       table_size=12, n_examples=2)           # <2 家族
    with pytest.raises(ValueError):
        gen_mixed_pool(1, correct_family="bogus", families=["decode", "filter"], n_within=2,
                       n_cross_per_family=2, table_size=12, n_examples=2)  # correct_family 不在 families


@pytest.mark.parametrize("fam", ["decode", "filter", "sort"])
def test_classify_family_by_shapes(fam):
    """形状分类器对真实生成 task 100% 路由到正确家族(sanity:操作类型可从 I/O 形状辨 → 路由任务良态)。"""
    correct = _sb_gen_correct(5, family=fam, table_size=12)
    task = _sb_gen_task(55, correct, n_examples=2)
    assert _classify_family_by_shapes(task) == fam


@pytest.mark.asyncio
async def test_route_family_llm_parses_and_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    task, _ = _mixed_pool()

    class _B:
        async def complete(self, **kw):
            return ModelResponse(model=kw["model"], content="decode", usage={"prompt_tokens": 40},
                                 cost_usd=0.001)
    routed, r_in, r_cost, r_fail, raw = await cap.route_family_llm(_B(), {"model": "r", "endpoint_id": None},
                                                                   task, ["decode", "filter"])
    assert routed == "decode" and r_fail is False and r_cost == 0.001 and r_in == 40

    class _BFail:                                              # 路由调用抛错 → failed tuple
        async def complete(self, **kw):
            raise RuntimeError("boom")
    routed2, _, _, r_fail2, _ = await cap.route_family_llm(_BFail(), {"model": "r", "endpoint_id": None},
                                                           task, ["decode", "filter"])
    assert routed2 == "" and r_fail2 is True


@pytest.mark.asyncio
async def test_solve_mixed_one_arms(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    task, fp = _mixed_pool("decode", ("decode", "filter"), n_within=2, n_cross=2)
    gold = '{"result":' + str(list(task.gold)).replace("'", '"') + '}'
    entry = {"model": "s", "endpoint_id": None}
    # oracle:喂 gold → solved
    rec_o = await cap._solve_mixed_one(_FakeBackend([(gold, None)]), entry, task, fp, arm="oracle",
                                        correct_family="decode", max_tokens=4096, effort=None, seed=1)
    assert rec_o.outcome == "solved" and rec_o.pool_mode == "mixed"
    # mixed_all:chosen = 全 pool(2+3 = 5 跨 decode+filter);喂 gold → solved
    rec_a = await cap._solve_mixed_one(_FakeBackend([(gold, None)]), entry, task, fp, arm="mixed_all",
                                        correct_family="decode", max_tokens=4096, effort=None, seed=2)
    assert rec_a.outcome == "solved"
    assert rec_a.inventory_tokens > rec_o.inventory_tokens     # mixed_all doc > oracle doc
    # router_gold:chosen ⊆ 正确家族;喂 gold → solved;inventory 介于 oracle 与 mixed_all
    rec_g = await cap._solve_mixed_one(_FakeBackend([(gold, None)]), entry, task, fp, arm="router_gold",
                                        correct_family="decode", max_tokens=4096, effort=None, seed=3)
    assert rec_g.outcome == "solved"
    assert rec_o.inventory_tokens < rec_g.inventory_tokens < rec_a.inventory_tokens
    # router(正确路由):route_correct=1 → chosen=decode 族 → 喂 gold solved
    rec_rc = await cap._solve_mixed_one(_FakeBackend([(gold, None)]), entry, task, fp, arm="router",
                                         correct_family="decode", routed=("decode", 40, 0.001, False, ""),
                                         max_tokens=4096, effort=None, seed=4)
    assert rec_rc.route_correct == 1 and rec_rc.routed_family == "decode" and rec_rc.routing_cost_usd == 0.001
    # router(误路由到 filter):route_correct=0 → chosen=filter 族(correct 不在);喂非-gold → unsolved
    rec_rw = await cap._solve_mixed_one(_FakeBackend([('{"result":["Z"]}', None)]), entry, task, fp,
                                         arm="router", correct_family="decode",
                                         routed=("filter", 40, 0.001, False, ""),
                                         max_tokens=4096, effort=None, seed=5)
    assert rec_rw.route_correct == 0 and rec_rw.outcome == "unsolved"
    # router(路由彻底失败):不调主脑 → route_fail
    rec_rf = await cap._solve_mixed_one(_FakeBackend([]), entry, task, fp, arm="router",
                                         correct_family="decode", routed=("", 0, 0.0, True, ""),
                                         max_tokens=4096, effort=None, seed=6)
    assert rec_rf.outcome == "route_fail" and rec_rf.route_correct == 0


def _mxcase(task_id, correct_family, arm, solved, *, solver="s", route_correct=-1, routed="", rea=0):
    return SkillBloatCase(run_id="r", task_id=task_id, cell=arm, solver=solver, arm=arm, k=0,
                          n_skills=20, table_size=12, family=correct_family, difficulty=12.0,
                          pool_mode="mixed", correct_family=correct_family, routed_family=routed,
                          route_correct=route_correct, valid_output=1, inventory_tokens=100,
                          input_tokens=200, reasoning_tokens=rea,
                          outcome=("solved" if solved else "unsolved"), solved=1 if solved else 0)


def test_aggregate_mixed_decomposition():
    """oracle/router_gold 全解、mixed_all 全败、router 全解(正确路由)→ cross_region_value>0,
    within_region_bloat=0,route_accuracy=1.0。"""
    cases = []
    for fam in ("decode", "filter"):
        for i in range(3):
            tid = f"{fam}-t{i}"
            cases.append(_mxcase(tid, fam, "oracle", True))
            cases.append(_mxcase(tid, fam, "router_gold", True))
            cases.append(_mxcase(tid, fam, "mixed_all", False))
            cases.append(_mxcase(tid, fam, "router", True, route_correct=1, routed=fam))
    summ = aggregate_capability_mixed_router(cases, [{"model": "s"}], confidence=0.95, run_id="r",
                                             families=["decode", "filter"])
    cr = summ["per_family"]["decode"]["contrasts"]["cross_region_value"]["s"]["risk_difference"]
    assert cr["point"] is not None and cr["point"] > 0                    # router_gold 解 > mixed_all
    wr = summ["per_family"]["decode"]["contrasts"]["within_region_bloat"]["s"]["risk_difference"]
    assert wr["point"] == 0                                               # oracle == router_gold
    assert summ["overall"]["macro_mean"]["cross_region_value"]["s"] > 0
    assert summ["overall"]["route_accuracy"]["s"] == 1.0


@pytest.mark.asyncio
async def test_runner_mixed_structure_and_store(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    task0, _ = gen_mixed_pool(101, correct_family="decode", families=["decode", "filter"],
                              n_within=2, n_cross_per_family=2, table_size=12, n_examples=2)
    gold = '{"result":' + str(list(task0.gold)).replace("'", '"') + '}'

    class _B:
        async def complete(self, **kw):
            content = "decode" if "技能路由器" in kw.get("system", "") else gold   # 路由→家族;主脑→gold
            return ModelResponse(model=kw["model"], content=content, usage={"prompt_tokens": 50},
                                 cost_usd=0.0)

    recorded = []
    monkeypatch.setattr(cap.store, "record_case", lambda rec: recorded.append(rec))
    cases, entry = await run_capability_eval_mixed_router(
        families=["decode", "filter"], table_size=12, n_examples=2, n_instances=2, base_seed=101,
        n_within=2, n_cross_per_family=2, arms=["oracle", "mixed_all", "router_gold", "router"],
        solver_entries=[{"model": "s", "endpoint_id": None}],
        router_entry={"model": "r", "endpoint_id": None}, backend=_B(), run_id="run-mixed")
    assert {c.correct_family for c in cases} == {"decode", "filter"}      # 两 correct_family 都跑
    assert {c.arm for c in cases} == {"oracle", "mixed_all", "router_gold", "router"}
    assert len(recorded) == len(cases)                                    # 每 case 入库
    assert entry.summary["router_model"] == "r"


@pytest.mark.asyncio
async def test_runner_mixed_cost_gate_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))

    class _B:
        async def complete(self, **kw):
            content = "decode" if "技能路由器" in kw.get("system", "") else '{"result":["X"]}'
            return ModelResponse(model=kw["model"], content=content, usage={"prompt_tokens": 50},
                                 cost_usd=0.01)

    cases, entry = await run_capability_eval_mixed_router(
        families=["decode", "filter"], table_size=12, n_examples=2, n_instances=2, base_seed=200,
        n_within=2, n_cross_per_family=2, arms=["oracle", "mixed_all", "router_gold", "router"],
        solver_entries=[{"model": "s", "endpoint_id": None}],
        router_entry={"model": "r", "endpoint_id": None}, backend=_B(), run_id="run-mixed2",
        max_cost_usd=0.05)
    assert entry.summary["budget"]["incomplete"] is True


def test_mixed_router_markdown_renders():
    from brainregion.cli import _capability_mixed_router_markdown
    result = {
        "run_id": "r1", "mode": "skill_bloat_mixed", "n_instances": 2, "solvers": ["s"],
        "router_model": "r", "claim_scope": "test",
        "summary": {
            "families": ["decode", "filter"], "table_size": 12, "n_examples": 2, "n_within": 8,
            "n_cross_per_family": 8, "base_seed": 700,
            "per_family": {
                "decode": {"per_cell": {"s|oracle": {"solve_rate": 1.0, "reasoning_tokens_mean": 100,
                                                      "inventory_tokens_mean": 80, "cost_mean": 0.001},
                                         "s|router_gold": {"solve_rate": 0.9, "reasoning_tokens_mean": 300,
                                                           "inventory_tokens_mean": 200, "cost_mean": 0.002},
                                         "s|mixed_all": {"solve_rate": 0.4, "reasoning_tokens_mean": 800,
                                                         "inventory_tokens_mean": 600, "cost_mean": 0.004},
                                         "s|router": {"solve_rate": 0.85, "reasoning_tokens_mean": 320,
                                                      "inventory_tokens_mean": 200, "cost_mean": 0.002}},
                              "contrasts": {"cross_region_value": {"s": {"risk_difference": {"point": 0.5,
                                                                                              "low": 0.1, "high": 0.8},
                                                                         "n": 6}}},
                              "route_accuracy": {"s": 0.95}},
            },
            "overall": {"macro_mean": {"cross_region_value": {"s": 0.5},
                                       "within_region_bloat": {"s": 0.1},
                                       "routing_error_cost_solve": {"s": 0.05}},
                        "route_accuracy": {"s": 0.95}},
            "budget": {"incomplete": False}, "note": "test note",
        },
    }
    md = _capability_mixed_router_markdown(result)
    assert "跨区域 router" in md and "cross_region_value" not in md      # 用描述名非 key
    assert "跨区域 scoping" in md and "route_accuracy" in md
    assert "correct_family=decode" in md and "mixed_all" in md and "router_gold" in md
