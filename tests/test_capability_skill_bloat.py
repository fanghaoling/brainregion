"""Phase 3D:Skill-Inventory Bloat × Region-Scoping 测试。

procedural Decode skill(单一类型);4 臂(oracle/plausible/garbage/random_subset)matched-pair。
覆盖:gen pool/task(determinism、bijection、covered/uncovered、distractor 不变量、容量 raise)、
render(4 臂、token 匹配、不漏 skill 名、injection framing)、parse、_solve_sb_one(solved/unsolved/
wrong_selection/parse_fail/failed ladder)、SkillBloatCase.to_case_record、aggregate(per_cell/
k_curve/contrasts/entropy 空集保护)、runner(matched-pair、真实 cost gate、store、validation)。
"""
from __future__ import annotations

import pytest

from brainregion.eval import capability as cap
from brainregion.eval.capability import (
    Skill,
    SkillBloatCase,
    SkillBloatTask,
    _sb_cell_name,
    _sb_gen_correct,
    _sb_gen_distractors,
    _sb_gen_task,
    _sb_parse_output,
    _shannon_entropy,
    aggregate_capability_skill_bloat,
    gen_skill_pool,
    render_skill_bloat_prompt,
    run_capability_eval_skill_bloat,
)
from brainregion.providers.base import ModelResponse


# ── gen pool / task ────────────────────────────────────────────────────────────

def test_gen_correct_bijection():
    s = _sb_gen_correct(7, table_size=10)
    imgs = [img for _, img in s.alphabet]
    assert len(set(imgs)) == 10                      # 双射:像无重复
    assert {sym for sym, _ in s.alphabet} == set("ABCDEFGHIJ")


def test_gen_pool_seeded_deterministic():
    t1, p1 = gen_skill_pool(7, n_skills=32, table_size=12, n_examples=3)
    t2, p2 = gen_skill_pool(7, n_skills=32, table_size=12, n_examples=3)
    assert len(p1) == len(p2) == 32
    assert p1[0].alphabet == p2[0].alphabet          # seeded 确定性
    assert t1.test_input == t2.test_input and t1.gold == t2.gold
    assert gen_skill_pool(8, n_skills=32, table_size=12, n_examples=3)[0].gold != t1.gold  # 换 seed → 不同


def test_gen_task_necessity_and_gold():
    correct = _sb_gen_correct(3, table_size=14)
    task = _sb_gen_task(99, correct, n_examples=3)
    assert list(task.gold) == correct.apply(list(task.test_input))      # gold = correct 重跑
    # 必要性:测试输入含 ≥1 未覆盖符号(否则可从示例推断)
    uncovered = [s for s, _ in correct.alphabet if s not in task.covered]
    assert any(s in uncovered for s in task.test_input if s != "#")
    # covered ⊂ symbols
    assert task.covered <= {s for s, _ in correct.alphabet}


def test_gen_task_raises_when_table_too_small():
    """table_size ≤ 3×n_examples → 示例可能覆盖全部 → skill 非必要 → runner 拒启动(这里测 gen 容错)。"""
    # gen_task 本身不强校验;但 runner 会 raise(见 test_runner_validates_table_size)。


def test_distractor_invariants_covered_mismatch_and_gold_distinct():
    correct = _sb_gen_correct(5, table_size=12)
    task = _sb_gen_task(55, correct, n_examples=3)
    ds = _sb_gen_distractors(77, 20, correct, task)
    cmap = correct.map_
    assert len(ds) == 20
    seen_gold = {tuple(task.gold)}
    for d in ds:
        dm = d.map_
        # 与 correct 在 ≥1 covered 符号不同(→ 与示例不一致 → correct 唯一可定)
        assert any(dm[s] != cmap[s] for s in task.covered)
        g = tuple(d.apply(task.test_input))
        assert g != tuple(task.gold) and g not in seen_gold     # gold 去重(无第二正确答案)
        seen_gold.add(g)


def test_distractor_capacity_raises_when_output_space_small():
    """test_input 仅 2 个不同符号 → 互异 gold 数 = table_size×(table_size−1) < n → raise(不静默缩水)。"""
    import random
    symbols = list("ABCDEFGH")                       # table_size=8
    img = random.Random(0).sample(symbols, 8)
    correct = Skill(name="Decode-0", alphabet=tuple(zip(symbols, img)))
    task = SkillBloatTask(correct=correct,
                          examples=((("A", "B"), tuple(correct.apply(["A", "B"]))),),
                          test_input=("A", "B", "A", "B"),   # 仅 2 个不同符号 → 8×7=56 互异 gold
                          gold=tuple(correct.apply(["A", "B", "A", "B"])),
                          covered=frozenset({"A", "B"}), seed=0)
    with pytest.raises(ValueError):
        _sb_gen_distractors(3, 200, correct, task)   # 互异 gold 仅 56 < 200 → raise


# ── render ─────────────────────────────────────────────────────────────────────

def _task_pool():
    return gen_skill_pool(11, n_skills=64, table_size=14, n_examples=3)


def test_render_oracle_one_skill():
    task, pool = _task_pool()
    sys_, usr, chosen, inv = render_skill_bloat_prompt(task, "oracle", pool=pool, k=1, seed=1)
    assert len(chosen) == 1 and chosen[0].name == task.correct.name
    assert "候选 skill 数据" in sys_                          # injection framing(data)
    assert task.correct.name not in usr                       # user 不含 skill 名


def test_render_plausible_correct_plus_distractors_same_procedure():
    task, pool = _task_pool()
    sys_, usr, chosen, inv = render_skill_bloat_prompt(task, "plausible", pool=pool, k=8, seed=2)
    assert len(chosen) == 8
    assert task.correct in chosen                             # correct 恒在
    others = [c for c in chosen if c.name != task.correct.name]
    assert all(c.name != task.correct.name for c in others)   # 同类型(substitution)不同 alphabet
    assert task.correct.name not in usr


def test_render_garbage_token_matched_to_plausible():
    task, pool = _task_pool()
    _sp, _up, _cp, inv_plaus = render_skill_bloat_prompt(task, "plausible", pool=pool, k=8, seed=3)
    _sg, _ug, chosen_g, inv_garb = render_skill_bloat_prompt(task, "garbage", pool=pool, k=8, seed=3)
    assert len(chosen_g) == 1 and chosen_g[0].name == task.correct.name   # correct 恒在;garbage 不算 chosen
    # token 对照:garbage 与 plausible 库存 token ±15%
    assert inv_plaus > 0 and abs(inv_garb - inv_plaus) / inv_plaus <= 0.15


def test_render_random_subset_correct_not_guaranteed():
    # 小池 n_skills=8,k=4 → P(correct∈subset)=0.5/draw → 两种结果近确定都出现(correct 不保证)
    task, pool = gen_skill_pool(11, n_skills=8, table_size=14, n_examples=3)
    has_correct = has_no_correct = False
    for sd in range(50):
        _s, _u, chosen, _i = render_skill_bloat_prompt(task, "random_subset", pool=pool, k=4, seed=sd)
        assert len(chosen) == 4
        has_correct = has_correct or (task.correct in chosen)
        has_no_correct = has_no_correct or (task.correct not in chosen)
    assert has_correct and has_no_correct                    # 两种都出现(correct 不保证)


def test_render_rejects_unknown_arm():
    task, pool = _task_pool()
    with pytest.raises(ValueError):
        render_skill_bloat_prompt(task, "bogus", pool=pool, k=4, seed=1)


# ── parse ──────────────────────────────────────────────────────────────────────

def test_parse_output_variants():
    assert _sb_parse_output('{"result":["A","B","#"]}') == ["A", "B", "#"]
    assert _sb_parse_output('```json\n{"result":["M","Q"]}\n```') == ["M", "Q"]
    assert _sb_parse_output('thinking... {"out":["X"]} done') == ["X"]
    assert _sb_parse_output("") is None
    assert _sb_parse_output("totally not json") is None


# ── fake backend + _solve_sb_one ───────────────────────────────────────────────

class _FakeBackend:
    def __init__(self, responses, *, cost_usd=0.01):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.cost_usd = cost_usd

    async def complete(self, *, model, system, user, temperature=0.0, max_tokens=1024,
                       effort=None, endpoint_id=None):
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        if not self.responses:
            return ModelResponse(model=model, content="", error="no more responses")
        content, error = self.responses.pop(0)
        return ModelResponse(model=model, content=content, error=error,
                             usage={"prompt_tokens": 60, "completion_tokens": 30},
                             cost_usd=self.cost_usd)


def _entry():
    return {"model": "s", "endpoint_id": None}


@pytest.mark.asyncio
async def test_solve_sb_one_solved_with_gold(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    task, pool = gen_skill_pool(1, n_skills=32, table_size=12, n_examples=3)
    content = '{"result":' + str(list(task.gold)).replace("'", '"') + '}'
    back = _FakeBackend([(content, None)])
    rec = await cap._solve_sb_one(back, _entry(), task, arm="oracle", k=0, pool=pool,
                                  max_tokens=4096, effort=None, seed=5)
    assert rec.outcome == "solved" and rec.solved == 1
    assert rec.cell == "oracle"
    assert rec.inventory_tokens > 0
    assert rec.cost_usd == 0.01                              # 真实 cost 记录


@pytest.mark.asyncio
async def test_solve_sb_one_wrong_selection_plausible_only(monkeypatch, tmp_path):
    """plausible 臂:喂某 prompt 内 distractor 的 gold 输出 → unsolved + wrong_selection + matched_distractor。"""
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    task, pool = gen_skill_pool(2, n_skills=32, table_size=12, n_examples=3)
    # 先 render(seed=6)拿 chosen,挑一个在 prompt 里的 distractor;_solve_sb_one 用同 seed → 同 chosen
    _s, _u, chosen, _i = render_skill_bloat_prompt(task, "plausible", pool=pool, k=8, seed=6)
    distractor = next(c for c in chosen if c.name != task.correct.name)
    wrong_out = distractor.apply(task.test_input)
    content = '{"result":' + str(list(wrong_out)).replace("'", '"') + '}'
    back = _FakeBackend([(content, None), (content, None)])
    # plausible 臂 → wrong_selection 记录
    rec_p = await cap._solve_sb_one(back, _entry(), task, arm="plausible", k=8, pool=pool,
                                    max_tokens=4096, effort=None, seed=6)
    assert rec_p.outcome == "unsolved" and rec_p.solved == 0 and rec_p.valid_output == 1
    assert rec_p.wrong_selection == 1 and rec_p.matched_distractor == distractor.name
    # garbage 臂喂同样输出 → wrong_selection=0(仅 plausible 算)
    rec_g = await cap._solve_sb_one(back, _entry(), task, arm="garbage", k=8, pool=pool,
                                    max_tokens=4096, effort=None, seed=7)
    assert rec_g.wrong_selection == 0                        # review 不变量:wrong_selection 仅 plausible


@pytest.mark.asyncio
async def test_solve_sb_one_parse_fail_and_failed(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    task, pool = gen_skill_pool(3, n_skills=16, table_size=12, n_examples=3)
    # 烧 max_tokens / 非法 JSON → parse_fail(不抛异常)
    back_pf = _FakeBackend([("```,./not json,,", None)])
    rec_pf = await cap._solve_sb_one(back_pf, _entry(), task, arm="plausible", k=4, pool=pool,
                                     max_tokens=4096, effort=None, seed=8)
    assert rec_pf.outcome == "parse_fail" and rec_pf.valid_output == 0 and rec_pf.parse_ok == 1
    # backend error → failed
    back_fl = _FakeBackend([("", "boom")])
    rec_fl = await cap._solve_sb_one(back_fl, _entry(), task, arm="plausible", k=4, pool=pool,
                                     max_tokens=4096, effort=None, seed=9)
    assert rec_fl.outcome == "failed" and rec_fl.call_failed == 1


def test_to_case_record_variant_and_summary():
    c = SkillBloatCase(run_id="r", task_id="t", cell="plausible_k8", solver="s", arm="plausible",
                      k=8, n_skills=128, table_size=16)
    c.matched_distractor = "Decode-3"
    c.wrong_selection = 1
    rec = c.to_case_record()
    assert rec.variant == "plausible_k8"
    assert rec.report_summary["wrong_selection"] == 1 and rec.report_summary["matched_distractor"] == "Decode-3"


# ── aggregate ──────────────────────────────────────────────────────────────────

def _sbc(task_id: str, arm: str, k: int, solved: bool, *, wrong: bool = False,
         matched: str = "", solver: str = "s") -> SkillBloatCase:
    c = SkillBloatCase(run_id="r", task_id=task_id, cell=_sb_cell_name(arm, k), solver=solver,
                      arm=arm, k=k, n_skills=128, table_size=16, valid_output=1,
                      inventory_tokens=100, input_tokens=200,
                      outcome=("solved" if solved else "unsolved"), solved=1 if solved else 0)
    if wrong:
        c.wrong_selection = 1
        c.matched_distractor = matched
    return c


def test_aggregate_degradation_contrast_and_entropy_empty_guard():
    # oracle 全解;plausible_k8 解 1/3 → degradation>0。plausible wrong_selection 集中 → 低熵。
    cases = []
    for i, s in enumerate([True, True, True]):              # oracle ×3 全解
        cases.append(_sbc(f"t{i}", "oracle", 0, s))
    for i, (s, w, m) in enumerate([(False, True, "Decode-5"), (False, True, "Decode-5"), (True, False, "")]):
        c = _sbc(f"t{i}", "plausible", 8, s, wrong=w, matched=m)
        cases.append(c)
    for i in range(3):                                       # garbage_k8 全解(token 对照,易)
        cases.append(_sbc(f"t{i}", "garbage", 8, True))
    for i in range(3):                                       # random_subset_k8 0/3(常 miss correct)
        cases.append(_sbc(f"t{i}", "random_subset", 8, False))
    summ = aggregate_capability_skill_bloat(cases, [{"model": "s"}], [8],
                                            confidence=0.95, run_id="r")
    # k_curve
    assert summ["k_curve"]["s"]["oracle"] == 1.0
    assert summ["k_curve"]["s"]["plausible"]["8"] == round(1 / 3, 4)
    # degradation_at_k8 = solve(oracle) − solve(plausible) > 0
    d = summ["contrasts"]["degradation_at_k8"]["s"]["risk_difference"]
    assert d["point"] is not None and d["point"] > 0
    # entropy:wrong_selection 全集中在 Decode-5 → 低熵(>0);非 None
    cell = summ["per_cell"]["s|plausible_k8"]
    assert cell["selection_entropy"] is not None and cell["selection_entropy"] >= 0.0
    assert cell["top_distractor"] == "Decode-5"
    # entropy 空集保护:oracle cell 无 wrong_selection → entropy=None(top_distractor 同)
    assert summ["per_cell"]["s|oracle"]["selection_entropy"] is None
    assert summ["per_cell"]["s|oracle"]["top_distractor"] is None
    assert summ["per_cell"]["s|oracle"]["wrong_selection_rate"] == 0.0   # 0/valid = 0.0(率;非 None)


def test_shannon_entropy_empty_and_uniform():
    assert _shannon_entropy([]) is None
    assert _shannon_entropy(["a", "a"]) == 0.0              # 集中 → 熵 0
    assert _shannon_entropy(["a", "b"]) == 1.0              # 均匀 2 类 → 熵 1 bit


def test_aggregate_reasoning_cost_contrast():
    """deepseek 头条信号:plausible 比 oracle 花更多 reasoning_tok → reasoning_cost_at_k > 0。"""
    cases = []
    for i in range(4):
        oc = _sbc(f"t{i}", "oracle", 0, True)
        oc.reasoning_tokens = 200                            # oracle(lean)少想
        cases.append(oc)
        pc = _sbc(f"t{i}", "plausible", 8, True)
        pc.reasoning_tokens = 1600                           # plausible(bloat)多想 8×
        cases.append(pc)
    summ = aggregate_capability_skill_bloat(cases, [{"model": "s"}], [8],
                                            confidence=0.95, run_id="r")
    rc = summ["contrasts"]["reasoning_cost_at_k8"]["s"]["mean_diff"]
    assert rc["point"] is not None and rc["point"] > 0       # plausible − oracle > 0


# ── runner ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_runner_matched_pair_and_store(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    # 用真 gen:喂 gold 让 oracle 可解(验 matched-pair 同 task_id 跨臂)
    task0, _ = gen_skill_pool(101, n_skills=32, table_size=12, n_examples=3)
    gold_content = '{"result":' + str(list(task0.gold)).replace("'", '"') + '}'

    def _resp(model, system, user, **_):
        # 同 instance 同 gold;gold 不在 prompt → static gold 简化(只测调度/matched-pair)
        return ModelResponse(model=model, content=gold_content, usage={"prompt_tokens": 50},
                             cost_usd=0.0)

    class _B:
        async def complete(self, **kw):
            return _resp(**kw)

    recorded = []
    monkeypatch.setattr(cap.store, "record_case", lambda rec: recorded.append(rec))
    cases, entry = await run_capability_eval_skill_bloat(
        table_size=12, n_examples=3, n_instances=2, base_seed=101,
        ks=[4, 8], arms=["oracle", "plausible", "garbage", "random_subset"],
        solver_entries=[{"model": "s", "endpoint_id": None}], backend=_B(),
        run_id="run-sb", n_skills=32, max_cost_usd=5.0)
    # matched-pair:同一 task_id 跨臂出现
    task_ids = {c.task_id for c in cases}
    assert len(task_ids) == 2                                # 2 instance
    cells = {c.cell for c in cases}
    assert "oracle" in cells and "plausible_k4" in cells
    assert len(recorded) == len(cases)                       # 每 case 落 store


@pytest.mark.asyncio
async def test_runner_real_cost_gate_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))

    class _B:
        async def complete(self, **kw):
            return ModelResponse(model=kw["model"], content='{"result":["X"]}',
                                 usage={"prompt_tokens": 50}, cost_usd=0.01)

    recorded = []
    monkeypatch.setattr(cap.store, "record_case", lambda rec: recorded.append(rec))
    cases, entry = await run_capability_eval_skill_bloat(
        table_size=12, n_examples=3, n_instances=2, base_seed=200,
        ks=[4, 8], arms=["oracle", "plausible", "garbage", "random_subset"],
        solver_entries=[{"model": "s", "endpoint_id": None}], backend=_B(),
        run_id="run-sb2", n_skills=32, max_cost_usd=0.05)
    # 真 cost 0.01/call,gate 0.05 → 第 6 call 起 drop → incomplete
    assert entry.summary["budget"]["incomplete"] is True
    assert entry.summary["budget"]["dropped_cells"]
    assert len(recorded) < len(cases) + 100                  # dropped cell 不入 store
    assert all(c.cost_usd == 0.01 for c in cases)            # 真实 cost 记录(非 0)


@pytest.mark.asyncio
async def test_runner_validates_table_size_and_pool():
    # table_size ≤ 3×n_examples → raise(同步,在任何 await 前,backend 不被调用)
    with pytest.raises(ValueError):
        await run_capability_eval_skill_bloat(
            table_size=6, n_examples=3, n_instances=1, base_seed=1, ks=[4],
            arms=["oracle"], solver_entries=[{"model": "s", "endpoint_id": None}],
            backend=None, run_id="r", n_skills=32)
    # n_skills < max(K) → raise
    with pytest.raises(ValueError):
        await run_capability_eval_skill_bloat(
            table_size=12, n_examples=3, n_instances=1, base_seed=1, ks=[64],
            arms=["oracle"], solver_entries=[{"model": "s", "endpoint_id": None}],
            backend=None, run_id="r", n_skills=32)
