"""能力基准探针:抓"指纹一致但能力下降"(量化/snapshot 降级/sampling 被改/隐藏 prompt 拖累)。

设计支柱:**参数化模板 + 本地算出期望答案 → 全确定性判分**(无 judge、判分零成本);
数值槽位随 seed 随机化(防中转缓存探针/针对性路由;模板难度可控,跨 run 按通过率可比)。
单项二元判分噪声大(n=30 时 ±1 项≈3.3pp),故基线对比按**类目通过率聚合**,
degraded 阈值取 20pp(远超正常波动),10-20pp 记 suspicious。

类目(默认 30 项):math 8 / instruction 8 / logic 6 / code_output 6 / niah 2。
与 fingerprint.py 同声明的诚实框架:偏差≠欺诈;结果是信号强度。
"""
from __future__ import annotations

import json
import math
import random
import re
import string

from .fingerprint import _first_scalar

DISCLAIMER = (
    "偏差≠欺诈:provider 侧负载/采样波动也会造成小幅波动;degraded=能力信号显著下降,"
    "结合 behavior/usage 指纹一起定位原因(换模型 vs 同模型变弱)。"
)

#: 类目判定阈值(百分点,按类目通过率聚合后的整体下降幅度)
DEGRADED_DROP_PP = 20.0
SUSPICIOUS_DROP_PP = 10.0


# ---------------------------------------------------------------------------
# 判分:check 是自包含 dict,kind 分发
# ---------------------------------------------------------------------------


def _norm_text(s: str) -> str:
    return (s or "").strip().strip("\"'`*#·。.,!?！？:;：； \t「」『』【】").strip()


def _maybe_unshell(s: str, kind: str) -> str:
    """JSON 壳提取(实测 GLM 对短约束问题爱答 {"answer":"..."}):能解析且含标量叶子
    则用叶子作为答案本体。json_field_number 需要原始 JSON,跳过。"""
    if kind == "json_field_number":
        return s
    if s[:1] in "{[":
        try:
            leaf = _first_scalar(json.loads(s))
            if leaf is not None:
                return leaf
        except Exception:  # noqa: BLE001
            pass
    return s


def _first_number(s: str) -> float | None:
    """取首行**最后一个**数字:答案惯例在句尾("47 × 83 = 3901" 应得 3901 而非题干的 47)。"""
    line = (s or "").splitlines()[0] if (s or "").strip() else ""
    matches = re.findall(r"-?\d+(?:\.\d+)?", line)
    return float(matches[-1]) if matches else None


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % p == 0:
            return False
    return True


def grade(answer: str, check: dict) -> bool:
    """严格判分:探针题目都带强格式约束,不遵循格式本身就是被测能力的一部分。"""
    s = (answer or "").strip()
    if s.startswith("```"):  # 剥 fence(与 fingerprint.normalize_answer 同因:模型爱包壳)
        s = "\n".join(ln for ln in s.splitlines() if not ln.lstrip().startswith("```")).strip() or s
    kind = check.get("kind")
    s = _maybe_unshell(s, kind)
    first_line = s.splitlines()[0].strip() if s else ""
    if kind == "number":
        v = _first_number(s)
        return v is not None and abs(v - float(check["value"])) < 1e-6
    if kind == "exact":
        return _norm_text(first_line) == _norm_text(str(check["value"]))
    if kind == "regex":
        return bool(re.fullmatch(check["pattern"], first_line))
    if kind == "prime3":
        m = re.fullmatch(r"【答案】\s*(\d{3})", first_line.strip())
        return bool(m) and _is_prime(int(m.group(1)))
    if kind == "word_count_eq":
        return len(first_line.split()) == int(check["n"])
    if kind == "sentence_count_eq":
        parts = [p for p in re.split(r"[。！？.!?]", s) if p.strip()]
        return len(parts) == int(check["n"])
    if kind == "line_count_eq":
        lines = [ln for ln in s.splitlines() if ln.strip()]
        return len(lines) == int(check["n"])
    if kind == "len_le":
        body = s.replace("\n", "")
        return len(body) <= int(check["n"]) and bool(re.search(check["contains"], body))
    if kind == "not_contains":
        body = s.replace("\n", "")
        return len(body) >= int(check.get("min_len", 5)) and check["value"] not in body
    if kind == "json_field_number":
        try:
            obj = json.loads(s if s[:1] in "{[" else first_line)
            v = obj.get(check["field"])
            return v is not None and abs(float(v) - float(check["value"])) < 1e-6
        except Exception:  # noqa: BLE001
            return False
    return False


# ---------------------------------------------------------------------------
# 模板:每项 build(rng) -> {id, category, prompt, check, max_tokens}
# 期望答案在 build 时本地算出(不调模型、无 judge)
# ---------------------------------------------------------------------------

_NAMES = ["甲", "乙", "丙", "丁", "戊", "己"]
_ODD_ONE_SETS = [
    (["苹果", "香蕉", "樱桃", "土豆"], "土豆"),
    (["狗", "猫", "金鱼", "老虎"], "金鱼"),
    (["玫瑰", "菊花", "松树", "百合"], "松树"),
    (["篮球", "足球", "象棋", "排球"], "象棋"),
]
_NIAH_WORDS = ["松果", "灯塔", "青柠", "雪松", "琥珀", "风铃", "麦浪", "珊瑚"]
_FILLER = [
    "仓库管理员在盘点第{shelf}排货架,记录本上的字迹有些潦草。",
    "列车准点驶过第{mile}公里处的信号灯,窗外是一望无际的麦田。",
    "图书馆第{room}阅览室的灯还亮着,有人把书放回了原位。",
    "天气预报提到明天风力{wind}级,渔民们决定不出海。",
    "车间的机器运转正常,质检员在本子上记下了编号{no}。",
]


def _math_items(rng: random.Random) -> list[dict]:
    a, b = rng.randint(11, 99), rng.randint(11, 99)
    c, d = rng.randint(101, 999), rng.randint(12, 89)
    dv, dq = rng.randint(3, 12), rng.randint(8, 30)
    x, m, n = rng.randint(2, 20), rng.randint(3, 30), rng.randint(5, 60)
    nums = [rng.randint(7, 99) for _ in range(5)]
    g1, g2 = rng.randint(12, 40) * 2, rng.randint(12, 40) * 3
    s0, step = rng.randint(2, 9), rng.randint(3, 11)
    return [
        {
            "id": "math_mul2",
            "category": "math",
            "prompt": f"计算 {a} × {b} = ?只输出数字本身,不要任何其他文字。",
            "check": {"kind": "number", "value": a * b},
            "max_tokens": 32,
        },
        {
            "id": "math_mul3",
            "category": "math",
            "prompt": f"计算 {c} × {d} = ?只输出数字本身,不要任何其他文字。",
            "check": {"kind": "number", "value": c * d},
            "max_tokens": 32,
        },
        {
            "id": "math_div",
            "category": "math",
            "prompt": f"计算 {dv * dq} ÷ {dv} = ?只输出数字本身,不要任何其他文字。",
            "check": {"kind": "number", "value": dq},
            "max_tokens": 32,
        },
        {
            "id": "math_linear",
            "category": "math",
            "prompt": f"解方程:{m}x + {n} = {m * x + n},x = ?只输出数字本身。",
            "check": {"kind": "number", "value": x},
            "max_tokens": 32,
        },
        {
            "id": "math_sum",
            "category": "math",
            "prompt": "求 " + ",".join(str(v) for v in nums) + " 的和。只输出数字本身。",
            "check": {"kind": "number", "value": sum(nums)},
            "max_tokens": 32,
        },
        {
            "id": "math_avg",
            "category": "math",
            "prompt": "求 " + ",".join(str(v) for v in nums) + " 的平均数。只输出数字本身。",
            "check": {"kind": "number", "value": sum(nums) / len(nums)},
            "max_tokens": 32,
        },
        {
            "id": "math_gcd",
            "category": "math",
            "prompt": f"{g1 * 6} 和 {g2 * 6} 的最大公约数是多少?只输出数字本身。",
            "check": {"kind": "number", "value": math.gcd(g1 * 6, g2 * 6)},
            "max_tokens": 32,
        },
        {
            "id": "math_seq",
            "category": "math",
            "prompt": f"等差数列 {s0}, {s0 + step}, {s0 + 2 * step}, {s0 + 3 * step}, ... 的下一项是多少?只输出数字。",
            "check": {"kind": "number", "value": s0 + 4 * step},
            "max_tokens": 32,
        },
    ]


def _instruction_items(rng: random.Random) -> list[dict]:
    wc = rng.randint(6, 10)
    sc = rng.randint(2, 4)
    ja, jb = rng.randint(17, 89), rng.randint(17, 89)
    return [
        {
            "id": "instr_marker",
            "category": "instruction",
            "prompt": "回答必须以'【答案】'开头,后跟一个 3 位质数(100 到 999 之间),不要任何其他内容。",
            "check": {"kind": "prime3"},
            "max_tokens": 48,
        },
        {
            "id": "instr_word_count",
            "category": "instruction",
            "prompt": f"Write exactly {wc} English words about the weather. Output only that one sentence, nothing else.",
            "check": {"kind": "word_count_eq", "n": wc},
            "max_tokens": 96,
        },
        {
            "id": "instr_sentence_count",
            "category": "instruction",
            "prompt": f"用恰好 {sc} 句话介绍长城。不要列表、标题、编号或表情。",
            "check": {"kind": "sentence_count_eq", "n": sc},
            "max_tokens": 256,
        },
        {
            "id": "instr_all_caps",
            "category": "instruction",
            "prompt": "Translate into English and write it entirely in capital letters: 'the cat sleeps quietly'. Output only the translation.",
            "check": {"kind": "regex", "pattern": r"[A-Z' ,.-]{10,}"},
            "max_tokens": 64,
        },
        {
            "id": "instr_json",
            "category": "instruction",
            "prompt": f"计算 {ja} + {jb},并只输出 JSON 对象 {{\"result\": <数字>}},不要任何其他文字。",
            "check": {"kind": "json_field_number", "field": "result", "value": ja + jb},
            "max_tokens": 64,
        },
        {
            "id": "instr_len_le",
            "category": "instruction",
            "prompt": "用不超过 20 个字符(含标点)回答:地球绕太阳一圈大约要多久?只输出这个短回答。",
            "check": {"kind": "len_le", "n": 20, "contains": r"365|一年|1年"},
            "max_tokens": 64,
        },
        {
            "id": "instr_line_count",
            "category": "instruction",
            "prompt": "列出恰好 5 个欧洲国家,每行一个,不要编号、标点或其他文字。",
            "check": {"kind": "line_count_eq", "n": 5},
            "max_tokens": 128,
        },
        {
            "id": "instr_not_contains",
            "category": "instruction",
            "prompt": "用一句话解释光合作用,但不许出现'太阳'这两个字,也不要列表。",
            "check": {"kind": "not_contains", "value": "太阳", "min_len": 10},
            "max_tokens": 128,
        },
    ]


def _logic_items(rng: random.Random) -> list[dict]:
    perm = rng.sample(_NAMES, 5)
    rank = rng.choice([0, 1, 4])
    rank_q = {0: "最高", 1: "第二高", 4: "最矮"}[rank]
    stmts = [f"{perm[i]} 比 {perm[i + 1]} 高" for i in range(4)]
    rng.shuffle(stmts)
    colors = ["红色", "绿色", "蓝色"]
    people = rng.sample(_NAMES, 3)
    assign = list(zip(people, rng.sample(colors, 3)))
    ask_c = rng.choice(colors)
    dow = rng.randint(1, 7)  # 1=星期一
    k = rng.randint(40, 200)
    target = (dow - 1 + k) % 7 + 1  # "再过 k 天":再过 1 天=明天
    m2, n2, z2 = rng.randint(3, 15), rng.randint(5, 30), rng.randint(5, 30)
    names4 = rng.sample(_NAMES, 4)
    odd_pool, odd_ans = rng.choice(_ODD_ONE_SETS)
    shuffled_odd = odd_pool[:]
    rng.shuffle(shuffled_odd)
    return [
        {
            "id": "logic_ordering",
            "category": "logic",
            "prompt": "已知:" + ";".join(stmts) + f"。{rank_q}的是谁?只输出名字。",
            "check": {"kind": "exact", "value": perm[rank]},
            "max_tokens": 32,
        },
        {
            "id": "logic_assignment",
            "category": "logic",
            "prompt": "已知:" + ";".join(f"{p} 喜欢 {c}" for p, c in assign) + f"。谁喜欢{ask_c}?只输出名字。",
            "check": {"kind": "exact", "value": next(p for p, c in assign if c == ask_c)},
            "max_tokens": 32,
        },
        {
            "id": "logic_dow",
            "category": "logic",
            "prompt": f"今天是星期{'一二三四五六日'[dow - 1]}。再过 {k} 天(再过 1 天指明天)是星期几?只输出'星期X'。",
            "check": {"kind": "exact", "value": "星期" + "一二三四五六日"[target - 1]},
            "max_tokens": 32,
        },
        {
            "id": "logic_chain",
            "category": "logic",
            "prompt": f"甲比乙大 {m2};乙比丙小 {n2};丙是 {z2}。甲是多少?只输出数字。",
            "check": {"kind": "number", "value": z2 + n2 + m2},
            "max_tokens": 32,
        },
        {
            "id": "logic_queue",
            "category": "logic",
            "prompt": f"四人排队从前往后依次是:{','.join(names4)}。排在最前面的是谁?只输出名字。",
            "check": {"kind": "exact", "value": names4[0]},
            "max_tokens": 32,
        },
        {
            "id": "logic_odd_one",
            "category": "logic",
            "prompt": ",".join(shuffled_odd) + " 中哪一个与其他不同类?只输出那一个词。",
            "check": {"kind": "exact", "value": odd_ans},
            "max_tokens": 32,
        },
    ]


def _code_items(rng: random.Random) -> list[dict]:
    s = "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
    da, db = rng.randint(10, 60), rng.randint(61, 99)
    nums = [rng.randint(1, 9) for _ in range(5)]
    loop_n = rng.randint(5, 12)
    fa, fb = rng.randint(3, 50), rng.randint(3, 50)
    sa = rng.randint(1, 5)
    sa_set = sorted(rng.sample(range(1, 10), sa))
    sb_set = sorted(rng.sample(range(1, 10), sa + rng.randint(0, 2)))
    inter = sorted(set(sa_set) & set(sb_set))

    def code_prompt(snippet: str) -> str:
        return "以下 Python 代码的输出是什么?只输出输出内容本身(不要代码围栏、不要解释):\n```python\n" + snippet + "\n```"

    return [
        {
            "id": "code_slice",
            "category": "code_output",
            "prompt": code_prompt(f's = "{s}"\nprint(s[::-1].upper())'),
            "check": {"kind": "exact", "value": s[::-1].upper()},
            "max_tokens": 48,
        },
        {
            "id": "code_dict",
            "category": "code_output",
            "prompt": code_prompt(f'd = {{"a": {da}, "b": {db}}}\nprint(d.get("c", {da + db}))'),
            "check": {"kind": "exact", "value": str(da + db)},
            "max_tokens": 48,
        },
        {
            "id": "code_sumcomp",
            "category": "code_output",
            "prompt": code_prompt(f"nums = {nums}\nprint(sum(x * 2 for x in nums))"),
            "check": {"kind": "number", "value": sum(nums) * 2},
            "max_tokens": 48,
        },
        {
            "id": "code_loop",
            "category": "code_output",
            "prompt": code_prompt(f"t = 0\nfor i in range({loop_n}):\n    t += i\nprint(t)"),
            "check": {"kind": "number", "value": loop_n * (loop_n - 1) // 2},
            "max_tokens": 48,
        },
        {
            "id": "code_fstring",
            "category": "code_output",
            "prompt": code_prompt(f'a = {fa}\nb = {fb}\nprint(f"{{a}}+{{b}}={{a + b}}")'),
            "check": {"kind": "exact", "value": f"{fa}+{fb}={fa + fb}"},
            "max_tokens": 48,
        },
        {
            "id": "code_set",
            "category": "code_output",
            "prompt": code_prompt(f"a = {set(sa_set)}\nb = {set(sb_set)}\nprint(len(a & b))"),
            "check": {"kind": "number", "value": len(inter)},
            "max_tokens": 48,
        },
    ]


def _niah_items(rng: random.Random, target_tokens: int = 2500) -> list[dict]:
    items = []
    for idx in range(2):
        word = rng.choice(_NIAH_WORDS)
        # 先生成整段 filler,再把密码词插到 20%-80% 随机位置(避免过靠前的 primacy 送分)
        paras: list[str] = []
        total_chars = 0
        i = 0
        while total_chars < target_tokens * 7 // 4:
            i += 1
            tpl = _FILLER[i % len(_FILLER)]
            paras.append(tpl.format(shelf=i, mile=i * 3, room=i % 9 + 1, wind=i % 8 + 1, no=i * 7))
            total_chars += len(paras[-1])
        marker_at = int(len(paras) * rng.uniform(0.2, 0.8))
        paras.insert(
            marker_at, f"重要备忘:本段的密码词是「{word}」。请记住它,文末会提问。"
        )
        items.append(
            {
                "id": f"niah_{idx + 1}",
                "category": "niah",
                "prompt": "\n".join(paras) + "\n\n上文给出的密码词是什么?只输出这个词本身(两个字),不要引号。",
                "check": {"kind": "exact", "value": word},
                "max_tokens": 32,
            }
        )
    return items


def build_items(seed: int | None = None, include_niah: bool = True) -> list[dict]:
    rng = random.Random(seed)
    items = _math_items(rng) + _instruction_items(rng) + _logic_items(rng) + _code_items(rng)
    if include_niah:
        items += _niah_items(rng)
    return items


# ---------------------------------------------------------------------------
# 运行与对比
# ---------------------------------------------------------------------------


async def _complete(backend, **kwargs):
    try:
        return await backend.complete(**kwargs)
    except TypeError:
        kwargs.pop("thinking", None)
        return await backend.complete(**kwargs)


async def run_capability_probe(
    backend, *, model: str, endpoint_id: str | None = None, seed: int | None = None
) -> dict:
    """跑全套能力基准(temperature=0,逐项严格判分),返回类目通过率 + 逐项明细。"""
    items = build_items(seed=seed)
    per_item: list[dict] = []
    cost = 0.0
    served_model = None
    n_err = 0
    n_rescues = 0
    for it in items:
        resp = await _complete(
            backend,
            model=model,
            system="你是能力基准的被测端点。严格遵守每道题的输出格式要求,不要解释。",
            user=it["prompt"],
            temperature=0.0,
            max_tokens=it["max_tokens"],
            endpoint_id=endpoint_id,
            thinking=False,
        )
        if not getattr(resp, "ok", False):
            n_err += 1
            per_item.append({"id": it["id"], "category": it["category"], "passed": False, "error": "call_failed"})
            continue
        cost += getattr(resp, "cost_usd", None) or 0.0
        served_model = getattr(resp, "served_model", None) or served_model
        content = getattr(resp, "content", "") or ""
        passed = grade(content, it["check"])
        rescued = False
        if not passed and not content.strip():
            # 始终思考模型(实测 glm-5.3):思考烧光小上限 → content 空被误判零分。
            # 放大上限重测该项;rescued 标记保留,空答案事件本身也是端点信号。
            resp2 = await _complete(
                backend,
                model=model,
                system="你是能力基准的被测端点。严格遵守每道题的输出格式要求,不要解释。",
                user=it["prompt"],
                temperature=0.0,
                max_tokens=1024,
                endpoint_id=endpoint_id,
                thinking=False,
            )
            if getattr(resp2, "ok", False):
                cost += getattr(resp2, "cost_usd", None) or 0.0
                content2 = getattr(resp2, "content", "") or ""
                passed = grade(content2, it["check"])
                rescued = True
                n_rescues += 1
        per_item.append(
            {
                "id": it["id"],
                "category": it["category"],
                "passed": passed,
                "rescued_truncation": rescued,
            }
        )
    cats: dict[str, dict] = {}
    for it in per_item:
        c = cats.setdefault(it["category"], {"n": 0, "passed": 0})
        c["n"] += 1
        c["passed"] += 1 if it["passed"] else 0
    for c in cats.values():
        c["rate"] = round(c["passed"] / c["n"], 4) if c["n"] else 0.0
    total_n = len(per_item)
    total_p = sum(1 for it in per_item if it["passed"])
    return {
        "ok": True,
        "categories": cats,
        "overall_rate": round(total_p / total_n, 4) if total_n else 0.0,
        "items": per_item,
        "n_items": total_n,
        "n_errors": n_err,
        "n_truncation_rescues": n_rescues,
        "seed": seed,
        "cost_usd": round(cost, 6),
        "served_model": served_model,
    }


def compare_capability(cur: dict, base: dict) -> dict:
    """当前能力通过率 vs 基线:按类目聚合看下降幅度(整体 + 单类目)。"""
    if not cur.get("ok") or not base.get("ok"):
        return {
            "check": "capability",
            "verdict": "unknown",
            "error": cur.get("error") or base.get("error"),
            "note": DISCLAIMER,
        }
    cur_c: dict = cur.get("categories") or {}
    base_c: dict = base.get("categories") or {}
    per_cat = {}
    for cat in sorted(set(cur_c) | set(base_c)):
        b = (base_c.get(cat) or {}).get("rate")
        c = (cur_c.get(cat) or {}).get("rate")
        per_cat[cat] = {
            "baseline": b,
            "current": c,
            "delta_pp": round((b - c) * 100, 1) if b is not None and c is not None else None,
        }
    b_all, c_all = base.get("overall_rate"), cur.get("overall_rate")
    drop_pp = round((b_all - c_all) * 100, 1) if b_all is not None and c_all is not None else None
    if drop_pp is None:
        verdict = "unknown"
    elif drop_pp >= DEGRADED_DROP_PP:
        verdict = "degraded"
    elif drop_pp >= SUSPICIOUS_DROP_PP:
        verdict = "suspicious"
    else:
        verdict = "match"
    flags: list[str] = []
    dropped = [f"{cat}(-{v['delta_pp']}pp)" for cat, v in per_cat.items() if (v["delta_pp"] or 0) >= DEGRADED_DROP_PP]
    if dropped:
        flags.append("category_drop:" + ",".join(dropped))
    if (cur.get("n_errors") or 0) > 2:
        flags.append("high_call_failure_rate")
        verdict = "suspicious" if verdict == "match" else verdict
    if (cur.get("n_truncation_rescues") or 0) >= 5:
        # 端点频繁烧光小 token 上限(始终思考模型/注水):能力分是放大上限后的"真实能力",
        # 但这个端点对小请求的可用性确实差,单独亮旗
        flags.append("many_truncation_rescues")
    if (cur.get("n_items") or 0) < 20:
        flags.append("low_confidence_few_items")
    if b_all is not None and c_all is not None and (c_all - b_all) * 100 >= SUSPICIOUS_DROP_PP:
        flags.append("improved_vs_baseline")
    failed_now = [i["id"] for i in (cur.get("items") or []) if not i.get("passed")]
    return {
        "check": "capability",
        "verdict": verdict,
        "baseline_created_at": base.get("baseline_created_at"),
        "overall": {"baseline": b_all, "current": c_all, "drop_pp": drop_pp},
        "per_category": per_cat,
        "thresholds": {"suspicious_drop_pp": SUSPICIOUS_DROP_PP, "degraded_drop_pp": DEGRADED_DROP_PP},
        "flags": flags,
        "failed_items": failed_now[:15],
        "cost_usd": cur.get("cost_usd"),
        "note": DISCLAIMER,
    }
