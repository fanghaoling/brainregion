"""BrainRegion CLI — 不依赖 MCP/Claude Code 也能跑审查（开源友好 + 可进 CI/脚本）。

复用 server.review_document（纯 async 函数；import server 仅触发 FastMCP 实例化 + load_dotenv，
无启动副作用）。日志走 server 配的 stderr，stdout 保持干净（json/sarif 可直接管道消费）。

用法：
  brain-region plan path/to/plan.md --output markdown
  brain-region plan --text "# 方案..." --dimensions planner safety
  cat plan.md | brain-region plan -                     # stdin
  brain-region code src/a.py src/b.py --output sarif --output-file out.sarif
  brain-region doc rfc.md --type rfc
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

def _read_text_input(args) -> str:
    """plan/doc 输入优先级：--text > 文件路径 > stdin(-)。"""
    if args.text is not None:
        return args.text
    if args.input == "-":
        return sys.stdin.read()
    return Path(args.input).read_text(encoding="utf-8")


def _emit(result: dict, args) -> None:
    """json 输出整 dict；markdown/sarif 输出 result['rendered']。--output-file 写文件。"""
    if args.output_format == "json":
        text = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        text = result.get("rendered", "")
    if args.output_file:
        Path(args.output_file).write_text(text, encoding="utf-8")
    else:
        print(text)


def _add_review_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--adapter", default="auto", choices=["auto", "unity", "generic"])
    p.add_argument("--panel", nargs="*", default=None, help="模型列表，缺省用 config panel")
    p.add_argument("--dimensions", nargs="*", default=None)
    p.add_argument("--output", dest="output_format", default="json", choices=["json", "markdown", "sarif"])
    p.add_argument("--output-file", default=None, help="写文件（默认 stdout）")
    p.add_argument("--retrieve-top-k", type=int, default=5)
    p.add_argument("--extra-context", default="")
    p.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-cost-usd", type=float, default=None)
    p.add_argument("--timeout", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BrainRegion（脑区）AI 协作 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="审方案/计划（markdown）")
    p_plan.add_argument("input", nargs="?", default="-", help="文件路径或 -（stdin，默认）")
    p_plan.add_argument("--text", default=None, help="直接传文本（优先于 input 文件）")
    _add_review_args(p_plan)

    p_code = sub.add_parser("code", help="审代码（传文件路径，可多个）")
    p_code.add_argument("files", nargs="+", help="代码文件路径（可多个）")
    _add_review_args(p_code)

    p_doc = sub.add_parser("doc", help="审文档（指定 --type: markdown/adr/rfc/config）")
    p_doc.add_argument("input", nargs="?", default="-", help="文件路径或 -（stdin，默认）")
    p_doc.add_argument("--text", default=None)
    p_doc.add_argument("--type", dest="document_type", default="markdown",
                       choices=["markdown", "adr", "rfc", "config"])
    _add_review_args(p_doc)

    p_eval = sub.add_parser("eval", help="跑评测 harness（bootstrap 尺子：retrieve off/on/garbage + 盲评）")
    p_eval.add_argument("fixtures_dir", help="fixtures 目录（*.yaml 任务，每文件一个 EvalTask 或 list）")
    p_eval.add_argument("--adapter", default="auto", choices=["auto", "unity", "generic"])
    p_eval.add_argument("--panel", nargs="*", default=None, help="review panel 覆盖（建议单便宜模型控成本）")
    p_eval.add_argument("--dimensions", nargs="*", default=None)
    p_eval.add_argument(
        "--variants", default="retrieve_off:0,retrieve_on:5,retrieve_garbage:5g",
        help="变体 name:k[,..]，k 后缀 g 或第三段 g = garbage 负对照",
    )
    p_eval.add_argument("--judges", nargs="*", default=None, help="judge 模型列表（默认 normalizer_model）")
    p_eval.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    p_eval.add_argument("--max-cost-usd", type=float, default=None)
    p_eval.add_argument("--rubric", default=None, help="rubric 文件（默认 eval/rubrics/review_v1.md）")
    p_eval.add_argument("--export", default=None, help="导出本次 run 为 JSONL 路径")
    p_eval.add_argument("--output", dest="output_format", default="json", choices=["json", "markdown"])
    p_eval.add_argument("--output-file", default=None)

    p_cal = sub.add_parser("calibrate", help="judge 校准：用 gold 对测盲评 judge 能否稳定排序 good>bad")
    p_cal.add_argument("gold", help="gold YAML 文件或目录（每条：id/failure_mode/task/good/bad/note）")
    p_cal.add_argument("--judges", nargs="*", default=None, help="judge 模型列表（默认 normalizer_model）")
    p_cal.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    p_cal.add_argument("--threshold", type=float, default=0.7, help="agreement 达标阈值（默认 0.7）")
    p_cal.add_argument("--rubric", default=None, help="rubric 文件（默认 review_v1.md；--advice 用 advice_v1.md）")
    p_cal.add_argument("--advice", action="store_true",
                       help="校准 advice judge（outcome eval 用；落 CalibrationRecord，gate 前置）")
    p_cal.add_argument("--output", dest="output_format", default="json", choices=["json", "markdown"])
    p_cal.add_argument("--output-file", default=None)

    p_route = sub.add_parser("routing", help="量 wake_gate 路由精度（A=no_defense vs B=full，免费不调模型）")
    p_route.add_argument("fixtures_dir", help="fixtures 目录（*.yaml 任务，需带 gold_regions）")
    p_route.add_argument("--regions-dir", default=None, help="region yaml 目录（默认内置 REGIONS_DIR）")
    p_route.add_argument("--output", dest="output_format", default="json", choices=["json", "markdown"])
    p_route.add_argument("--output-file", default=None)

    p_out = sub.add_parser(
        "outcome",
        help="量 wake_gate→consult 建议质量（A=default vs B=routed，真调模型+盲评+gate）",
    )
    p_out.add_argument("fixtures_dir", help="fixtures 目录（*.yaml consult 任务，需带 gold_regions）")
    p_out.add_argument("--adapter", default="auto", choices=["auto", "unity", "generic"])
    p_out.add_argument("--panel", nargs="*", default=None, help="consult panel 覆盖（建议单便宜模型控成本）")
    p_out.add_argument("--judges", nargs="*", default=None, help="judge 模型列表（默认 normalizer_model）")
    p_out.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    p_out.add_argument("--max-cost-usd", type=float, default=None)
    p_out.add_argument("--rubric", default=None, help="rubric 文件（默认 eval/rubrics/advice_v1.md）")
    p_out.add_argument("--regions-dir", default=None, help="region yaml 目录（默认内置 REGIONS_DIR）")
    p_out.add_argument("--export", default=None, help="导出本次 run 为 JSONL 路径")
    p_out.add_argument("--additive", action="store_true",
                       help="加 routed_additive 变体（叠加式映射：base ∪ region 专题）做 3-way A/B")
    p_out.add_argument("--memory", action="store_true",
                       help="Phase2A.5：4 臂 memory 研究实验 OFF/RELEVANT/IRRELEVANT/STALE（主比较 RELEVANT vs IRRELEVANT，控 token 长度）")
    p_out.add_argument("--scoped", action="store_true",
                       help="scoped-eval：scoped（woken region 过滤）vs unscoped memory 注入（单变量=scope，验证 Phase A scoping）")
    p_out.add_argument("--output", dest="output_format", default="json", choices=["json", "markdown"])
    p_out.add_argument("--output-file", default=None)

    p_cap = sub.add_parser(
        "capability",
        help="NP 能力基准:程序化 3-SAT + 客观验证,测注入 context(instruction interference)对 hard-solve solve-rate 的影响",
    )
    p_cap.add_argument("--solvers", nargs="*", default=None,
                       help="solver 模型列表(默认 normalizer_model;建议传便宜模型如 deepseek-v4-flash 控成本)")
    p_cap.add_argument("--vars", type=int, default=20, help="3-SAT 变量数(>=3)")
    p_cap.add_argument("--alpha", default="3.5,4.0,4.2,4.26,4.4",
                       help="逗号分隔 clause/variable 比(均匀随机 3-SAT 相变 α≈4.26 最难)")
    p_cap.add_argument("--n", type=int, default=20, help="每 α 的 instance 数(smoke=3/pilot≈20/formal≥50)")
    p_cap.add_argument("--seed", type=int, default=0,
                       help="instance base seed(pilot seed 0 / formal seed 100+ 用 disjoint 防 selection bias)")
    p_cap.add_argument("--arms", default="baseline,relevant,neutral,distractor", help="逗号分隔臂子集")
    p_cap.add_argument("--distractor-label", default=None, help="用哪个 distractor 候选(默认首个;pilot 筛后填)")
    p_cap.add_argument("--seeds", default=None, help="memory_seeds.yaml 路径(默认内置 capability_fixtures)")
    p_cap.add_argument("--max-cost-usd", type=float, default=5.0)
    p_cap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    p_cap.add_argument("--manipulation-check", action="store_true",
                       help="pilot 用:事后问用了哪些 notes(claimed_note_usage,self-report,mention ≠ 因果 use)")
    p_cap.add_argument("--export", default=None, help="导出本次 run 为 JSONL 路径")
    p_cap.add_argument("--output", dest="output_format", default="json", choices=["json", "markdown"])
    p_cap.add_argument("--output-file", default=None)
    # Phase 3D:skill-inventory bloat × region-scoping(架构提升 A/B)
    p_cap.add_argument("--skill-bloat", action="store_true",
                       help="Phase 3D:多家族 skill 库 bloat × region-scoping A/B(§0 generality + region 地基)")
    p_cap.add_argument("--sb-families", default="decode,filter",
                       help="逗号分隔家族(registry:decode=map/filter=subset 跨模型通用;sort=reorder 仅推理"
                            "模型(deepseek)能执行,glm 非 oracle<0.9 floor)")
    p_cap.add_argument("--sb-inventory", default="2,8,32", help="逗号分隔库存大小 K(Stage 1 早拐点;formal 可扩 64)")
    p_cap.add_argument("--sb-arms", default=None,
                       help="逗号分隔臂子集(不传则按 --sb-pool 取全默认:single=oracle/plausible/garbage/random_subset;"
                            "mixed=oracle/mixed_all/router_gold/router)")
    p_cap.add_argument("--sb-table-size", type=int, default=12,
                       help="alphabet 符号数(校准值 12:oracle≈1 + 退化可见;>3×--sb-examples 保 skill 必要)")
    p_cap.add_argument("--sb-examples", type=int, default=2, help="一致性示例数(校准值 2;揭示部分 alphabet 供选择)")
    p_cap.add_argument("--sb-skills", type=int, default=128, help="pool 大小(≥ max K;random_subset 命中概率=K/pool)")
    p_cap.add_argument("--sb-max-tokens", type=int, default=4096, help="输出 max_tokens(全 cell 固定)")
    p_cap.add_argument("--sb-pool", default="single", choices=["single", "mixed"],
                       help="single=§0 单家族 bloat(默认);mixed=Phase 3E 跨区域 router(混合家族 pool + 现实路由)")
    p_cap.add_argument("--sb-n-within", type=int, default=8,
                       help="[mixed] 正确家族内同族异参 distractor 数(= §0 同族内 bloat)")
    p_cap.add_argument("--sb-n-cross", type=int, default=8,
                       help="[mixed] 每个别家族的 cross-family distractor 数(= 跨区域噪声)")
    p_cap.add_argument("--sb-router-model", default="modelbridge_openai/gpt-5.4-mini",
                       help="[mixed] 现实路由小模型(读示例+家族描述→家族;默认 gpt-5.4-mini;haiku/flash 备选)")

    p_sb = sub.add_parser(
        "sandbox",
        help="沙盒:闭环 agent harness(§15 控制环 keystone 的 code-regime 验证场;让测试过任务)",
    )
    p_sb_sub = p_sb.add_subparsers(dest="sandbox_command", required=True)
    p_sb_run = p_sb_sub.add_parser("run", help="单跑一个 fixture(看 agent 轨迹)")
    p_sb_run.add_argument("--task", default=None, help="fixture id(默认 off_by_one)")
    p_sb_run.add_argument("--arm", default="none", choices=["none", "brainregion"], help="顾问臂")
    p_sb_run.add_argument("--main-brain", default=None, help="主脑模型(deepseek-v4-flash / glm-5.2 等,非 Claude Code)")
    p_sb_run.add_argument("--brain-verify", action="store_true",
                          help="run 末尾对专家补丁跑 forced-trace + 对照客观 tests_green(§15.8,默认关=sidecar)")
    p_sb_run.add_argument("--brain-delegate", action="store_true",
                          help="run 末尾基于 brain_verify 信号跑 Delegate 步(action+下一步子目标,§15.1);"
                          "隐含 --brain-verify(delegate 消费 verify 信号);默认关=sidecar")
    p_sb_run.add_argument("--brain-loop", action="store_true",
                          help="§15.1 认知环外环:测试败+delegate 给出 next_subgoal 时用它重跑 expert(同 worktree "
                          "累改),loop 到 accept/give_up/budget/max_iterations。隐含 --brain-delegate --brain-verify;"
                          "默认关=sidecar")
    p_sb_run.add_argument(
        "--verification-region", action="store_true",
        help="真实补丁落盘后由 VerificationOptionRegion 自动运行 task pytest；默认关",
    )
    p_sb_run.add_argument(
        "--evidence-region", action="store_true",
        help="由无模型 EvidenceRegion 预读任务明确点名的文件，并发布到共享工作台；默认关",
    )
    p_sb_run.add_argument("--max-iterations", type=int, default=3,
                          help="外环最大迭代数(仅 --brain-loop 生效,默认 3)")
    p_sb_run.add_argument("--orthogonal-brain", default=None,
                          help="escalate 正交复查的第二模型(不同家族,如 main=deepseek-v4-flash → glm-5.2);"
                          "仅 --brain-loop 下生效:test 过但 forced-trace 怀疑(弱测试)时,盲审同 patch 作 tiebreaker"
                          "(正交 FAILED→redelegate / SOLVED→accepted);默认关")
    p_sb_run.add_argument("--max-steps", type=int, default=None)
    p_sb_run.add_argument("--max-cost-usd", type=float, default=None)
    p_sb_run.add_argument("--max-tokens", type=int, default=None)
    p_sb_run.add_argument("--thinking", default="off", choices=["off", "on"],
                          help="Provider 原生思考模式(off=关,默认;on=开;Claude 可配 --effort)")
    p_sb_run.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                          help="思考开时的强度(--thinking on 才生效;deepseek: low/medium→high,xhigh→max)")
    p_sb_run.add_argument(
        "--cognitive-scaffold",
        action="store_true",
        help="启用证据关联的外部认知状态；不记录或请求思维链",
    )
    p_sb_run.add_argument(
        "--cognitive-mode",
        default="runtime_checkpoint",
        choices=["runtime_checkpoint", "model_managed"],
        help="认知状态模式(runtime_checkpoint=运行时维护客观状态并按需检查点；默认)",
    )
    p_sb_run.add_argument(
        "--checkpoint-period",
        type=int,
        default=3,
        help="runtime_checkpoint 的最长周期步数(默认 3；错误和验证失败可提前触发)",
    )
    p_sb_run.add_argument(
        "--tool-result-lifecycle",
        default="full",
        choices=["full", "compact"],
        help="工具结果上下文策略(full=完整保留；compact=证据约束 receipt 卸载)",
    )
    p_sb_run.add_argument(
        "--tool-result-live-reads",
        type=int,
        default=3,
        help="compact 模式保持完整的最近 read_text 结果数(默认 3)",
    )
    p_sb_run.add_argument("--keep", action="store_true", help="失败时保留 run_dir 供检视")
    # --- worktree 模式(真实仓库任务)---
    p_sb_run.add_argument("--worktree", action="store_true",
                          help="真实仓库 worktree 模式:在 repo 的独立检出里跑 agent(需 --repo/--goal 或 --task-spec)")
    p_sb_run.add_argument("--task-spec", default=None, help="worktree 模式:WorktreeTask JSON 文件")
    p_sb_run.add_argument("--repo", default=None, help="worktree 模式:源仓库路径")
    p_sb_run.add_argument("--base", default=None, help="worktree 模式:base ref(默认 HEAD)")
    p_sb_run.add_argument("--goal", default=None, help="worktree 模式:任务目标(钉聚焦区,真仓库大)")
    p_sb_run.add_argument("--task-id", default=None, help="worktree 模式:任务 id(内联模式默认 worktree-<ts>)")
    p_sb_run.add_argument("--test-args", default=None, help='worktree 模式:pytest 参数(如 "tests/test_x.py -q")')
    p_sb_run.add_argument("--bootstrap", default=None,
                          help='worktree 模式:harness bootstrap 命令(如 "uv sync --extra dev";默认自动探测)')
    p_sb_run.add_argument("--no-bootstrap", action="store_true", help="worktree 模式:跳过 bootstrap")
    p_sb_run.add_argument("--python", default=None,
                          help="worktree 模式:评测用 python(默认探测 worktree .venv,再回退 sys.executable)")
    p_sb_eval = p_sb_sub.add_parser("eval", help="A/B(none vs brainregion)matched-pair + bootstrap CI gate")
    p_sb_eval.add_argument("--tasks", default=None, help="逗号分隔 fixture id(默认全部)")
    p_sb_eval.add_argument("--main-brain", default=None, help="主脑模型")
    p_sb_eval.add_argument("--max-steps", type=int, default=None)
    p_sb_eval.add_argument("--max-cost-usd", type=float, default=None)
    p_sb_eval.add_argument("--max-tokens", type=int, default=None)
    p_sb_eval.add_argument("--thinking", default="off", choices=["off", "on"],
                           help="DeepSeek 思考模式(off=关=便宜快,默认;on=开+--effort)")
    p_sb_eval.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    p_sb_eval.add_argument("--keep", action="store_true", help="失败时保留 run_dir")
    p_sb_eval.add_argument("--out", default=None, help="报告 JSON 输出目录(默认 .brain-region/sandbox/)")

    p_sb_delegation = p_sb_sub.add_parser(
        "delegation-eval",
        help="Matched eager and triggered expert evaluation on executable fixtures",
    )
    p_sb_delegation.add_argument("--tasks", required=True, help="Comma-separated fixture ids")
    p_sb_delegation.add_argument("--main-brain", default=None, help="Main executor model reference")
    p_sb_delegation.add_argument(
        "--expert",
        action="append",
        required=True,
        help="Independent REGION=MODEL or ASSIGNMENT:REGION=MODEL; repeat for multiple experts",
    )
    p_sb_delegation.add_argument(
        "--arms", default="main_only,single_expert,multi_expert", help="Comma-separated delegation arms"
    )
    p_sb_delegation.add_argument("--repeats", type=int, default=1)
    p_sb_delegation.add_argument("--max-steps", type=int, default=None)
    p_sb_delegation.add_argument("--max-cost-usd", type=float, default=None, help="Per-main-run cost limit")
    p_sb_delegation.add_argument("--max-tokens", type=int, default=None)
    p_sb_delegation.add_argument("--expert-max-context-tokens", type=int, default=6000)
    p_sb_delegation.add_argument("--expert-max-tokens", type=int, default=1200)
    p_sb_delegation.add_argument("--expert-temperature", type=float, default=0.1)
    p_sb_delegation.add_argument(
        "--trigger-after-steps",
        type=int,
        default=2,
        help="Trigger after this many completed steps without a workspace effect",
    )
    p_sb_delegation.add_argument(
        "--trigger-min-remaining-steps",
        type=int,
        default=2,
        help="Do not trigger unless this many main-model turns remain",
    )
    p_sb_delegation.add_argument("--bootstrap-samples", type=int, default=None)
    p_sb_delegation.add_argument("--thinking", default="off", choices=["off", "on"])
    p_sb_delegation.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    p_sb_delegation.add_argument("--keep", action="store_true", help="Keep failed arm directories")
    p_sb_delegation.add_argument("--out", default=None, help="Report directory")

    p_sb_cognitive = p_sb_sub.add_parser(
        "cognitive-eval",
        help="Matched 2x2: provider-native thinking x external cognitive state",
    )
    p_sb_cognitive.add_argument("--tasks", required=True, help="Comma-separated fixture ids")
    p_sb_cognitive.add_argument("--main-brain", default=None, help="Main executor model reference")
    p_sb_cognitive.add_argument(
        "--arms",
        default="plain,native_thinking,external_scaffold,combined",
        help="Comma-separated cognitive evaluation arms",
    )
    p_sb_cognitive.add_argument("--repeats", type=int, default=1)
    p_sb_cognitive.add_argument("--max-steps", type=int, default=None)
    p_sb_cognitive.add_argument("--max-cost-usd", type=float, default=None, help="Per-run cost limit")
    p_sb_cognitive.add_argument("--max-tokens", type=int, default=None)
    p_sb_cognitive.add_argument("--bootstrap-samples", type=int, default=None)
    p_sb_cognitive.add_argument(
        "--effort",
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Native-thinking effort; applied only to native_thinking and combined arms",
    )
    p_sb_cognitive.add_argument(
        "--scaffold-mode",
        default="runtime_checkpoint",
        choices=["runtime_checkpoint", "model_managed"],
        help="External-scaffold implementation used by scaffold arms",
    )
    p_sb_cognitive.add_argument(
        "--checkpoint-period",
        type=int,
        default=3,
        help="Maximum objective-event interval between runtime checkpoints",
    )
    p_sb_cognitive.add_argument(
        "--tool-result-lifecycle",
        default="full",
        choices=["full", "compact"],
        help="Tool-result transcript policy applied to every selected arm",
    )
    p_sb_cognitive.add_argument(
        "--tool-result-live-reads",
        type=int,
        default=3,
        help="Recent full read_text results retained in compact mode",
    )
    p_sb_cognitive.add_argument("--out", default=None, help="Report directory")

    p_sb_functional_regions = p_sb_sub.add_parser(
        "functional-region-eval",
        help="Matched main/passive/evidence/evidence+verification functional Region evaluation",
    )
    p_sb_functional_regions.add_argument("--tasks", required=True, help="Comma-separated fixture ids")
    p_sb_functional_regions.add_argument("--main-brain", default=None, help="Main executor model reference")
    p_sb_functional_regions.add_argument(
        "--arms",
        default=(
            "main_only,passive_context,evidence_region,"
            "evidence_verification_regions"
        ),
        help="Comma-separated functional Region evaluation arms",
    )
    p_sb_functional_regions.add_argument("--repeats", type=int, default=1)
    p_sb_functional_regions.add_argument("--max-steps", type=int, default=None)
    p_sb_functional_regions.add_argument(
        "--max-cost-usd", type=float, default=None, help="Per-run cost limit"
    )
    p_sb_functional_regions.add_argument("--max-tokens", type=int, default=None)
    p_sb_functional_regions.add_argument("--bootstrap-samples", type=int, default=None)
    p_sb_functional_regions.add_argument("--thinking", default="off", choices=["off", "on"])
    p_sb_functional_regions.add_argument(
        "--effort",
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
    )
    p_sb_functional_regions.add_argument(
        "--tool-result-lifecycle",
        default="full",
        choices=["full", "compact"],
        help="Common tool-result transcript policy for every arm",
    )
    p_sb_functional_regions.add_argument(
        "--tool-result-live-reads",
        type=int,
        default=3,
        help="Recent full read_text results retained in compact mode",
    )
    p_sb_functional_regions.add_argument("--out", default=None, help="Report directory")

    p_sb_tool_results = p_sb_sub.add_parser(
        "tool-result-eval",
        help="Matched full vs compact tool-result transcript lifecycle evaluation",
    )
    p_sb_tool_results.add_argument("--tasks", required=True, help="Comma-separated fixture ids")
    p_sb_tool_results.add_argument("--main-brain", default=None, help="Main executor model reference")
    p_sb_tool_results.add_argument(
        "--arms",
        default="full,compact",
        help="Comma-separated lifecycle arms; full is control and compact is treatment",
    )
    p_sb_tool_results.add_argument("--repeats", type=int, default=1)
    p_sb_tool_results.add_argument("--max-steps", type=int, default=None)
    p_sb_tool_results.add_argument(
        "--max-cost-usd", type=float, default=None, help="Per-run cost limit"
    )
    p_sb_tool_results.add_argument("--max-tokens", type=int, default=None)
    p_sb_tool_results.add_argument("--bootstrap-samples", type=int, default=None)
    p_sb_tool_results.add_argument("--thinking", default="off", choices=["off", "on"])
    p_sb_tool_results.add_argument(
        "--effort",
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
    )
    p_sb_tool_results.add_argument(
        "--cognitive-scaffold",
        action="store_true",
        help="Apply the same external cognitive scaffold to both lifecycle arms",
    )
    p_sb_tool_results.add_argument(
        "--scaffold-mode",
        default="runtime_checkpoint",
        choices=["runtime_checkpoint", "model_managed"],
    )
    p_sb_tool_results.add_argument("--checkpoint-period", type=int, default=3)
    p_sb_tool_results.add_argument(
        "--tool-result-live-reads",
        type=int,
        default=3,
        help="Recent full read_text results retained by the compact arm",
    )
    p_sb_tool_results.add_argument(
        "--shared-prefix-turns",
        type=int,
        default=2,
        help="Exact pre-treatment model responses replayed into the second arm (0..2)",
    )
    p_sb_tool_results.add_argument("--out", default=None, help="Report directory")

    p_sb_shadow = p_sb_sub.add_parser(
        "delegation-shadow",
        help="Replay candidate expert-activation gates from a delegation report without model calls",
    )
    p_sb_shadow.add_argument("--report", required=True, help="Existing delegation report JSON")
    p_sb_shadow.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Original main-run step budget; required for legacy reports that did not record it",
    )

    p_sb_vb = p_sb_sub.add_parser(
        "verify-brain",
        help="主脑 grounding-first 验证(§15.8):对 run.json 里的专家补丁跑 forced-trace,"
        "对照 run 里存的客观 tests_green → agree / 弱测试信号 / trace 漏检",
    )
    p_sb_vb.add_argument("--run", required=True, help="run.json 路径(sandbox run --worktree 的产物)")
    p_sb_vb.add_argument("--main-brain", default=None, help="主脑模型(默认 sandbox_main_brain 或 deepseek-v4-flash)")
    p_sb_vb.add_argument("--test-req", default=None, help="覆盖测试要求文本(默认取 task.goal)")

    p_sb_env = p_sb_sub.add_parser(
        "env",
        help="env-regime:主脑操作文本环境,observe/act 作 tool 复用 run_agent;实时调试窗 + replay 回放",
    )
    p_sb_env.add_argument(
        "--env", default="gridworld", choices=["gridworld", "urban-delivery"],
        help="环境:gridworld 或 urban-delivery",
    )
    p_sb_env.add_argument("--size", type=int, default=None, help="网格边长(gridworld 默认 5;配送默认 13)")
    p_sb_env.add_argument("--orders", type=int, default=3, help="配送订单数(urban-delivery,1..8;默认 3)")
    p_sb_env.add_argument("--vehicles", type=int, default=2, help="临时阻挡车辆数(urban-delivery,0..8;默认 2)")
    p_sb_env.add_argument("--goal-text", default=None, help="目标描述(默认:到达目标 G;fog 下不透露位置)")
    p_sb_env.add_argument("--arm", default="none", choices=["none", "brainregion"], help="顾问臂(Phase A 默认 none)")
    p_sb_env.add_argument("--main-brain", default=None, help="主脑模型(deepseek-v4-flash / glm-5.2 等)")
    p_sb_env.add_argument("--max-steps", type=int, default=None)
    p_sb_env.add_argument("--max-cost-usd", type=float, default=None)
    p_sb_env.add_argument("--max-tokens", type=int, default=None)
    p_sb_env.add_argument("--thinking", default="off", choices=["off", "on"],
                          help="DeepSeek 思考模式(off=关=便宜快非推理,默认;on=开+--effort)")
    p_sb_env.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    p_sb_env.add_argument("--debug", action="store_true",
                          help="开调试窗(后台 serve_debug_dashboard,SSE 实时看 env.step 事件)")
    # --- Phase B fog(部分可观)---
    p_sb_env.add_argument("--fog", action="store_true",
                          help="Phase B fog:部分可观(只看到角色周围 --visibility-radius 格,余 `?`)")
    p_sb_env.add_argument("--visibility-radius", type=int, default=None,
                          help="fog 可见半径(Chebyshev;默认 --fog 时 2;须 ≥0)")
    p_sb_env.add_argument("--goal-x", type=int, default=None, help="显式 goal 列(默认远角)")
    p_sb_env.add_argument("--goal-y", type=int, default=None, help="显式 goal 行(默认远角)")
    p_sb_env.add_argument("--random-goal", action="store_true",
                          help="seeded 随机 goal(fog 下优先藏在 start 可见域外,逼探索)")
    p_sb_env.add_argument("--seed", type=int, default=None, help="场景种子(随机 goal/迷宫/配送车辆;默认 0)")
    p_sb_env.add_argument("--wall-density", type=float, default=None,
                          help="随机墙密度(0..0.6,占可放格比例;启用需配 --wall-seed)")
    p_sb_env.add_argument("--wall-seed", type=int, default=None,
                          help="随机墙种子(启用随机墙;BFS 保证 start→goal 可达)")
    # --- Phase C 记忆脑区(strict 部分可观 + recall_map)---
    p_sb_env.add_argument("--memory", action="store_true",
                          help="Phase C 记忆脑区:严格部分可观(observe 只给当前视野)+ recall_map 拿累积探索图;"
                          "自动启用 fog(strict_obs=True)")
    # --- Phase D 记忆脑区(真 LLM,region-as-tool)---
    p_sb_env.add_argument("--memory-region", action="store_true",
                          help="Phase D.2 记忆脑区(有状态 LLM):region 自给 dead-reckon + 维护定性 rough_map"
                          "(不收 env 完美图,自己挣大致地图理解给主脑);隐含 --memory。A/B vs --memory(被动倒图)")
    p_sb_env.add_argument("--memory-log-len", type=int, default=32,
                          help="记忆脑区 movement_log 有界长度(默认 32;--memory-region 用)")
    # --- Phase D.3 策略脑区(多脑区协同,region-as-tool)---
    p_sb_env.add_argument("--strategy-region", action="store_true",
                          help="Phase D.3 策略脑区(LLM 规划器):plan 工具调策略脑区(读记忆脑区理解,提意图"
                          "去哪/子目标,不给动作);隐含 --memory-region。A/B:Memory-only vs Memory+Strategy")
    # --- Phase 4.2/4.3/4.4 env-regime 旋钮(单 episode 可视化用;复刻 env-eval 臂配置)---
    p_sb_env.add_argument("--visual-ephemeral", action="store_true",
                          help="Phase 4.2:剥历史视觉观察出 transcript(只留最新 <visual>);act 动作结果持久。"
                          "逼主脑调 recall_map 拿历史视觉(测脑区是否变必需)。配 --memory-region 用。")
    p_sb_env.add_argument("--registry", default="none", choices=["none", "cap", "full"],
                          help="Phase 4.3 脑区注册表块:none=无 | cap=仅能力(显著性)| full=能力+客观触发。"
                          "需 --memory-region/--strategy-region 才列脑区。")
    p_sb_env.add_argument("--memory-dummy", action="store_true",
                          help="Phase 4.4 matched-source dummy 记忆:同 LLM 调用,喂回固定 content-free "
                          "rough_map(content-null 控制臂)。需 --memory-region。A/B:real vs dummy 内容价值。")
    # --- Phase 4.5 迷宫地形(recursive backtracker)---
    p_sb_env.add_argument("--maze", action="store_true",
                          help="Phase 4.5 迷宫地形(recursive backtracker:走廊+死胡同,会迷路 → 记忆变必需)。"
                          "用 --seed 作 maze_seed。开放网格(wall_density=0)记忆不必需 → 换迷宫测真价值。"
                          "隐含 fog(经 --memory-region)。配 --maze-braid 调难度。")
    p_sb_env.add_argument("--maze-braid", type=float, default=0.2,
                          help="迷宫去死胡同比例(0=完美迷宫最难,0.2=地牢感略易,1.0=全去)。默认 0.2。")
    p_sb_env.add_argument("--ego-actions", action="store_true",
                          help="Phase 4.8:ego-relative action(agent 有朝向,action=forward/turn_left/turn_right)")
    p_sb_env.add_argument("--debug-port", type=int, default=8765, help="--debug 调试窗端口(默认 8765)")

    # --- Phase 4 formal A/B harness(env-regime)---
    p_sb_env_eval = p_sb_sub.add_parser(
        "env-eval",
        help="Phase 4 formal A/B:多 run × arms(控制臂 Echo)+ 过程指标 + config 级 bootstrap CI/gate",
    )
    p_sb_env_eval.add_argument("--main-brain", default=None, help="主脑模型(deepseek-v4-flash / glm-5.2 等)")
    p_sb_env_eval.add_argument("--sizes", default="8", help="网格边长 list 逗号分隔(默认 8;笛卡尔积 × seeds)")
    p_sb_env_eval.add_argument("--seeds", default="1,2", help="random_goal_seed list(默认 1,2;笛卡尔积 × sizes)")
    p_sb_env_eval.add_argument("--visibility-radius", type=int, default=None, help="fog 半径(默认 2)")
    p_sb_env_eval.add_argument("--wall-density", type=float, default=None, help="随机墙密度(0..0.6)")
    p_sb_env_eval.add_argument("--wall-seed", type=int, default=None, help="随机墙种子")
    p_sb_env_eval.add_argument("--maze", action="store_true",
                               help="Phase 4.5 迷宫地形(recursive backtracker;seed 作 maze_seed)。覆盖随机墙。")
    p_sb_env_eval.add_argument("--maze-braid", type=float, default=0.2,
                               help="迷宫去死胡同比例(0=完美最难,0.2 地牢感,1.0 全去)。默认 0.2。")
    p_sb_env_eval.add_argument("--ego-actions", action="store_true",
                               help="Phase 4.8:ego-relative action(forward/turn;config 级,across all arms)")
    p_sb_env_eval.add_argument(
        "--max-steps", type=int, default=None,
        help="per-run 环境动作预算(act 次数,含 turn/blocked;默认 30)",
    )
    p_sb_env_eval.add_argument(
        "--max-main-turns", type=int, default=None,
        help="主脑模型轮次安全上限(默认由动作预算派生为 max(4x, +8))",
    )
    p_sb_env_eval.add_argument("--repeats", type=int, default=3, help="每 (config,arm) 重复 run 数(pilot=3;formal≥10)")
    p_sb_env_eval.add_argument("--metronome-period", type=int, default=3,
                               help="Phase 4.1 push 臂:每 N 步注入 region_status(默认 3;正式扫 {2,3,5})")
    p_sb_env_eval.add_argument("--arms", default="memory-strategy",
                               help="臂预设(memory-strategy=D.3+Echo 控制臂 / memory-baseline / all)")
    p_sb_env_eval.add_argument("--arm", action="append", default=None,
                               help="显式 feature-config(可多次):如 --arm mem=region,strat=real --arm mem=region,strat=echo "
                                    "(覆盖 --arms;未来加脑区=加 flag,不动预设)")
    p_sb_env_eval.add_argument("--max-cost-usd", type=float, default=None, help="全局成本上限(超即停+标 cost_capped)")
    p_sb_env_eval.add_argument("--max-tokens", type=int, default=None)
    p_sb_env_eval.add_argument("--thinking", default="off", choices=["off", "on"],
                               help="主脑思考(off=便宜快非推理,默认;on=provider 默认)")
    p_sb_env_eval.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    p_sb_env_eval.add_argument("--out", default=None, help="报告输出目录(默认 .brain-region/sandbox/)")

    p_sb_delivery_eval = p_sb_sub.add_parser(
        "delivery-eval",
        help="城区配送成对 A/B:main_only vs grounded 导航执行脑区，统计效率、动作卸载、token 与成本",
    )
    p_sb_delivery_eval.add_argument("--main-brain", default=None, help="主脑模型")
    p_sb_delivery_eval.add_argument("--sizes", default="9", help="地图边长，逗号分隔(默认 9)")
    p_sb_delivery_eval.add_argument("--seeds", default="0,1", help="场景种子，逗号分隔(默认 0,1)")
    p_sb_delivery_eval.add_argument("--orders", type=int, default=2, help="每局订单数(1..8;默认 2)")
    p_sb_delivery_eval.add_argument("--vehicles", type=int, default=2, help="每局车辆数(0..8;默认 2)")
    p_sb_delivery_eval.add_argument("--visibility-radius", type=int, default=1, help="车辆发现半径(默认 1)")
    p_sb_delivery_eval.add_argument("--max-env-actions", type=int, default=120, help="每局环境动作预算(默认 120)")
    p_sb_delivery_eval.add_argument("--max-main-turns", type=int, default=None, help="每局主脑轮次上限")
    p_sb_delivery_eval.add_argument("--option-actions", type=int, default=16, help="每次脑区执行动作上限(1..16)")
    p_sb_delivery_eval.add_argument("--repeats", type=int, default=2, help="每个 config 每臂重复次数(默认 2)")
    p_sb_delivery_eval.add_argument("--max-cost-usd", type=float, default=None, help="全局成本上限")
    p_sb_delivery_eval.add_argument("--max-tokens", type=int, default=None)
    p_sb_delivery_eval.add_argument("--thinking", default="off", choices=["off", "on"])
    p_sb_delivery_eval.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    p_sb_delivery_eval.add_argument("--out", default=None, help="报告输出目录(默认 .brain-region/sandbox/)")

    p_ins = sub.add_parser(
        "inspect",
        help="只读调试窗口（v5.x）：activation/memory/run/calibration 可观测面，不调模型不写",
    )
    p_ins.add_argument("--view", default="all",
                       choices=["all", "activation", "memory", "run", "calibration"],
                       help="视图过滤（默认 all 出全部 4 section）")
    p_ins.add_argument("--run", default=None, help="run view：run_id（省略 → 最近 N run 历史表）")
    p_ins.add_argument("--region", default=None, help="memory view：按 region 过滤")
    p_ins.add_argument("--judge", default=None, help="calibration view：按 judge_id 过滤")
    p_ins.add_argument("--goal", default=None, help="activation view 输入")
    p_ins.add_argument("--problem", default=None, help="activation view 输入")
    p_ins.add_argument("--context", default=None, help="activation view 输入")
    p_ins.add_argument("--gold-regions", default=None, help="activation view：逗号分隔 gold region（判漏唤醒）")
    p_ins.add_argument("--memory-preview-k", type=int, default=3)
    p_ins.add_argument("--top-k", type=int, default=3, help="activation view：wake top_k")
    p_ins.add_argument("--escalate-confidence", type=float, default=0.5)
    p_ins.add_argument("--history-limit", type=int, default=20, help="run view 无 --run 时列多少条")
    p_ins.add_argument("--output-file", default=None, help="写文件（默认 stdout，json）")

    p_snap = sub.add_parser(
        "snapshot",
        help="脑状态可视化（Phase 1）：投影 Inspector → 自包含静态 HTML 面板（region 中心）",
    )
    p_snap.add_argument("--goal", default=None, help="激活查询输入（给则出 Activation 段）")
    p_snap.add_argument("--problem", default=None, help="激活查询输入")
    p_snap.add_argument("--context", default=None, help="激活查询输入")
    p_snap.add_argument("--gold-regions", default=None, help="逗号分隔 gold region（判漏唤醒）")
    p_snap.add_argument("--run", default=None, help="聚焦单个 run（出 timeline 详情）")
    p_snap.add_argument("--region", default=None, help="memory view 按 region 过滤")
    p_snap.add_argument("--judge", default=None, help="calibration view 按 judge 过滤")
    p_snap.add_argument("--history-limit", type=int, default=20)
    p_snap.add_argument("--memory-preview-k", type=int, default=5)
    p_snap.add_argument("--top-k", type=int, default=3)
    p_snap.add_argument("--out", default=None, help="HTML 输出路径（默认 ./brain_region_snapshot.html）")
    p_snap.add_argument("--save", default=None, help="把 snapshot 落盘 JSON（可后续 --from 复渲染）")
    p_snap.add_argument("--from", dest="from_file", default=None,
                        help="从已存 snapshot JSON 加载渲染（不调 Inspector/DB，确定性）")
    p_snap.add_argument("--json", dest="as_json", action="store_true", help="输出 snapshot dict 到 stdout（不渲染 HTML）")
    p_snap.add_argument("--diff", nargs=2, metavar=("A", "B"), default=None,
                        help="对比两 snapshot JSON（A before / B after）→ diff HTML（不调 Inspector/DB）")
    p_snap.add_argument("--label-a", default="A", help="diff 页 A 侧标签")
    p_snap.add_argument("--label-b", default="B", help="diff 页 B 侧标签")
    p_snap.add_argument("--open", dest="open_browser", action="store_true", help="写完后用浏览器打开 HTML")

    p_debug = sub.add_parser(
        "debug",
        help="本地调试仪表盘：持续刷新脑区激活强度、调用状态和建议工具",
    )
    p_debug.add_argument("--host", default="127.0.0.1")
    p_debug.add_argument("--port", type=int, default=8765)
    p_debug.add_argument("--goal", default=None)
    p_debug.add_argument("--problem", default=None)
    p_debug.add_argument("--context", default=None)
    p_debug.add_argument("--gold-regions", default=None, help="逗号分隔 gold region，用于观察漏唤醒")
    p_debug.add_argument("--run", default=None)
    p_debug.add_argument("--region", default=None)
    p_debug.add_argument("--judge", default=None)
    p_debug.add_argument("--history-limit", type=int, default=20)
    p_debug.add_argument("--memory-preview-k", type=int, default=5)
    p_debug.add_argument("--top-k", type=int, default=5)
    p_debug.add_argument("--refresh-ms", type=int, default=2000)
    p_debug.add_argument("--open", dest="open_browser", action="store_true")

    return parser


def _eval_markdown(result: dict) -> str:
    """eval 汇总的简易 markdown 渲染（json 是主输出）。"""
    s = result.get("summary", {})
    pv = s.get("per_variant", {})
    lines = [
        f"# Eval run {result.get('run_id', '')}", "",
        f"tasks={result.get('n_tasks')} variants={result.get('variants')} "
        f"judges={result.get('judge_models')}", "",
    ]
    for name, m in pv.items():
        lines.append(
            f"- **{name}**: useful_rate={m.get('useful_advice_rate')} "
            f"cost/useful={m.get('cost_per_useful_advice')} "
            f"mean_overall={m.get('mean_overall')} "
            f"p50={m.get('latency_p50_ms')}ms p95={m.get('latency_p95_ms')}ms"
        )
    sanity = s.get("sanity", {})
    if sanity.get("errors"):
        lines += ["", "## ❌ Sanity errors"] + [f"- {e}" for e in sanity["errors"]]
    if sanity.get("warnings"):
        lines += ["", "## ⚠️ Sanity warnings"] + [f"- {w}" for w in sanity["warnings"]]
    return "\n".join(lines)


def _calibrate_markdown(result: dict) -> str:
    """calibrate 汇总的简易 markdown 渲染（review + advice 两种 summary 都兼容）。"""
    s = result.get("summary", {})
    verdict = "✅ 校准达标" if s.get("calibrated") else "❌ 未达标（judge/rubric 需调）"
    lines = [
        f"# Calibrate {result.get('run_id', '')}", "",
        f"judges={result.get('judge_models')} pairs={result.get('n_pairs')} threshold={s.get('threshold')}",
        f"agreement={s.get('agreed')}/{s.get('total')} = {s.get('agreement_rate')} → {verdict}",
    ]
    if "wilson_lower" in s:
        lines.append(f"wilson_lower={s.get('wilson_lower')} tie_rate={s.get('tie_rate')}（下界过门槛才 calibrated；n 小不硬放行）")
    lines += ["", "## 按 failure_mode"]
    for fm, rate in (s.get("per_failure_mode") or {}).items():
        lines.append(f"- {fm}: {rate}")
    if s.get("per_metric"):
        lines += ["", "## 按 metric"] + [
            f"- {m}: {r if isinstance(r, (int, float)) else r.get('agreement')}"
            for m, r in s["per_metric"].items()
        ]
    if s.get("penalty_metrics"):
        lines += ["", "## penalty metrics（lower=better，diagnostic）"] + [
            f"- {m}: correct_direction={r.get('correct_direction_rate')}"
            for m, r in s["penalty_metrics"].items()
        ]
    if s.get("wrong_pairs"):
        lines += ["", "## ❌ 错判（good 未 > bad）"] + [f"- {w}" for w in s["wrong_pairs"]]
    return "\n".join(lines)


def _routing_markdown(result: dict) -> str:
    """routing 汇总的简易 markdown 渲染。"""
    s = result.get("summary", {})
    pv = s.get("per_variant", {})
    lines = [
        f"# Routing eval {result.get('run_id', '')}", "",
        f"tasks={result.get('n_tasks')} variants={result.get('variants')}（A=no_defense vs B=full）", "",
        "| variant | precision | recall | missed_wake_rate | false_wake_rate |",
        "|---|---|---|---|---|",
    ]
    for name, m in pv.items():
        lines.append(
            f"| {name} | {m.get('precision')} | {m.get('recall')} | "
            f"{m.get('missed_wake_rate')} | {m.get('false_wake_rate')} |"
        )
    sanity = s.get("sanity", {})
    if sanity.get("errors"):
        lines += ["", "## ❌ Sanity errors"] + [f"- {e}" for e in sanity["errors"]]
    if sanity.get("warnings"):
        lines += ["", "## ⚠️ Sanity warnings"] + [f"- {w}" for w in sanity["warnings"]]
    return "\n".join(lines)


def _outcome_markdown(result: dict) -> str:
    """outcome 汇总的简易 markdown 渲染（json 是主输出）。"""
    s = result.get("summary", {})
    pv = s.get("per_variant", {})
    gate = result.get("gate", {})
    lines = [
        f"# Outcome eval {result.get('run_id', '')}", "",
        f"tasks={result.get('n_tasks')} variants={result.get('variants')} "
        f"judges={result.get('judge_models')} "
        f"overlap(routed≡default)={s.get('routed_default_overlap_rate')}", "",
        "| variant | useful_rate | cost/useful | inference$ | missed_wake | missed_critical | p95ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in pv.items():
        lines.append(
            f"| {name} | {m.get('useful_advice_rate')} | {m.get('cost_per_useful_advice')} | "
            f"{m.get('inference_cost_usd')} | {m.get('missed_wake_rate')} | "
            f"{m.get('missed_critical_total')} | {m.get('latency_p95_ms')} |"
        )
    lines += ["", f"## Gate: {gate.get('decision')}"]
    diag = gate.get("diagnostics") or {}
    if diag.get("pilot"):
        lines.append(f"_pilot 模式：有效 n={diag.get('effective_n')} < formal_min_n，不宣称可信闸门_")
    for r in (gate.get("reasons") or []):
        lines.append(f"- {r}")
    ci_block = []
    for label, key in (("cost_ratio", "cost_ratio_ci"), ("useful_delta", "useful_delta_ci"),
                       ("missed_critical_delta", "missed_critical_delta_ci")):
        ci = diag.get(key) or {}
        if ci.get("point") is not None:
            ci_block.append(
                f"- {label}: point={round(ci['point'], 4)} CI=[{round(ci['low'], 4)}, {round(ci['high'], 4)}]"
                f" eff_rate={ci.get('effective_rate')}"
            )
    if ci_block:
        lines += ["", "## Bootstrap CI（估计量层，per metric 独立流）"] + ci_block
    sanity = s.get("sanity", {})
    if sanity.get("errors"):
        lines += ["", "## ❌ Sanity errors"] + [f"- {e}" for e in sanity["errors"]]
    if sanity.get("warnings"):
        lines += ["", "## ⚠️ Sanity warnings"] + [f"- {w}" for w in sanity["warnings"]]
    return "\n".join(lines)


def _capability_markdown(result: dict) -> str:
    """capability 基准汇总的简易 markdown 渲染（json 是主输出）。"""
    s = result.get("summary", {})
    lines = [
        f"# Capability eval {result.get('run_id', '')}", "",
        f"声明范围: {result.get('claim_scope', '')}",
        f"solvers={result.get('solvers')} arms={result.get('variants')} n/α={result.get('n_instances_per_alpha')}",
        f"n_vars={s.get('n_vars')} α={s.get('alphas')} base_seed={s.get('base_seed')}", "",
        "## per-cell(solver|α|arm):solve_rate_given_valid / valid_output / output_tok",
    ]
    for cell, m in (s.get("per_cell") or {}).items():
        lines.append(
            f"- {cell}: solve_given_valid={m.get('solve_rate_given_valid')} "
            f"valid_out={m.get('valid_output_rate')} overall={m.get('overall_solve_rate')} "
            f"out_tok={m.get('output_tokens_given_valid')} rea_tok={m.get('reasoning_tokens_given_valid')} call_fail={m.get('call_fail_rate')} n={m.get('n')}"
        )
    gaps = s.get("gaps") or {}
    if gaps:
        lines += ["", "## gaps(risk_difference per α;distractor_vs_neutral 为主)"]
        for glabel, peralpha in gaps.items():
            lines.append(f"- **{glabel}**:")
            for ak, gm in peralpha.items():
                rd = gm.get("risk_difference") or {}
                orr = (gm.get("odds_ratio") or {}).get("point")
                lines.append(f"  - {ak}: Δ={rd.get('point')} CI[{rd.get('low')}, {rd.get('high')}] OR={orr}")
    inter = s.get("interaction") or {}
    solver_inter = {k: v for k, v in inter.items() if isinstance(v, dict)}
    if solver_inter:
        lines += ["", f"## 交互 Δ_interaction(easy={inter.get('easy_alpha')}→hard={inter.get('hard_alpha')})"]
        for k, v in solver_inter.items():
            lines.append(f"- {k}: gap_easy={v.get('gap_easy')} gap_hard={v.get('gap_hard')} Δ={v.get('delta_interaction')}")
        lines.append(f"_{inter.get('note')}_")
    claimed = s.get("claimed_note_usage")
    if claimed:
        lines += ["", "## claimed_note_usage(manip-check,self-report)", *[f"- {k}: {v}" for k, v in claimed.items()]]
    budget = s.get("budget") or {}
    if budget.get("incomplete"):
        lines += ["", f"⚠️ 预算超限 incomplete:dropped {len(budget.get('dropped_cells') or [])} cells"]
    return "\n".join(lines)


def _capability_skill_bloat_markdown(result: dict) -> str:
    """Phase 3D 多家族 skill-bloat 汇总:overall(跨族 generality 头条)+ per-family(K×臂 + contrasts + 诊断)。"""
    s = result.get("summary", {})
    ks = s.get("ks") or []
    families = s.get("families") or []
    solvers = result.get("solvers") or []
    lines = [
        f"# Capability eval (skill-bloat,多家族) {result.get('run_id', '')}", "",
        f"声明范围: {result.get('claim_scope', '')}",
        f"solvers={solvers} families={families} ks={ks} arms={s.get('arms')} n={result.get('n_instances')} "
        f"table_size={s.get('table_size')} n_examples={s.get('n_examples')} pool={s.get('n_skills')} "
        f"base_seed={s.get('base_seed')} max_tokens={s.get('max_tokens')}", "",
    ]
    overall = s.get("overall") or {}
    macro = overall.get("macro_mean") or {}
    gen = overall.get("generalization") or {}
    lines.append("## Overall(跨家族;§0 generality 头条)")
    lines.append("| contrast | " + " | ".join(solvers) + " |")
    lines.append("|" + "|".join(["---"] * (1 + len(solvers))) + "|")
    labels = ([f"degradation_at_k{k}" for k in ks] + [f"plausibility_effect_at_k{k}" for k in ks]
              + [f"reasoning_cost_at_k{k}" for k in ks] + [f"coverage_value_at_k{k}" for k in ks])
    for label in labels:
        row = [label]
        for slv in solvers:
            mm = (macro.get(label) or {}).get(slv)
            g = (gen.get(label) or {}).get(slv) or {}
            star = "✅" if g.get("generalizes") else ("⚠️" if g.get("n_ci_excludes_0", 0) else "·")
            row.append(f"{star} mean={mm} ({g.get('n_ci_excludes_0')}/{g.get('n_families')} 族 CI排0)")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    def _rd(ps: dict) -> str:
        rd = ps.get("risk_difference") or {}
        return f"Δ={rd.get('point')} CI[{rd.get('low')},{rd.get('high')}] n={ps.get('n')}"

    for fam in families:
        fa = (s.get("per_family") or {}).get(fam) or {}
        kc, contrasts, per_cell = fa.get("k_curve") or {}, fa.get("contrasts") or {}, fa.get("per_cell") or {}
        lines.append(f"## family={fam}")
        for slv in solvers:
            curve = kc.get(slv) or {}
            header = ["arm"] + [f"K={k}" for k in ks]
            lines.append(f"_solver={slv}_  oracle(upper bound)={curve.get('oracle')}")
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
            for arm in ("plausible", "garbage", "random_subset"):
                lines.append("| " + " | ".join([arm] + [str((curve.get(arm) or {}).get(str(k))) for k in ks]) + " |")
        for k in ks:
            for slv in solvers:
                ps = (contrasts.get(f"degradation_at_k{k}") or {}).get(slv)
                if ps:
                    lines.append(f"- degradation_k{k}[{slv}] {_rd(ps)}")
                ps = (contrasts.get(f"reasoning_cost_at_k{k}") or {}).get(slv)
                if ps:
                    md = ps.get("mean_diff") or {}
                    lines.append(f"- reasoning_cost_k{k}[{slv}] Δ={md.get('point')} CI[{md.get('low')},{md.get('high')}]")
        for cell, m in per_cell.items():
            if "|plausible_k" in cell:
                lines.append(f"- {cell}: wrong_sel={m.get('wrong_selection_rate')} "
                             f"top={m.get('top_distractor')} entropy={m.get('selection_entropy')} "
                             f"inv_tok={m.get('inventory_tokens_mean')}")
        lines.append("")
    budget = s.get("budget") or {}
    if budget.get("incomplete"):
        lines += [f"⚠️ 预算超限:dropped {len(budget.get('dropped_cells') or [])} cells "
                  f"(spent={budget.get('spent_usd')}/{budget.get('max_usd')})"]
    lines += ["", f"_{s.get('note')}_"]
    return "\n".join(lines)


def _capability_mixed_router_markdown(result: dict) -> str:
    """Phase 3E mixed-pool 跨区域 router 汇总:overall(分解 contrast 头条)+ per correct_family(4 arm 原始 + 分解)。"""
    s = result.get("summary", {})
    families = s.get("families") or []
    solvers = result.get("solvers") or []
    lines = [
        f"# Capability eval (skill-bloat mixed,跨区域 router) {result.get('run_id', '')}", "",
        f"声明范围: {result.get('claim_scope', '')}",
        f"solvers={solvers} router={result.get('router_model')} families={families} "
        f"n={result.get('n_instances')} table_size={s.get('table_size')} n_examples={s.get('n_examples')} "
        f"n_within={s.get('n_within')} n_cross/fam={s.get('n_cross_per_family')} base_seed={s.get('base_seed')}", "",
        "分解(oracle gap):mixed_all→router_gold=跨区域 scoping(头条,router 可修);"
        "router_gold→oracle=同族内 bloat(router 修不了,诚实);router vs router_gold=误路由代价。", "",
    ]
    overall = s.get("overall") or {}
    macro = overall.get("macro_mean") or {}
    route_acc = overall.get("route_accuracy") or {}
    lines.append("## Overall(跨 correct_family macro-mean)")
    lines.append("| contrast | " + " | ".join(solvers) + " |")
    lines.append("|" + "|".join(["---"] * (1 + len(solvers))) + "|")
    for lab, desc in [
        ("cross_region_value", "跨区域 scoping(router_gold−mixed_all)"),
        ("within_region_bloat", "同族内 bloat(oracle−router_gold)"),
        ("routing_error_cost_solve", "误路由代价(router_gold−router)"),
        ("cross_region_reasoning", "跨区域 reasoning(mixed_all−router_gold)"),
        ("within_region_reasoning", "同族内 reasoning(router_gold−oracle)"),
    ]:
        row = [desc]
        for slv in solvers:
            row.append(str((macro.get(lab) or {}).get(slv)))
        lines.append("| " + " | ".join(row) + " |")
    acc_row = ["route_accuracy"]
    for slv in solvers:
        acc_row.append(str(route_acc.get(slv)))
    lines.append("| " + " | ".join(acc_row) + " |")
    lines.append("")

    def _rd(ps: dict) -> str:
        rd = (ps or {}).get("risk_difference") or (ps or {}).get("mean_diff") or {}
        return f"Δ={rd.get('point')} CI[{rd.get('low')},{rd.get('high')}] n={ps.get('n')}"

    for fam in families:
        fa = (s.get("per_family") or {}).get(fam) or {}
        pc = fa.get("per_cell") or {}
        contrasts = fa.get("contrasts") or {}
        racc = fa.get("route_accuracy") or {}
        lines.append(f"## correct_family={fam}")
        for slv in solvers:
            lines.append(f"_solver={slv}_  route_accuracy={racc.get(slv)}")
            lines.append("| arm | solve | reasoning_tok | inv_tok | cost |")
            lines.append("|---|---|---|---|---|")
            for arm in ("oracle", "router_gold", "mixed_all", "router"):
                m = pc.get(f"{slv}|{arm}") or {}
                lines.append(f"| {arm} | {m.get('solve_rate')} | {m.get('reasoning_tokens_mean')} | "
                             f"{m.get('inventory_tokens_mean')} | {m.get('cost_mean')} |")
        for lab, desc in [("cross_region_value", "跨区域"), ("within_region_bloat", "同族内"),
                          ("routing_error_cost_solve", "误路由")]:
            for slv in solvers:
                ps = (contrasts.get(lab) or {}).get(slv)
                if ps:
                    lines.append(f"- {desc}[{slv}] {_rd(ps)}")
        lines.append("")
    budget = s.get("budget") or {}
    if budget.get("incomplete"):
        lines += [f"⚠️ 预算超限:dropped {len(budget.get('dropped_cells') or [])} cells "
                  f"(spent={budget.get('spent_usd')}/{budget.get('max_usd')})"]
    lines += ["", f"_{s.get('note')}_"]
    return "\n".join(lines)


def run_inspect(args) -> None:
    """inspect 子命令：只读调试窗口，json 输出（结构化 debug 数据，无需 markdown）。"""
    from brainregion.inspector import inspect as inspect_facade

    gold = [g.strip() for g in (args.gold_regions or "").split(",") if g.strip()] or None
    result = inspect_facade(
        view=args.view,
        goal=args.goal or "", problem=args.problem or "", context=args.context or "",
        gold_regions=gold, run_id=args.run or None, region=args.region or None,
        judge_id=args.judge or None, escalate_confidence=args.escalate_confidence,
        top_k=args.top_k, memory_preview_k=args.memory_preview_k,
        history_limit=args.history_limit,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_file:
        Path(args.output_file).write_text(text, encoding="utf-8")
    else:
        print(text)


def run_snapshot(args) -> None:
    """snapshot 子命令：build（或 --from 加载）→ 渲染 HTML / 落盘 JSON / stdout JSON。

    --from 硬不变量：从已存 snapshot.json 加载渲染，**绝不调 Inspector/DB**（save→render 确定性）。
    """
    import webbrowser

    from brainregion.viz import BrainSnapshot, build_snapshot, render_diff, render_html

    # 0. --diff:对比两 snapshot → diff HTML（不调 Inspector/DB,确定性）
    if args.diff:
        from brainregion.viz import build_diff

        a = BrainSnapshot.from_dict(json.loads(Path(args.diff[0]).read_text(encoding="utf-8")))
        b = BrainSnapshot.from_dict(json.loads(Path(args.diff[1]).read_text(encoding="utf-8")))
        diff = build_diff(a, b, label_a=args.label_a, label_b=args.label_b)
        out_path = Path(args.out) if args.out else Path.cwd() / "brain_region_diff.html"
        out_path.write_text(render_diff(diff), encoding="utf-8")
        print(f"diff → {out_path}")
        if args.open_browser:
            webbrowser.open(out_path.resolve().as_uri())
        return

    # 1. 取 snapshot：--from 优先（不调 Inspector/DB），否则 build
    if args.from_file:
        data = json.loads(Path(args.from_file).read_text(encoding="utf-8"))
        snap = BrainSnapshot.from_dict(data)
    else:
        gold = [g.strip() for g in (args.gold_regions or "").split(",") if g.strip()] or None
        snap = build_snapshot(
            goal=args.goal or "", problem=args.problem or "", context=args.context or "",
            gold_regions=gold, run_id=args.run or None, region=args.region or None,
            judge_id=args.judge or None, history_limit=args.history_limit,
            memory_preview_k=args.memory_preview_k, top_k=args.top_k,
        )

    # 2. --save：落盘 JSON（capture；无论是否渲染 HTML）
    if args.save:
        Path(args.save).write_text(
            json.dumps(snap.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 3. 输出：--json 走 stdout；否则渲染 HTML 写文件（+ 可选 --open）
    if args.as_json:
        print(json.dumps(snap.to_dict(), ensure_ascii=False, indent=2))
        return

    html_text = render_html(snap)
    out_path = Path(args.out) if args.out else Path.cwd() / "brain_region_snapshot.html"
    out_path.write_text(html_text, encoding="utf-8")
    print(f"快照 → {out_path}")
    if args.open_browser:
        webbrowser.open(out_path.resolve().as_uri())


def run_debug(args) -> None:
    """启动本地 HTTP 调试仪表盘。"""
    from brainregion.viz.debug_server import DebugDashboardOptions, parse_gold_regions, serve_debug_dashboard

    options = DebugDashboardOptions(
        host=args.host,
        port=args.port,
        goal=args.goal or "",
        problem=args.problem or "",
        context=args.context or "",
        gold_regions=parse_gold_regions(args.gold_regions),
        run_id=args.run or None,
        region=args.region or None,
        judge_id=args.judge or None,
        history_limit=args.history_limit,
        memory_preview_k=args.memory_preview_k,
        top_k=args.top_k,
        refresh_ms=args.refresh_ms,
    )
    serve_debug_dashboard(options, open_browser=args.open_browser)


def main() -> None:
    # Windows GBK 控制台无法 print emoji（🔴⚠️ 等，output/markdown 与 eval 都会用到）→ 重配 stdout
    # 为 utf-8 + errors=replace，至少不崩（实际显示取决于终端 codepage）。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — stdout 不可重配（如被捕获）时静默
        pass
    args = build_parser().parse_args()
    if args.command in {"eval", "calibrate", "routing", "outcome", "capability"}:
        from brainregion.eval import cli as eval_cli

    if args.command == "eval":
        result = asyncio.run(eval_cli.run(args))
        if args.output_format != "json":
            result["rendered"] = _eval_markdown(result)
        _emit(result, args)
        return
    if args.command == "calibrate":
        runner = eval_cli.run_calibrate_advice if getattr(args, "advice", False) else eval_cli.run_calibrate
        result = asyncio.run(runner(args))
        if args.output_format != "json":
            result["rendered"] = _calibrate_markdown(result)
        _emit(result, args)
        return
    if args.command == "routing":
        result = eval_cli.run_routing(args)  # 同步、不调模型
        if args.output_format != "json":
            result["rendered"] = _routing_markdown(result)
        _emit(result, args)
        return
    if args.command == "outcome":
        result = asyncio.run(eval_cli.run_outcome(args))
        if args.output_format != "json":
            result["rendered"] = _outcome_markdown(result)
        _emit(result, args)
        return
    if args.command == "capability":
        result = asyncio.run(eval_cli.run_capability(args))
        if args.output_format != "json":
            if result.get("mode") == "skill_bloat_mixed":
                md = _capability_mixed_router_markdown
            elif result.get("mode") == "skill_bloat":
                md = _capability_skill_bloat_markdown
            else:
                md = _capability_markdown
            result["rendered"] = md(result)
        _emit(result, args)
        return
    if args.command == "sandbox":
        if args.sandbox_command == "delegation-shadow":
            from brainregion.sandbox.delegation_shadow import (
                render_shadow_gate_summary,
                replay_shadow_report,
            )

            try:
                result = replay_shadow_report(args.report, max_steps=args.max_steps)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            print(render_shadow_gate_summary(result))
            return
        from brainregion.sandbox import cli as sandbox_cli

        if args.sandbox_command == "run":
            asyncio.run(sandbox_cli.run(args))
        elif args.sandbox_command == "eval":
            asyncio.run(sandbox_cli.run_eval(args))
        elif args.sandbox_command == "delegation-eval":
            asyncio.run(sandbox_cli.run_delegation_eval(args))
        elif args.sandbox_command == "cognitive-eval":
            asyncio.run(sandbox_cli.run_cognitive_eval(args))
        elif args.sandbox_command == "functional-region-eval":
            asyncio.run(sandbox_cli.run_functional_regions_eval(args))
        elif args.sandbox_command == "tool-result-eval":
            asyncio.run(sandbox_cli.run_tool_result_lifecycle_eval(args))
        elif args.sandbox_command == "verify-brain":
            asyncio.run(sandbox_cli.verify_brain(args))
        elif args.sandbox_command == "env":
            asyncio.run(sandbox_cli.run_env(args))
        elif args.sandbox_command == "env-eval":
            asyncio.run(sandbox_cli.run_env_eval(args))
        elif args.sandbox_command == "delivery-eval":
            asyncio.run(sandbox_cli.run_delivery_eval(args))
        return
    if args.command == "inspect":
        run_inspect(args)
        return
    if args.command == "snapshot":
        run_snapshot(args)
        return
    if args.command == "debug":
        run_debug(args)
        return
    common = dict(
        adapter=args.adapter, panel=args.panel, dimensions=args.dimensions,
        output_format=args.output_format, retrieve_top_k=args.retrieve_top_k,
        extra_context=args.extra_context, effort=args.effort,
        max_cost_usd=args.max_cost_usd, timeout=args.timeout,
    )
    from brainregion.server import review_document

    if args.command == "code":
        files = {f: Path(f).read_text(encoding="utf-8") for f in args.files}
        result = asyncio.run(review_document(content="", document_type="code", files=files, **common))
    else:
        content = _read_text_input(args)
        dtype = "markdown" if args.command == "plan" else args.document_type
        result = asyncio.run(review_document(content=content, document_type=dtype, **common))
    _emit(result, args)


if __name__ == "__main__":
    main()
