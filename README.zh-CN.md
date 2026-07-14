# 脑区 BrainRegion

[English](README.md) | [简体中文](README.zh-CN.md)

![CI](https://github.com/fanghaoling/brainregion/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/github/license/fanghaoling/brainregion.svg)
![Python](https://img.shields.io/badge/python-%E2%88%803.10-blue.svg)

BrainRegion 是面向 review、consultation、planning 和 memory 的 AI 协作基础设施。

这个项目原名 `design-review-mcp`。内部 Python 包名已经迁移到 `brainregion`；重命名期间旧命令别名仍然保留，避免已有配置突然失效。

当前 MCP server 和 CLI 可以把方案、代码变更或文档分发给多个 LLM reviewer，结合项目知识库检索、finding 归一化和共识汇总，生成更容易落地处理的审查报告。同时也支持外援会诊：当主模型卡住或需要第三方视角时，用 `consult_problem` 调用外部专家模型。

核心 pipeline 保持项目无关，项目特定行为放在 adapter 中，因此默认形态适合通用的产品、架构、代码和文档设计审查。可选 adapter 可以在不改变 core pipeline 的前提下注入领域知识。

## 功能概览

- 审查方案、代码、Markdown、ADR、RFC 和配置文档。
- 支持 `planner`、`safety`、`architecture`、`performance`、`feasibility`、`visionary` 等 reviewer 角色。
- 支持单模型或多模型 panel，包括官方 LiteLLM provider 和 OpenAI/Anthropic 兼容中转站。
- 审查前检索框架知识库和项目本地知识库。
- 将重复 finding 归一到 canonical bucket，并区分 consensus、majority、individual。
- 支持 JSON、Markdown、SARIF 输出。
- 支持 Review Memory：通过 `mark_finding` 记录 finding 是否采纳，后续用于校准模型可信度。
- 支持外援会诊：通过 `consult_problem` 请求专家模型建议，并用 `mark_advice` 记录建议是否有用。
- 支持任务规划：通过 `plan_task` 把目标拆成可执行计划，再交给 `review_plan` 审查。
- 支持脑区路由：通过 `route_regions` 本地判断目标/问题可能相关的 Brain Regions，作为后续 Context Scheduler 的前置能力。
- 支持显式工作流建议：通过 `suggest_workflow` 推荐下一步手动工具调用，但不自动调用工具或模型。
- 支持模型路由自检：通过 `list_model_routes` 区分裸模型名和 endpoint 中转模型，避免走错 key。
- 支持模型面板推荐：通过 `suggest_panel` 基于 profile 标签和成本/速度/质量评分推荐 panel，但不调用模型。
- 支持 builtin、全局配置、项目配置、环境变量、显式调用参数的分层默认值。

## 架构

主要模块都可以替换。项目特定逻辑应留在 adapter 内，不进入 `core/`。

| 层 | 协议 | 默认实现 |
|---|---|---|
| `ModelBackend` | `async complete(...)` | `LiteLLMBackend` |
| `KnowledgeProvider` | `retrieve/list_cases/add_case` | `YamlKnowledgeProvider` |
| `ProjectAdapter` | `read_context/version/convention + reviewers/knowledge` | `GenericAdapter`、可选领域 adapter |
| `ReportRenderer` | `render(ReviewReport)` | Markdown / JSON / SARIF renderer |
| `Stage` | `process(ctx) -> ctx` | retrieve、context、prompt、review、parse、normalize、consensus、score |

后续增加 Rust、C++、Web 等项目类型时，通常只需要新增 adapter 包，不需要改 core pipeline。

## Review Pipeline

```text
ReviewDocument
  -> RetrieveStage
  -> ContextStage
  -> PromptStage
  -> ReviewStage      # panel x dimensions fan-out
  -> ParseStage
  -> NormalizeStage   # canonical finding bucket
  -> ConsensusStage
  -> ScoreStage
  -> ReviewReport
```

pipeline 用来降低“模型很自信但没有证据”的反馈：

- finding 必须带 evidence quote。
- 知识库检索可以注入项目踩坑和版本相关案例。
- reviewer prompt 按角色拆分。
- canonical normalize 减少不同模型之间的同义重复。
- calibrated confidence 会结合模型共识、严重性、知识库命中和 Review Memory。

## 外援会诊 Consult

`consult_problem` 用于主模型卡住、没有把握、连续调试失败或需要第三方视角时，请外部专家模型给出结构化建议。它不执行命令、不修改文件，只返回可追踪的诊断和下一步实验。

```python
consult_problem(
    problem="FlowField 更新偶发死锁",
    context="Unity ECS 项目，已尝试双 Buffer 和 JobHandle.CombineDependencies。",
    logs="偶发卡在 CompleteDependency() 附近",
    attempts=[
        "改成双 Buffer",
        "合并 JobHandle 依赖",
    ],
    question="还有哪些 ECS 架构层面的排查方向？",
    mode="architecture",
)
```

常用 `mode`：

- `debugging`：调试与根因定位。
- `architecture`：架构边界、状态流和长期维护。
- `performance`：性能、延迟、token/API 成本。
- `simplicity`：YAGNI、简化和更小 MVP。
- `game_design`：玩法和玩家体验。
- `challenge`：反方挑战当前想法。
- `planning`：任务拆解、风险和验收标准。

配置建议：

```jsonc
{
  "consult_panel": ["modelbridge_openai/gpt-5.4-mini"],
  "consult_consultants": ["debugger", "critic"],
  "consult_max_cost_usd": 0.03,
  "consult_max_input_chars": 24000
}
```

如果没有配置 `consult_panel`，会诊会回退到 `panel`。生产使用建议单独配置较便宜、响应快的 `consult_panel`，避免一次会诊展开完整审查面板。

`consult_problem` 返回 `consultation_id`，每条 `individual` advice 也带稳定 `id`。如果某条建议有用或无用，可以反馈给 Advice Memory：

```python
mark_advice(
    advice_id="consult-abc123-0",
    consultation_id="consult-abc123",
    decision="accepted",
    reason="定位到了实际竞态",
    outcome="按建议加了最小复现测试"
)
```

`decision` 可选 `accepted` / `rejected` / `partial` / `unknown`。系统只记录 advice 元数据和用户反馈，不保存原始 prompt、问题正文或完整 advice 文本。

## 任务规划 Planner

`plan_task` 用于把目标拆成结构化、可审查的实现计划。它是一个轻量 Planner MVP：不执行命令、不修改文件，也不做多模型辩论；它会按配置的模型面板顺序尝试，返回第一个可解析的计划。

```python
plan_task(
    goal="给 BrainRegion 增加 Planner MVP",
    context="这是一个 Python MCP server，已经有 consult_problem 和 review_plan。",
    constraints=[
        "不要自动执行任务",
        "复用现有预算和输入脱敏护栏",
    ],
    success_criteria=[
        "MCP 工具返回里程碑、任务、风险、验收标准和测试计划",
        "单元测试覆盖解析和路由",
    ],
)
```

推荐流程：

```text
Goal -> plan_task -> review_plan -> implement -> review_code -> mark_finding / mark_advice
```

可选配置：

```jsonc
{
  "planner_panel": ["modelbridge_openai/gpt-5.4-mini"],
  "planner_max_cost_usd": 0.03,
  "planner_max_input_chars": 24000
}
```

如果没有配置 `planner_panel`，规划会回退到 `consult_panel`，再回退到 `panel`。

## 脑区路由 Brain Regions

`route_regions` 是 Region-based context scheduling 的第一步。它刻意保持本地、确定性和只读：不调用模型、不读取记忆，也不会触发 review / consult / planner。它只根据静态 region 定义里的触发词做排序，并返回 activation trace。

```python
route_regions(
    goal="优化 Unity ECS FlowField 系统，减少内存分配",
    files={
        "Assets/Scripts/FlowFieldSystem.cs": "...",
    },
    top_k=3,
)
```

返回结构示例：

```jsonc
{
  "selected": [
    {"id": "unity_ecs", "score": 4, "matched_triggers": [...]},
    {"id": "performance", "score": 4, "matched_triggers": [...]}
  ],
  "trace": {
    "strategy": "deterministic_keyword_v1",
    "input": {"file_contents_used": false}
  }
}
```

内置 regions 目前包括 `planning`、`review`、`debugging`、`performance`、`security`、`memory`、`research` 和 `unity_ecs`。这个工具只提供建议；未来的 scheduler 必须显式决定是否消费它的结果。

## 工作流建议 Workflow Suggestions

`suggest_workflow` 基于 `route_regions` 返回下一步工具调用建议，供主模型或用户显式确认。它仍然是本地、确定性和只读的：不调用模型、不运行 review / consult / planner、不读取记忆，也不修改文件。

```python
suggest_workflow(
    goal="优化 Unity ECS FlowField 系统，并审查实现计划",
    files={
        "Assets/Scripts/FlowFieldSystem.cs": "...",
    },
)
```

返回的建议可能包括 `plan_task`、`consult_problem`、`review_document` 或 `review_code`。每个建议都会包含 `requires_user_approval: true`、原因、建议参数、来源 region 和 trace 元数据。它是从脑区路由走向未来 Context Scheduler 的保守过渡层。

## 知识库

审查质量很依赖项目知识。内置 adapter 包可以带一些种子案例，但最有价值的架构决策、历史 bug 和团队约定，通常还是应该放在项目本地知识库里。

推荐位置：

```text
<project-root>/.brain-region/knowledge/*.yaml
```

旧的 `.design-review/knowledge/` 目录仍会先加载以保持兼容。新的 `.brain-region/knowledge/` 会后加载，并可用相同 `id` 覆盖旧 case。

示例：

```yaml
- id: API-001
  title: "Keep breaking API changes behind a migration path"
  version: {service: ">=2.0"}
  triggers: ["breaking change", "API contract", "migration"]
  category: compatibility
  bad_pattern: "Change a public request or response shape without a versioned fallback or migration notes."
  recommended_pattern: "Add a compatible path, document the migration window, and test old and new clients."
  source: "ADR-014#api-versioning"
```

建议：

- 每条 case 只写一个具体、可复现的 gotcha。
- `triggers` 写方案或代码里可能真实出现的词。
- 敏感项目知识放本地并加入 gitignore。
- 用 `list_knowledge` 查看当前加载了哪些框架和本地案例。

## 安装

```bash
cd <path-to-brain-region-mcp>
uv sync --extra dev
```

运行测试：

```bash
uv run pytest tests/ -q
uv run --extra dev ruff check .
```

## MCP 配置

在 Codex、Claude Code 或其它 MCP client 中注册 stdio server：

```jsonc
{
  "type": "stdio",
  "command": "uv",
  "args": [
    "run",
    "--directory",
    "<path-to-brain-region-mcp>",
    "brain-region-mcp"
  ],
  "env": {
    "UNITY_PROJECT_ROOT": "<path-to-project-root>",
    "BRAIN_REGION_CONFIG": "<path-to-brain-region-mcp>/brain_region_config.json"
  }
}
```

`UNITY_PROJECT_ROOT` 是为了兼容保留的项目根目录变量名，把它指向要审查的项目根目录即可。
API key 建议放在 `.env` 或进程环境变量里。不要提交 `.env` 或本地 `brain_region_config.json`。

## CLI

`brain-region` CLI 和 MCP server 使用同一套 pipeline。重命名期间旧的 `design-review` 命令仍作为 alias 保留。

```bash
uv run brain-region plan path/to/plan.md --output markdown
cat plan.md | uv run brain-region plan -
uv run brain-region plan --text "# Plan" --dimensions planner feasibility
uv run brain-region code src/a.py src/b.py --output sarif --output-file review.sarif
uv run brain-region doc docs/rfc.md --type rfc --output markdown
uv run brain-region --config brain_region_config.json --env-file .env sandbox delivery-eval --main-brain buzz_anthropic/claude-sonnet-5
```

常用参数：

- `--config`：显式加载本地 JSON 配置；这是顶层启动参数，必须放在 `plan`、`sandbox` 等子命令之前。
- `--env-file`：显式加载 API key 文件且不覆盖进程中已有的环境变量；同样必须放在子命令之前。
- `--panel`：模型列表或 endpoint 快捷写法。
- `--dimensions`：审查维度。
- `--adapter`：`auto`、`generic` 或已安装的领域 adapter。
- `--retrieve-top-k`：检索知识库案例数量。
- `--effort`：支持时传入 reasoning/thinking 强度。
- `--max-cost-usd`：预估成本上限。
- `--timeout`：单模型超时。

## 配置

默认值按以下顺序合并：

```text
builtin < global config < project config < env < explicit tool args
```

详见：

- [配置优先级](docs/config_precedence.zh-CN.md)
- [Endpoint 配置](docs/endpoint_config.zh-CN.md)

常用本地配置位置：

```text
<path-to-brain-region-mcp>/brain_region_config.json
```

CLI 不会隐式信任当前工作目录里的同名文件。可通过顶层 `--config`/`--env-file` 传入，或继续使用
`BRAIN_REGION_CONFIG` 和进程环境变量；MCP server 仍建议在 client 配置的 `env` 中声明路径。

`brain_region_config.json` 可以放这些默认值：

- `panel`
- `dimensions`
- `retrieve_top_k`
- `timeout`
- `normalizer_model`
- `effort`
- `max_cost_usd`
- `endpoints`
- `model_profiles`
- `privacy_policy`
- `context_modes`

## 自定义中转站 Endpoint

`endpoints` 用来接入 New API、one-api、OpenRouter 风格代理或内部模型桥接等 OpenAI/Anthropic 兼容中转站。建议一个 endpoint 只对应一种协议。

```json
{
  "endpoints": {
    "modelbridge_openai": {
      "provider": "openai",
      "base_url": "https://www.modelbridge.cloud/v1",
      "api_key_env": "MODEBRIDGE_API_KEY",
      "models": ["gpt-5.5", "gpt-5.4-mini"]
    },
    "modelbridge_anthropic": {
      "provider": "anthropic",
      "base_url": "https://www.modelbridge.cloud",
      "api_key_env": "MODEBRIDGE_API_KEY",
      "models": ["claude-haiku-4-5", "claude-opus-4-8"]
    }
  },
  "panel": ["modelbridge_openai/gpt-5.5", "modelbridge_anthropic/claude-opus-4-8"]
}
```

Panel 快捷写法：

- `"endpoints"`：展开所有 endpoint 下声明的所有模型。
- `"endpoint_id"`：展开某个 endpoint 下声明的所有模型。
- `"endpoint_id/model"`：通过某个 endpoint 调用单个模型。
- `"gpt-4o"`、`"deepseek/deepseek-chat"` 等原生 LiteLLM 字符串不走 endpoint 配置，直接走对应 provider 的环境变量。

例如 `"claude-opus-4-8"` 是裸官方 provider 路由，通常需要 `ANTHROPIC_API_KEY`；而
`"modelbridge_anthropic/claude-opus-4-8"` 会走配置里的中转站，使用它声明的 `MODEBRIDGE_API_KEY`。不确定时可以先调用
`list_model_routes` 查看实际路由。

模型 profile 是可选的描述性元数据，会显示在 `list_model_routes`、`suggest_panel` 和工具返回的 `routing` 里：

```jsonc
{
  "model_profiles": {
    "modelbridge_openai/gpt-5.4-mini": {
      "activation_role": "sleep",
      "tier": "economy",
      "cost": "low",
      "latency": "fast",
      "tags": ["cheap", "fast"],
      "quality_score": 0.65,
      "cost_score": 0.9,
      "speed_score": 0.85
    },
    "modelbridge_anthropic/claude-opus-4-8": {
      "activation_role": "awake",
      "tier": "flagship",
      "cost": "high",
      "tags": ["deep_reasoning", "architecture"],
      "quality_score": 0.98,
      "cost_score": 0.2
    }
  }
}
```

`suggest_panel(strategy="cheap_fast" | "best_reasoning" | "balanced" | "sleep" | "awake" | "structured_output")`
会根据这些 profile 元数据给配置好的模型路由排序，并返回 `selected_panel`。它不会调用模型，也不会自动执行后续工具。

## 成本与 Effort 控制

两个可选控制项：

- `max_cost_usd`：单次 review 预估成本上限。工具会按 panel 顺序保留 job，直到估算成本超过上限。
- `effort`：对支持的 provider 传入 reasoning/thinking 强度。不支持的 provider 会忽略。

报告中会包含预算估算，以及 provider 返回的实际 usage/cost。

## 隐私模式

默认情况下，panel 中每个模型都会收到完整审查内容。对于敏感方案，可以用 `privacy_policy` 开启 strict 模式：可信模型看全文，对抗模型只看脱敏摘要，最后再由可信模型补充 evidence 评估。

```json
{
  "privacy_policy": {
    "policy": "strict",
    "trusted": {"endpoint": "trusted_gateway", "model": "trusted-model", "label": "trusted"},
    "min_coverage": 0.5
  }
}
```

strict privacy 更适合 plan review。code review 在脱敏后可能损失太多语义。

## 城区配送三臂评测

配送沙盒用确定性路网测试运动控制卸载，主脑保留 `pickup`、`deliver` 和完成判断：

```powershell
brain-region --config brain_region_config.json --env-file .env sandbox delivery-eval `
  --main-brain buzz_anthropic/claude-sonnet-5 --sizes 9 --seeds 0,1 --repeats 2
```

三条匹配评测臂分别是 `main_only`、`navigation_interface` 和 `navigation_region`。界面对照臂与真实脑区
接收相同公开观察、提示/工具契约和交互事件唤醒，但永远不产生或执行移动。报告分别计算
`navigation_interface - main_only`（界面暴露效应）与 `navigation_region - navigation_interface`
（grounded 执行策略增量），并保留端到端比较。成本截断产生的不完整三元组只作为 orphan 诊断保留，
不进入配对统计。
`--max-cost-usd` 在模型调用边界检查；provider 只会在请求结束后返回真实用量，因此最后一个在途请求可能
轻微越界，报告会显式给出 `budget_overrun_usd`，不把它描述成无法保证的硬上限。

## Review Memory

用 `mark_finding` 标记 finding 是否有用：

```text
mark_finding(finding_id="gpt-4o-3", decision="accepted", params_hash="...")
```

合法 decision 是 `accepted`、`rejected`、`partial`。反馈会写入本地 SQLite review 数据库，并在后续按 `(model, dimension)` 校准模型可信度。

## 输出

报告包含：

- `consensus`：所有模型都同意的 finding。
- `majority`：多个模型支持的 finding。
- `individual`：单模型 finding。
- `failed_models`：隔离的模型失败。
- `budget`、`usage`、`risk`、`context_compression` 等元信息。

SARIF 输出可以上传到 GitHub Code Scanning，也可以被 IDE 消费。

## 项目结构

```text
brainregion/
  server.py              # MCP server 入口
  cli.py                 # brain-region CLI
  core/                  # pipeline、stage、schema、report model
  adapters/              # generic / 可选领域 adapter
  providers/             # LLM backend
  knowledge/             # 检索实现
  privacy/               # 隐私策略
  output/                # 输出渲染
tests/                   # pytest 覆盖
docs/                    # 聚焦文档
```

## 安全说明

- 不要提交 `.env`、`.env.local`、API key、生成数据库或本地 `brain_region_config.json`。
- 旧的 `design_review_config.json` 仍然兼容，但也不应提交。
- 配置中优先使用 `api_key_env`，避免明文 `api_key`。
- `brain_region_reviews.db` 等生成的 review 数据库属于本地数据，测试不应依赖它。旧的 `design_reviews.db` 存在时仍会读取。

## License

Apache-2.0
