# 模型指纹探针(model_fingerprint_check)

检测端点是否被**偷换模型**(Opus 实际跑 Haiku、GPT 实际跑开源套壳)、**注水计量**、
**注入隐藏 prompt**,或上游**静默换 snapshot/量化**导致的"降智"。

## 用法

```
# 首次:建立基线(先 usage 后 behavior;behavior 约 200 次小请求,几分之一美分)
model_fingerprint_check(model="zhipu/glm-5.2", mode="baseline", checks=["usage", "behavior"], seed=7)

# 之后任意时刻:对比(怀疑降智/换模型时)
model_fingerprint_check(model="zhipu/glm-5.2", mode="compare", checks=["usage", "behavior"], seed=7)
```

`model` 支持 `endpoint_id/model` 短引用(与 panel 同语义)。基线存
`.brain-region/probe/probe.db`(SQLite,重建基线会覆盖旧基线但历史保留)。

## 两类探针

### usage 指纹(1 次请求)

固定 canonical prompt 的 `usage.prompt_tokens` 由 tokenizer 决定:同 tokenizer 下确定性
一致,o200k / cl100k / GLM / Qwen 对同一文本计数差异通常 >5%。计数漂移 = 后端 tokenizer
变了 = 换模型。顺带检查:

| 信号 | 含义 |
|------|------|
| `suspected_hidden_prompt` | prompt_tokens 凭空多出大量 token(中转注入系统提示,实测案例:+1447) |
| `completion_watering` | 只让答 OK 却烧几百 completion token(注水计量) |
| `served_model_changed` | 响应回显的 model 字段变了(可被中转伪造,但变了必有事) |
| `system_fingerprint_changed` | OpenAI snapshot 更新提示 |

### behavior 指纹(随机数分布,arXiv:2607.10252)

LLM 说不出真随机数:"1-100 随机数/颜色/硬币"这类单 token 问题的答案分布,每个模型有
稳定且独特的偏置(42/7/37 现象)。默认 8 格 × 25 采样(temperature=1.0, max_tokens=16,
thinking 关闭),与基线算 Jensen-Shannon 散度:

| mean JSD | 判定 | 依据 |
|----------|------|------|
| ≤ 0.25 | match | 同人自比 ≈ 0.14,跨 provider 同名模型 ≈ 0.23 |
| 0.25 – 0.35 | suspicious | |
| > 0.35 | mismatch | 跨模型 ≈ 0.46 |

附加信号:`single_cell_divergence`(均值被未漂移格子稀释,单格 JSD>0.6 至少 suspicious,
对应"只换部分行为"的量化/降智)、`backend_rotation_or_cache_collapse`(split-half 自一致
性,同一次采集内部前后两半分布不交 → 后端轮换或缓存塌缩)。

## 判定解读(重要)

**偏差 ≠ 欺诈**。以下都会触发漂移:量化版本、官方 snapshot 静默更新、sampling 参数变化、
provider 侧 infra 差异(OpenRouter 上 34 对同名模型不同 provider,10 对漂移超限)。输出是
**信号强度**,不是欺诈判定。建议:match 之外的结果先 `mode="baseline"` 重建基线再复测一次,
两次都漂移才值得找供应商对质。

## 局限(猫鼠游戏)

恶意中转可以识别探针请求、缓存探针响应、按 prompt 指纹路由到真模型。缓解:探针 phrasing
随机采样、固定 `seed` 保证可复现、定期重测;usage 指纹被动存在于每次请求,无法被针对性
放行。不承诺防住针对性对抗。

## 被动信号

`ModelResponse` 新增 `served_model` / `system_fingerprint` 字段(providers/litellm.py 捕获),
所有走 backend 的调用都可读,当前仅在探针内消费;后续可接入 runtime 事件做被动漂移监控。

## 方法出处

- 行为指纹与 JSD 阈值:arXiv:2607.10252(tosea.ai 实测落地,含 OpenRouter 套壳实锤)
- usage/tokenizer 计数法:PureLLM、Veridrop、relay-radar 等中转验真工具的通行做法
- logprob 置换检验(未实现,更灵敏但需端点支持 logprobs):arXiv:2512.03816
  (github.com/timothee-chauvin/track-llm-apis)
- 模型识别学术先例:LLMmap(USENIX Security '25)

## 后续规划(phase 3+)

- 能力基准探针(数学/代码/指令遵循/NIAH mini-benchmark,抓"指纹一致但能力下降")
- inspect 增加 model_health view 汇总漂移历史
- logprob LT 可选档
