# 模型指纹探针(model_fingerprint_check)

检测端点是否被**偷换模型**(Opus 实际跑 Haiku、GPT 实际跑开源套壳)、**注水计量**、
**注入隐藏 prompt**,或上游**静默换 snapshot/量化**导致的"降智"。

## 用法

```
# 首次:建立基线(usage 1 次请求;behavior 约 200 次小请求;capability 30 项)
model_fingerprint_check(model="zhipu_glm/glm-5.2", mode="baseline", checks=["usage", "behavior", "capability"], seed=7)

# 之后任意时刻:对比(怀疑降智/换模型时)
model_fingerprint_check(model="zhipu_glm/glm-5.2", mode="compare", checks=["usage", "behavior", "capability"], seed=7)

# 漂移历史汇总(只读)
inspect(view="model_health")
```

`model` 支持 `endpoint_id/model` 短引用(与 panel 同语义)。基线存
`.brain-region/probe/probe.db`(SQLite,重建基线会覆盖旧基线但历史保留)。

## 三类探针

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

## 真实验证(2026-08-14,4 模型 8 端点次实测)

行为指纹 JSD 矩阵(10 样本/格快速档;对角=自比):

| JSD | glm-5.2 | Qwen3.5-9B | deepseek-v4-flash | haiku-4.5 |
|-----|---------|-----------|-------------------|-----------|
| glm-5.2 | **0.133 match** | 1.000 | 1.000 | 1.000 |
| Qwen3.5-9B | 1.000 | 0.414 mismatch* | 1.000 | 1.000 |
| deepseek-v4-flash | 1.000 | 1.000 | **0.207 match**(25 样本复测) | 0.764 |
| haiku-4.5 | 1.000 | 1.000 | 0.764 | **0.082 match** |

**跨模型 6/6 全部 mismatch(0.74–1.00),区分度远超阈值**;自比 match 于 glm/haiku(10 样本
快速档)与 deepseek(25 样本复测,快速档 0.31 压线是采样噪声,25 样本降到 0.21)。有趣的指纹
特征:GLM-5.2 答 1-100 随机数 100% 是 42;Haiku 80-90% 是 42;DeepSeek 高熵分散(47/87/97)。

**\*Qwen3.5-9B@SiliconFlow 自比 mismatch 是端点真实不稳定,不是工具误报**:逐条诊断发现该
端点分钟级出现退化生成——tab 填充烧满 token 上限、空对象 `{  }`、胡言乱语 JSON
(`{"error": true, "message": "Duty-free policy change..."}`),split-half 轮换旗如实触发
(同一次采集前后两半分布不交)。与公开研究"同名模型不同 provider 10/34 对漂移超限"同类。
处置:该端点不适合做指纹基线;若已建基线后出现此模式,优先怀疑 provider 侧变更/多副本轮换。

实测发现与对策(已产品化):

- **GLM/Qwen 对短约束问题爱答 JSON 壳**(`{"answer":42}`),且小 max_tokens 会截断
  JSON 导致答案碎片化 → normalize_answer 已支持剥 fence/JSON 壳/截断 JSON 正则兜底,
  退化壳(`{`/`{  }`/tab 填充)归并为 `<unparsed>` 单键,行为探针 max_tokens=32。
- **GLM/Qwen 思考默认开会烧光小 max_tokens**(空答案)→ `_effort_kwargs` 已补
  GLM(`thinking.type`)/Qwen(`enable_thinking`)映射,`thinking=False` 全家可用。
- **usage 计数逐次完全稳定**(584/584、575/575、509/509、713/713),但 glm↔qwen 这类
  同量级 tokenizer 差异仅 ~1.6%(usage 档内判 match)→ **usage 指纹主战场是注水/隐藏
  prompt(增量 +几百 token)和跨 tokenizer 家族偷换(haiku 713 vs dsk 509 = +40%),
  同量级互换靠 behavior 档抓**。
- deepseek 官方端点会真实返回 `system_fingerprint`,可被动跟踪 snapshot。
- 快速档(10 样本/格)下高熵模型自比会压线 suspicious(采样噪声),**正式对比请用默认
  25 样本/格**。

### capability 指纹(30 项参数化能力基准)

抓"指纹一致但能力下降"(量化/snapshot 降级/sampling 被改)。5 类目:math 8 / 指令遵循 8 /
逻辑 6 / 代码输出预测 6 / 长上下文 NIAH 2。**参数化模板 + 本地算出期望答案 → 全确定性
判分**(无 judge、判分零成本);数值槽随 seed 随机化(防中转缓存探针)。

判定按类目通过率聚合(单项二元噪声大):整体通过率较基线下降 **≥20pp 判 degraded**、
≥10pp 判 suspicious;单类目下降 ≥20pp 打 `category_drop` 旗。实测注意:GLM 对短约束
问题爱答 JSON 壳,判分已统一剥壳(壳内答案真错仍判负,不洗白)。

真实验证(2026-08-14,glm-5.2 基线 0.90 / math 1.0 / code 1.0 / niah 1.0):自比 match
(-6.7pp,温度 0 抖动带内);glm-4.5-air 冒充 glm-5.2 → overall match 但 code_output
类目 -33.3pp 旗正确触发。**诚实边界:30 项轻量题库抓粗粒度降级(量化版/mini 偷换/
snapshot 降级),相邻档位(air vs 旗舰)的 overall 区分需要更难题库**;跨模型偷换由
behavior 指纹负责(实测 6/6 mismatch),capability 档的独特价值是"同模型变弱"场景。

### glm-5.3 上新实测(2026-08-14)

glm-5.3 是**始终思考**模型(open paas/v4 端点拒绝 thinking=disabled,也拒绝 type=low;
thinking 字符串档只被 Anthropic 兼容端点接受)。实测路径:`zhipu/glm-5.3`
(anthropic 端点)+ thinking 自适应。三个探针发现:

- **behavior**:5.3 自比 JSD 0.11 match;**5.3 vs 5.2 = 0.42 mismatch——新版本被清楚
  识别为不同分布**(随机数偏好也变了:5.2 是 42 压倒性,5.3 是 47 主导)。
- **capability**:思考烧光小 token 上限 → 空答案伪 degraded(40pp 假信号)。探针的
  **截断救援**(空答案用 1024 上限重测 + `many_truncation_rescues` 亮旗)后:5.3 真实
  0.9667,vs 5.2 的 0.90 为 match(-6.7pp)。这个案例是"工具报 degraded 先查截断旗,
  别直接下'模型变笨'结论"的活教材。
- usage:5.3@anthropic 端点 537 vs 5.2@openai 端点 584——跨端点协议不同,不可直接比。

### logprob 档(2x10 次 1-token 请求,LT,arXiv:2512.03816)

对固定短 prompt 只生成 1 个 token,取 top-20 logprob 当分布采样;两侧各 N=10 次,算逐
token 平均 logprob 的平均绝对距离 S,置换检验(B=1000,固定种子可复现)出 p 值。
**最灵敏**:论文能检出一次微调、2^-10 级剪枝;每次测试约 50 token。判定 p<=0.01
mismatch / p<=0.05 suspicious。

支持面(2026-08-14 实测):deepseek ✓;智谱两端点 ✗(静默不返回);SiliconFlow ✗
(litellm 对该 provider 映射会把 `logprobs` 参数剥掉,vLLM 报 400,extra_body 兜底也拦
不住)。不支持的端点返回 `unknown`+hint,不算失败。始终思考模型与 1-token 方法论不
兼容(思考烧光预算),LT 跳过。

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

## 后续规划

- 被动漂移监控:把 `served_model`/usage 基线接进 runtime 事件流,正常流量零成本积累
- LT 支持面随 litellm 版本演进复测(当前 deepseek 可用;siliconflow 被 litellm 剥参)
