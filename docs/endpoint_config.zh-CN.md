# Endpoint 配置

[English](endpoint_config.md) | [简体中文](endpoint_config.zh-CN.md)

BrainRegion（`brain-region-mcp`）同时支持官方 LiteLLM 模型字符串和自定义中转站 endpoint。

建议一个 endpoint 只对应一种网络协议。如果同一个中转站同时提供 OpenAI 兼容和 Anthropic 兼容 API，
请拆成不同的 endpoint ID。

## OpenAI 兼容中转站

适用于 `/v1/chat/completions`，认证方式通常是 `Authorization: Bearer ...`。

```json
{
  "endpoints": {
    "modelbridge_openai": {
      "provider": "openai",
      "base_url": "https://www.modelbridge.cloud/v1",
      "api_key_env": "MODEBRIDGE_API_KEY",
      "models": ["gpt-5.5", "gpt-5.4-mini"]
    }
  },
  "panel": ["modelbridge_openai/gpt-5.5"]
}
```

## Responses API 中转站

有些 OpenAI 兼容网关只允许 GPT/Codex 模型通过 `/v1/responses` 调用。应为这类 endpoint 显式设置 `api_mode`：

```json
{
  "endpoints": {
    "responses_relay": {
      "provider": "openai",
      "base_url": "https://relay.example/v1",
      "api_key_env": "RELAY_KEY",
      "api_mode": "responses",
      "models": ["gpt-5.5"]
    }
  }
}
```

`api_mode` 默认为 `chat_completions`，另一个可选值是 `responses`。Responses 调用会保留消息历史、请求 JSON 输出、显式设置 `store=false`，并在 `list_model_routes` 和模型调用遥测中显示传输模式。如果同一网关的不同模型组要求不同 API，应拆成不同 endpoint ID。

## Anthropic 兼容中转站

适用于 `/v1/messages`，认证方式通常是 `x-api-key` 和 `anthropic-version`。

Anthropic 兼容 endpoint 的 `base_url` 应写站点根地址。LiteLLM 会自动拼接 `/v1/messages`。

```json
{
  "endpoints": {
    "modelbridge_anthropic": {
      "provider": "anthropic",
      "base_url": "https://www.modelbridge.cloud",
      "api_key_env": "MODEBRIDGE_API_KEY",
      "models": ["claude-haiku-4-5", "claude-opus-4-8"]
    }
  },
  "panel": ["modelbridge_anthropic/claude-opus-4-8"]
}
```

## Panel 写法

`panel` 支持：

- `"endpoints"`：展开所有 endpoint 下声明的所有模型。
- `"endpoint_id"`：展开某个 endpoint 下声明的所有模型。
- `"endpoint_id/model"`：通过某个 endpoint 调用单个模型。
- `"gpt-4o"` 或其它 LiteLLM 字符串：直接走官方 provider 环境变量。

注意：裸模型名和 endpoint 引用是两条不同路由。`"claude-opus-4-8"` 会被当作官方 LiteLLM Anthropic 模型，通常需要
`ANTHROPIC_API_KEY`；`"modelbridge_anthropic/claude-opus-4-8"` 才会走 `modelbridge_anthropic` 这个 endpoint，
使用它声明的 `MODEBRIDGE_API_KEY`。如果同一个模型名同时存在于官方 provider 和中转站里，务必写 endpoint 前缀。

可以用 `list_model_routes` 检查当前配置会怎样解析：

```python
list_model_routes(panel=[
    "claude-opus-4-8",
    "modelbridge_anthropic/claude-opus-4-8",
])
```

它只返回路由、endpoint、模型列表和 key 是否存在，不会调用模型，也不会返回 API key 明文。

## 模型 Profile

`endpoints.<id>.models` 既可以继续写字符串，也可以写带 profile 的对象：

```jsonc
{
  "endpoints": {
    "modelbridge_openai": {
      "provider": "openai",
      "base_url": "https://www.modelbridge.cloud/v1",
      "api_key_env": "MODEBRIDGE_API_KEY",
      "models": [
        {
          "id": "gpt-5.4-mini",
          "activation_role": "sleep",
          "tier": "economy",
          "cost": "low",
          "latency": "fast",
          "tags": ["cheap", "fast"],
          "quality_score": 0.65,
          "cost_score": 0.9,
          "speed_score": 0.85,
          "context_window_tokens": 128000
        }
      ]
    }
  }
}
```

也可以保持 `models` 为字符串，把 profile 放在顶层 `model_profiles`。key 可以是裸模型名，也可以是 endpoint 引用：

```jsonc
{
  "model_profiles": {
    "modelbridge_anthropic/claude-opus-4-8": {
      "activation_role": "awake",
      "tier": "flagship",
      "cost": "high",
      "tags": ["deep_reasoning", "architecture"],
      "quality_score": 0.98,
      "cost_score": 0.2,
      "context_window_tokens": 200000
    }
  }
}
```

Profile 是给人和 scheduler 看的 preflight 元数据。`suggest_panel` 可以根据这些评分和标签对已配置的模型路由排序，
返回 `selected_panel`，但不会调用模型，也不会自动执行后续工具。
`context_window_tokens` 是可选字段，必须填写经过核实的正整数。BrainRegion 不会根据模型名称猜测容量；未配置时，
上下文压力遥测会明确将模型窗口容量标记为未知。

可以用下面的显式命令直接读取 resolved profile 的容量，执行低/高上下文合成召回实验，无需再维护一份容量表：

```powershell
brain-region sandbox context-pressure-eval --main-brain modelbridge_anthropic/claude-opus-4-8
```

该命令会真实调用选定模型并产生费用。默认只测试 middle 位置，并带有计划输入 token 护栏；只有在有意进行长上下文
pilot 时才应提高位置数量、负载比例或 token 上限。

解释 low/high 结果前，应先用完全相同的 prompt 检查请求顺序重复性。这个控制实验不要求已知模型窗口容量：

```powershell
brain-region sandbox context-stability-control --main-brain modelbridge_anthropic/claude-opus-4-8
```

稳定性控制通过，只表示相同请求的正确性、解析/错误状态和 input-token 计数一致，并不证明模型具备长上下文能力。
如果配置文件不在标准查找路径中，需要为 CLI 进程显式设置 `BRAIN_REGION_CONFIG`。

要测试模型选择记忆的能力，而不只是测试上下文长度，可以运行等负载的语义干扰 A/B：

```powershell
brain-region sandbox context-interference-eval --main-brain modelbridge_anthropic/claude-opus-4-8
```

`clean_memory` 与 `interference_memory` 使用相同的目标长度，并交替执行顺序。干扰臂会加入过期记忆、未经验证的假设和相似任务记忆。报告只保存答案与证据选择是否正确、token、延迟和成本，不保存生成的记忆内容或模型响应。

## 常见错误

- `Empty or invalid response` 并返回 HTML 页面：OpenAI 兼容 endpoint 的 `base_url` 通常缺少 `/v1`。
- `Invalid URL (POST /v1/v1/messages)`：Anthropic 兼容 endpoint 的 `base_url` 多写了 `/v1`，应改成站点根地址。
- `No permission to access auto group`：请求已经到达中转站，但当前 key 没有访问该模型/分组的权限。
- 提示缺少 `ANTHROPIC_API_KEY`：通常说明你传了裸 `claude-*` 模型名，实际没有走 `modelbridge_anthropic/...` endpoint。
