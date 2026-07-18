# Endpoint Configuration

[English](endpoint_config.md) | [简体中文](endpoint_config.zh-CN.md)

BrainRegion (`brain-region-mcp`) supports official LiteLLM model strings and custom gateway endpoints.

Use one endpoint per wire protocol. If the same gateway exposes both OpenAI-compatible and Anthropic-compatible APIs,
split them into separate endpoint IDs.

## OpenAI-Compatible Gateways

Use this for `/v1/chat/completions` APIs with `Authorization: Bearer ...`.

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

## Responses API Gateways

Some OpenAI-compatible gateways expose GPT/Codex models only through
`/v1/responses`. Set `api_mode` explicitly for those endpoint IDs:

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

`api_mode` defaults to `chat_completions`; the only other supported value is
`responses`. Responses calls preserve the message history, request JSON output,
set `store=false`, and expose their transport mode through `list_model_routes`
and model-call telemetry. Keep separate endpoint IDs when one gateway requires
different wire APIs for different model groups.

## Anthropic-Compatible Gateways

Use this for `/v1/messages` APIs with `x-api-key` and `anthropic-version`.

For Anthropic-compatible endpoints, set `base_url` to the site root. LiteLLM appends `/v1/messages`.

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

## Panel Shortcuts

`panel` accepts:

- `"endpoints"`: expand every model declared under every endpoint.
- `"endpoint_id"`: expand every model under one endpoint.
- `"endpoint_id/model"`: run one model through one endpoint.
- `"gpt-4o"` or other LiteLLM strings: run directly through the official provider environment variables.

Bare model names and endpoint references are different routes. `"claude-opus-4-8"` is treated as an official LiteLLM
Anthropic model and usually needs `ANTHROPIC_API_KEY`; `"modelbridge_anthropic/claude-opus-4-8"` routes through the
configured `modelbridge_anthropic` endpoint and uses its `MODEBRIDGE_API_KEY`. If the same model name exists under a
gateway, include the endpoint prefix when you want the gateway route.

Use `list_model_routes` to inspect how the current config will resolve a panel:

```python
list_model_routes(panel=[
    "claude-opus-4-8",
    "modelbridge_anthropic/claude-opus-4-8",
])
```

The tool only returns route metadata, endpoint declarations, model lists, and whether a key is present. It does not call
models or return API key values.

## Model Profiles

Endpoint `models` can be plain strings or objects with optional profile metadata:

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

You can also keep endpoint `models` as strings and place profiles in the top-level `model_profiles` map. Keys may be a
bare model name or an endpoint ref:

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

Profiles are descriptive preflight metadata for humans and schedulers. `suggest_panel` can rank configured routes from
these scores and tags, returning a `selected_panel` without calling models or automatically executing downstream tools.
`context_window_tokens` is optional and must be an explicitly verified positive integer. BrainRegion does not infer it
from a model name; when absent, context-pressure telemetry reports model capacity as unknown.

An explicit low/high synthetic recall probe can use the resolved capacity without copying it into another table:

```powershell
brain-region sandbox context-pressure-eval --main-brain modelbridge_anthropic/claude-opus-4-8
```

This command calls the selected model and may incur cost. Its bounded defaults use one middle-position needle and a
planned input-token guard; increase positions, ratios, or token caps only for an intentional long-context pilot.

Before interpreting a low/high result, check request-order repeatability with an identical-prompt control. This control
does not require a known model capacity:

```powershell
brain-region sandbox context-stability-control --main-brain modelbridge_anthropic/claude-opus-4-8
```

A passing stability control means repeated identical requests produced consistent correctness, parse/error state, and
input-token accounting. It does not prove long-context quality. When a config file lives outside the standard lookup
paths, set `BRAIN_REGION_CONFIG` for the CLI process explicitly.

To test memory selection rather than raw context length, run the matched semantic-interference A/B:

```powershell
brain-region sandbox context-interference-eval --main-brain modelbridge_anthropic/claude-opus-4-8
```

The clean and interference arms have matched target lengths and alternate execution order. The interference arm adds
stale, unverified, and similar-case memories. Reports retain only correctness, evidence-selection, usage, latency, and
cost metrics; generated memory contents and model responses are not persisted.

## Common Failures

- `Empty or invalid response` with an HTML page usually means an OpenAI-compatible `base_url` is missing `/v1`.
- `Invalid URL (POST /v1/v1/messages)` means an Anthropic-compatible `base_url` includes `/v1`; use the site root instead.
- `No permission to access auto group` comes from the gateway account/key permissions. The request reached the gateway, but the key cannot access that model/group.
- Missing `ANTHROPIC_API_KEY` often means a bare `claude-*` model name was used instead of a `modelbridge_anthropic/...`
  endpoint reference.
