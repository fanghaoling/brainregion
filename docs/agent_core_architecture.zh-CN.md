# BrainRegion Rust Agent Core 架构决策

状态：已接受，首个纵切已实现

日期：2026-08-15
协议：`brainregion.control.v1`

## 决策

新增 Rust 用户态常驻进程 `brainregiond`，作为桌面端、VR 客户端与执行环境之间的控制面。现有 Python BrainRegion MCP 保持为领域工具 worker，不整体重写为 Rust。

```text
桌面 / VR 客户端
        │  brainregion.control.v1（当前为 stdio JSONL；后续为 IPC/WSS）
        ▼
brainregiond（Rust）
├─ 会话、事件、审批、审计                 后续
├─ PTY / ConPTY、DAP、LSP                 后续
├─ Unity / Unreal Scene RPC               后续
└─ MCP Client / Python 子进程监管          当前纵切
        │  MCP 2025-11-25 stdio
        ▼
brainregion.server（Python FastMCP）
```

Rust 负责生命周期、传输、并发和特权执行边界；Python 继续负责模型供应商、review/plan/consult/memory、认知工作流和现有 MCP tools。只有出现可测量的性能、分发或安全需求时，才逐模块迁移 Python 能力。

## 当前纵切

`brainregiond` 启动直接 Python 子进程，不经 shell 拼接命令，也不默认经过 `uv run`。在仓库开发环境中会自动发现：

```text
.venv/Scripts/python.exe -m brainregion.server   # Windows
.venv/bin/python -m brainregion.server           # Unix
```

若没有本地虚拟环境，则尝试 PATH 中的 `brain-region-mcp`。产品打包时必须用 `BRAINREGIOND_MCP_PROGRAM` 和 `BRAINREGIOND_MCP_ARGS_JSON` 显式指向随应用分发的 Python worker。

Python 进程句柄由 `brainregiond` 自己的 MCP transport 持有，不依赖 SDK 的异步 Drop：初始化失败或超时、握手校验失败、断管和正常 shutdown 都会显式 kill/wait 并确认回收，`kill_on_drop` 只作为 panic/runtime 中止时的最后兜底。Windows 产品化仍应再用 Job Object 覆盖 Python 可能派生的整个进程树。

启动就绪条件全部满足后，才发送 `daemon/ready`：

1. Python 子进程已启动；
2. MCP initialize 成功协商 `2025-11-25`；
3. `tools/list` 成功并包含应用级 `ping`；
4. `tools/call(name="ping")` 返回 `ok=true`；
5. `name` 为 `brainregion`，项目版本符合语义版本形状。

初始化中的 `serverInfo.version` 是 Python MCP SDK 版本，不是 BrainRegion 项目版本。项目版本以应用级 `ping.version` 为准。

## 控制协议 v1

首版使用每行一个 JSON-RPC 2.0 消息。stdout 只输出协议消息，daemon 与 Python worker 日志均走 stderr。单条客户端输入帧上限为 1 MiB，超限会终止当前控制会话；单条 daemon 输出帧上限为 8 MiB，超限结果会被替换为相关联的 JSON-RPC error。

请求必须携带 string、integer 或 null 类型的 `id`，不接受小数 ID 或未声明字段。`params` 若出现必须是 object；无参数方法可以省略它，也可以传空对象 `{}`。`mcp/tools/call` 必须携带 `params`，且只允许 `name` 和可选的 `arguments`；省略或传 null 表示空参数，否则 `arguments` 必须是 object。

服务启动后先发送：

```json
{
  "jsonrpc": "2.0",
  "method": "daemon/ready",
  "params": {
    "protocolVersion": "brainregion.control.v1",
    "daemon": {
      "name": "brainregiond",
      "version": "0.2.0"
    },
    "status": "ready",
    "mcp": {
      "protocolVersion": "2025-11-25",
      "serverInfo": {
        "name": "brainregion",
        "version": "1.28.1"
      },
      "capabilities": {},
      "toolCount": 46,
      "brainregion": {
        "ok": true,
        "name": "brainregion",
        "legacy_name": "brain_region",
        "version": "0.2.0"
      }
    }
  }
}
```

支持的方法：

| 方法 | 行为 |
|---|---|
| `daemon/info` | 返回缓存的 daemon、MCP 和能力信息，不触发上游调用 |
| `daemon/health` | 实时调用 BrainRegion 应用级 `ping` |
| `mcp/tools/list` | 返回连接建立时缓存的工具清单 |
| `mcp/tools/call` | 调用现有 Python MCP tool；工具级错误保持为 MCP 结果 |
| `daemon/shutdown` | 关闭 MCP transport，限时等待并回收 Python 子进程 |

完整消息契约见 [`schemas/agent-core/v1/control-message.schema.json`](../schemas/agent-core/v1/control-message.schema.json)。它是控制协议的单一事实来源，并在构建时嵌入 `brainregiond` 二进制；分发后的 daemon 不依赖源码仓库中的 schema 路径。修改契约必须同步更新运行时契约测试并重新构建二进制。

`mcp/tools/call` 超时后，daemon 会向上游发送 MCP cancellation，再向客户端返回超时错误。Cancellation 是尽力而为的协议通知：Python tool 可能已经产生部分副作用，其最终结果仍是 unknown。客户端必须重新读取相关状态或请求人工确认，不得自动重试该调用。

v1 依次执行普通请求，同时最多有界缓存 32 个后续请求。`daemon/shutdown` 和操作系统终止信号可以抢占正在等待的 tool call；被抢占调用同样返回 outcome unknown 且不可自动重试。stdin 由独立的有界转发线程读取，因此父进程保持输入管道打开时也不会阻塞 daemon 完成关闭。若 MCP transport 已断开，`daemon/info` 报告 `degraded`，缓存的 `mcp/tools/list` 不再作为成功结果返回。

## 运行与验证

```powershell
rustup toolchain install 1.88.0 --profile minimal
cargo +1.88.0 check --workspace --all-targets --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --locked -p brainregiond --test real_python_mcp -- --ignored --test-threads=1
cargo run --locked -p brainregiond -- probe
cargo run --locked -p brainregiond -- serve
cargo run --locked -p brainregiond -- schema
cargo run --locked -p brainregiond -- scene-schema
```

`scene-schema` 输出独立的 Unity Player Runtime Scene RPC v1 契约；其运行时对象、安全和 IL2CPP 边界见 [`unity_runtime_scene_rpc.zh-CN.md`](unity_runtime_scene_rpc.zh-CN.md)。

显式指定 worker：

```powershell
cargo run --locked -p brainregiond -- probe `
  --mcp-program .venv\Scripts\python.exe `
  --mcp-arg -m `
  --mcp-arg brainregion.server
```

环境变量：

- `BRAINREGIOND_MCP_PROGRAM`
- `BRAINREGIOND_MCP_ARGS_JSON`：JSON 字符串数组，避免不安全的 shell 字符串解析
- `BRAINREGIOND_MCP_CWD`
- `BRAINREGIOND_MCP_PROTOCOL_VERSION`
- `BRAINREGIOND_STARTUP_TIMEOUT_MS`
- `BRAINREGIOND_HEALTH_TIMEOUT_MS`
- `BRAINREGIOND_REQUEST_TIMEOUT_MS`

## 安全边界

- v1 只提供本机 stdio，不监听网络端口。
- 不把模型密钥、SSH 凭据或 OAuth token 传给 VR 端。
- `mcp/tools/call` 仍受 Python BrainRegion 的 workspace root、SHA 写入保护和命令白名单约束。
- WSS、命名管道和 Scene RPC 上线前必须增加设备身份、capability token、审批、重放保护和审计日志。
- Windows 产品化需给 Python worker 增加 Job Object 的 kill-on-close，避免 daemon 崩溃留下孤儿进程。
- `brainregiond` 首期作为当前用户登录会话中的无窗口进程运行，不注册为 Session 0 系统服务。

## 暂不实现

- 不在 Rust 中重写模型循环、记忆、review 或 consult。
- 不直接把 MCP 或任意 shell 暴露给头显。
- 不承诺进程崩溃后的自动重启和跨重启会话恢复。
- 不实现 Unity/Unreal 任意代码 `eval`。

## 后续里程碑

1. 增加有界事件日志、请求取消、审批状态和指数退避重启。
2. 增加 Windows 命名管道及当前用户 DACL；随后再增加带配对的 WSS。
3. 增加 ConPTY 会话和 DAP client。
4. 实现 Unity Player Runtime Scene RPC：日志、层级查询、稳定对象 ID、属性预览/提交、Undo。
5. 增加源码 patch、编译状态、程序集 reload 后重连和验证闭环。
6. 最后支持独立头显；运行态只允许白名单场景操作和预构建资源，不依赖任意 IL2CPP 代码热注入。
