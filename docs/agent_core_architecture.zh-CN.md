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
├─ Unity Runtime Peer 会话内核             当前纵切
├─ Windows 命名管道 + HMAC 配对            已实现（WSS 后续）
└─ MCP Client / Python 子进程监管          已实现
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
| `scene/peers/list` | 返回已认证且当前仍连接的 Unity Player 快照 |
| `scene/peer/call` | 按主体、白名单方法和 capability 转发 Runtime Scene RPC 请求 |
| `daemon/shutdown` | 关闭 MCP transport，限时等待并回收 Python 子进程 |

完整消息契约见 [`schemas/agent-core/v1/control-message.schema.json`](../schemas/agent-core/v1/control-message.schema.json)。它是控制协议的单一事实来源，并在构建时嵌入 `brainregiond` 二进制；分发后的 daemon 不依赖源码仓库中的 schema 路径。修改契约必须同步更新运行时契约测试并重新构建二进制。

`mcp/tools/call` 超时后，daemon 会向上游发送 MCP cancellation，再向客户端返回超时错误。Cancellation 是尽力而为的协议通知：Python tool 可能已经产生部分副作用，其最终结果仍是 unknown。客户端必须重新读取相关状态或请求人工确认，不得自动重试该调用。

Runtime Scene RPC 已实现与具体传输无关的 peer registry 和双向 JSONL 会话。Windows 上还可显式启用当前用户命名管道；它先完成 challenge/HMAC 配对，再构造 `ScenePeerAuth` 交给会话层，不会直接相信 Player 自报的 `pairingProof`。同一主体的新连接会获得更大的 `connectionEpoch` 并立即替换旧连接，旧 pending 请求失败。请求具有 1 MiB 帧限制、128 个有界排队/等待上限、deadline 和 response correlation；迟到响应只会被丢弃，不触发自动重试。`scene/changed` 用于推进 daemon 观察到的 revision 和事件流。

该路径已用独立 Unity `6000.0.59f2` Windows x64 IL2CPP Development Player 做真实进程级验证：Player 完成 challenge/HMAC 注册后，Rust 调用 `runtime/info`、`scene/hierarchy` 和显式白名单属性；验证 preview 零副作用、apply 推进 revision、旧 revision 拒绝、同主体新 connection epoch 下精确幂等 replay、Undo 恢复及 Undo 后拒绝旧成功回放。真实 VR 项目仍保持未修改；其 Unity `6000.3.20f1` Player 联调需要先安装完全匹配的 Windows IL2CPP 模块。

`scene/peer/call` 只接受 Scene RPC v1 白名单方法，并同时检查 Player 宣告支持与已认证策略授予的 capability。含 `spawn` 的 preview 还必须具有独立的 `scene.spawn`。超时返回 outcome unknown、`retryable=false`；写操作调用方必须凭 `clientMutationId` 查询或重放完全相同的幂等请求，不能生成新 mutation ID 自动重试。

v1 依次执行普通请求，同时最多有界缓存 32 个后续请求。`daemon/shutdown` 和操作系统终止信号可以抢占正在等待的 tool call；被抢占调用同样返回 outcome unknown 且不可自动重试。stdin 由独立的有界转发线程读取，因此父进程保持输入管道打开时也不会阻塞 daemon 完成关闭。若 MCP transport 已断开，`daemon/info` 报告 `degraded`，缓存的 `mcp/tools/list` 不再作为成功结果返回。

## 运行与验证

```powershell
rustup toolchain install 1.88.0 --profile minimal
cargo +1.88.0 check --workspace --all-targets --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --locked -p brainregiond --test real_python_mcp -- --ignored --test-threads=1
cargo test --locked -p brainregiond --test unity_player_windows -- --ignored --test-threads=1
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

## Windows Runtime 命名管道

默认不创建管道。仅当设置 `BRAINREGIOND_SCENE_PIPE_NAME` 时启用，名称只能是 1–128 个 ASCII 字母、数字、点、下划线或连字符，daemon 固定展开为 `\\.\pipe\<name>`，不能配置远程 UNC 路径。配置项：

- `BRAINREGIOND_SCENE_PAIRING_SECRET`：必需，只从环境读取，32–4096 UTF-8 字节，不提供命令行参数；
- `BRAINREGIOND_SCENE_PRINCIPAL_ID`：默认 `unity-local`；
- `BRAINREGIOND_SCENE_CAPABILITIES_JSON`：默认 `["scene.read"]`，日志、写入、创建和 Undo 必须显式授权；
- `BRAINREGIOND_SCENE_MAX_CONNECTIONS`：1–32，默认 4；
- `BRAINREGIOND_SCENE_AUTH_TIMEOUT_MS`：默认 10000。

每个 pipe instance 使用保护型 DACL `O:<current-user-sid>D:P(A;;GA;;;<current-user-sid>)`，句柄不可继承，设置 `PIPE_REJECT_REMOTE_CLIENTS`；首实例使用 `FILE_FLAG_FIRST_PIPE_INSTANCE`，避免 daemon 启动时被同名管道抢占。DACL 是本机 OS 边界，challenge 认证是应用边界，两者都通过后才注册 peer。

daemon 先发送 `runtime/challenge`，其中包含 32 字节随机 nonce、过期时间、主体、算法和该连接实际获得的 capability。Player 只能据此构造 `AuthenticatedPeerContext`，不能使用自己声明或本地猜测的权限。Player 在时限内把 `pairingProof` 设为 `hmac-sha256.<base64url-no-padding>`。HMAC 输入不包含 proof 自身；每个普通字段编码为 `UTF8字节长度 + ':' + UTF8字节 + '\n'`，顺序为：

```text
challenge.protocolVersion
challenge.algorithm
challenge.nonce
challenge.expiresUnixMs（十进制）
challenge.principalId
challenge.grantedCapabilities 数量（十进制）
challenge.grantedCapabilities（按协议字符串升序，逐项编码）
registration.protocolVersion
registration.instanceId
registration.sessionId
registration.buildId
registration.unityVersion
registration.platform
registration.product
registration.sceneId
registration.sceneRevision（十进制）
registration.status
registration.error 存在标记（单字节 0/1；1 后再编码 error 字段）
registration.capabilities 数量（十进制）
registration.capabilities（按协议字符串升序，逐项编码）
```

HMAC 使用 SHA-256。nonce 每连接重新生成，grant 也包含在 proof 中，因此旧 registration/proof 无法跨连接重放，客户端也不能扩大服务器授予的权限。当前阶段使用预共享高熵密钥；后续 VR 配对 UI 应负责安全分发/轮换密钥，不能把短 PIN 直接当 HMAC key。

## 安全边界

- 当前可执行入口只提供本机 stdio 控制面；Runtime 传输只有显式启用的本机 Windows 命名管道，不监听 TCP/HTTP/WSS 端口。
- 不把模型密钥、SSH 凭据或 OAuth token 传给 VR 端。
- `mcp/tools/call` 仍受 Python BrainRegion 的 workspace root、SHA 写入保护和命令白名单约束。
- 当前命名管道已有当前用户 DACL、challenge/HMAC、连接级 capability 与重放保护；写权限正式面向用户开放前仍需增加审批与持久审计。未来 WSS 还必须补设备身份、证书固定和 capability token。
- Windows 产品化需给 Python worker 增加 Job Object 的 kill-on-close，避免 daemon 崩溃留下孤儿进程。
- `brainregiond` 首期作为当前用户登录会话中的无窗口进程运行，不注册为 Session 0 系统服务。

## 暂不实现

- 不在 Rust 中重写模型循环、记忆、review 或 consult。
- 不直接把 MCP 或任意 shell 暴露给头显。
- 不承诺进程崩溃后的自动重启和跨重启会话恢复。
- 不实现 Unity/Unreal 任意代码 `eval`。

## 后续里程碑

1. Unity Runtime package 的 opt-in Windows 命名管道、HMAC proof、有界 JSONL、重连和 connection epoch 已通过独立 Windows IL2CPP Player 的真实读写事务联调。
2. 增加 adapter 异常/回滚失败与响应前断线测试，补齐写入 outcome unknown 的查明闭环。
3. 增加有界事件日志、审批状态和指数退避重启。
4. 增加 ConPTY 会话和 DAP client。
5. 增加源码 patch、编译状态、程序集 reload 后重连和验证闭环。
6. 最后增加带证书固定和配对的出站 WSS，支持独立头显；运行态只允许白名单场景操作和预构建资源，不依赖任意 IL2CPP 代码热注入。
