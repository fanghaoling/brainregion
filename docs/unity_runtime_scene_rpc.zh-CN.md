# Unity Player 运行时开放编辑与 Scene RPC v1

状态：Unity Runtime 核心、Rust Runtime Peer 会话、opt-in Windows 命名管道客户端、Prefab Catalog 自动生成、AOT-safe 属性 binding codegen、运行时对象生命周期同步和 `WorldDocument v1` 存档恢复已实现；独立 Windows High-stripping IL2CPP Player 的注册、读取、写事务、失败回滚、响应丢失查明、生成属性、Prefab 安全 staging/spawn、Undo、动态对象注册/销毁、additive scene load/unload、断线重连，以及 save/list/load preview/load 与缺失 Prefab 重建闭环已通过，尚未接入实际 VR 项目。

## 结论

打包后的 Unity 游戏可以支持类似开放式 VR 沙盒的场景编辑，但它不是把 Unity Editor 搬进游戏。正确模型是：

- 游戏把可编辑对象、Prefab、属性和动作显式建模；
- 游戏在主线程执行小型、版本化的 Runtime Scene RPC；
- `brainregiond` 持有 Agent 会话、权限、审批、审计和外部终端；
- VR 内 UI 只是一个客户端，不持有模型密钥或任意系统执行权限。

现有 VR 项目 `D:\Unity\My Project\VR Project` 的只读检查结果：

- Unity `6000.3.20f1`；
- Standalone 与 Android 当前都设置为 IL2CPP；
- 已安装 `com.anklebreaker.unity-mcp 2.39.5`，但其程序集和实现属于 `Editor`，不会进入 Player；
- 项目处于早期阶段，因此本阶段没有修改它。

IL2CPP 会把 IL 提前转换并编译为平台原生代码，不适合把“运行时编译、替换任意 C#”作为开放系统的基础。[Unity IL2CPP 概述](https://docs.unity3d.com/cn/6000.0/Manual/scripting-backends-il2cpp.html) 同时，反射访问还会受到 managed code stripping 影响，需要显式 preserve；本方案因此默认不用任意反射路径。[Unity Managed Code Stripping](https://docs.unity3d.com/Manual/ManagedCodeStripping.html)

## 已实现的边界

可移植 UPM package 位于：

```text
unity/Packages/com.brainregion.runtime-bridge/
```

它的 Player 能力只在 Runtime 程序集中实现，没有 `UnityEditor` 引用，也没有自动启动的端口；单独的 Editor 程序集只负责打包前校验，不进入 Player。当前能力包括：

- `RpcObjectIdentity`：协议稳定字符串 ID，不暴露 Unity `InstanceID`；
- `RuntimePrefabCatalog`：应用拥有的 Prefab 白名单；
- `IRpcPropertyAdapter`：项目代码显式声明属性描述、读取、校验和写入；
- `RuntimePrefabCatalogGenerator`：从 `BrainRegionRuntimePrefab` 标签自动生成 GUID 稳定的全局 Catalog，额外资产标签成为排序后的内容 tag；
- `RuntimePropertyBindingGenerator`：读取显式 attribute，在构建前生成直接成员访问的 adapter；反射只存在于 Editor，Player 不走反射路径；
- `RuntimeSceneController`：层级、检查、Prefab 创建、Transform/active/显式属性修改；
- 运行时生命周期同步：active 动态对象在 `OnEnable` 后由主线程统一注册，销毁时清理死引用；显式打开 `includeLoadedScenes` 后同步 additive scene，外部生命周期变更按批次只推进一次 revision；
- `scene/preview -> scene/apply`：预览 token 绑定 revision 和 mutation ID；
- 单调 `sceneRevision`、幂等 mutation receipt、失败回滚和运行时 Undo；
- `WorldDocument v1`：固定 slot、配额、SHA-256 完整性摘要、原子替换、兼容性校验、无副作用加载预览和幂等恢复；
- `SceneRpcDispatcher`：认证主体、连接 epoch、逐方法 capability、1 MiB 帧上限、有界队列、deadline、每帧数量/时间预算；
- `WindowsScenePipeTransport`：Player 主动连接当前用户命名管道，严格解析 challenge，生成跨语言一致的 HMAC proof，把服务器签名绑定的 grant 传给 Dispatcher，并在断线后以新 epoch 重连；
- `BoundedJsonLineReader` / `BoundedSceneWriterQueue`：严格 UTF-8、1 MiB 单帧限制和有界非阻塞响应队列；队列过载会断开连接而不是卡住 Unity 主线程；
- `RuntimeLogBuffer`：线程安全、容量受限的 Player 日志轮询。
- 构建校验器：拒绝场景对象 ID 缺失/重复、无效 Prefab Catalog 和 active 的 Catalog Prefab root；菜单可为已加载场景分配缺失 ID。

协议 schema 与跨语言示例位于：

```text
schemas/scene-rpc/v1/
```

Rust 侧已经嵌入同一 schema，并提供严格 DTO、Runtime peer registry、请求关联、deadline、capability 检查、断线和重连 epoch：

```powershell
cargo run --locked -p brainregiond -- scene-schema
cargo run --locked -p brainregiond -- world-schema
```

## Runtime Scene RPC v1

Unity Player 是主动连接方。当前 Rust 会话层已接受经过认证的双向字节流，并在 Windows 上提供显式启用的当前用户命名管道；命名管道先发送一次性 `runtime/challenge`，Player 用高熵预共享密钥计算 HMAC-SHA256，再把 proof 放入 `runtime/register`。完整字段顺序与配置见 [Rust Agent Core 架构决策](agent_core_architecture.zh-CN.md#windows-runtime-命名管道)。注册包含新的进程实例 ID、会话 ID、build ID、Unity/platform 信息、revision 和 capabilities。进程重启必须产生新 session；同一认证主体重连时 daemon 会递增 connection epoch，让旧 pending 请求失败并重新取 snapshot。

Agent Core 控制面已经提供 `scene/peers/list` 和 `scene/peer/call`。前者查看当前连接快照；后者只允许 v1 方法，并同时要求 Player 支持与配对策略授予对应 capability。Windows Player 端现已具备连接实现，并已在独立 Unity `6000.0.59f2` Windows x64 IL2CPP Development Player 中完成 daemon 端到端 smoke。夹具包含稳定 ID、手写故障 adapter、attribute 生成的 AOT-safe adapter、构建期自动生成的单 Prefab Catalog、Prefab lifecycle probe 和动态/additive scene 驱动；真实 VR 项目仍未导入 package，也未被本阶段修改。

IL2CPP 下 `NamedPipeClientStream` 的 overlapped async 路径会触发内部 native completion callback 无法 marshal。当前 Windows transport 因此使用一个专用长运行 I/O 线程，通过 `PeekNamedPipe` 做非阻塞读取探测，再在同一线程串行读写；这样不会阻塞 Unity 主线程，也不会在同步 pipe handle 上并发 `ReadFile`/`WriteFile`。每轮分别最多处理 32 个入站和 32 个出站帧，空闲时等待 2 ms，主线程上的 Dispatcher 帧预算和有界响应队列保持不变。

Windows 传输默认关闭。Player 侧可在运行时调用 `SetPairingSecret`，或读取 `BRAINREGIOND_SCENE_PAIRING_SECRET`；管道名默认读取 `BRAINREGIOND_SCENE_PIPE_NAME`。密钥不会被 Unity 序列化，但环境变量和托管字符串无法保证内存零残留，因此正式产品应由启动器或配对流程注入并轮换。`connectOnEnable` 默认 false，只有项目显式启用才连接。

第一批方法：

| 方法 | 作用 | 权限 |
|---|---|---|
| `runtime/info` | Player、build、scene、revision 状态 | `scene.read` |
| `scene/hierarchy` | 有界、可分页的对象层级 | `scene.read` |
| `object/inspect` | Transform 与显式 adapter 属性 | `scene.read` |
| `prefab/list` | 可创建的 Prefab 目录 | `scene.read` |
| `scene/preview` | 全量校验命令并签发短时 token，不修改场景 | 对应写权限 |
| `scene/apply` | revision 一致时原子应用 preview | 对应写权限 |
| `history/undo` | 撤销最近一个 Runtime Scene RPC 事务 | `scene.undo` |
| `logs/poll` | 按 sequence 增量读取 Player 日志 | `logs.read` |
| `persistence/list` | 列出有效和损坏的固定 slot | `persistence.read` |
| `persistence/save` | 原子保存当前 sandbox 世界 | `persistence.write` + `scene.read` |
| `persistence/loadPreview` | 校验存档及结构恢复计划，不修改世界 | `persistence.read` + `scene.write` + `scene.spawn` |
| `persistence/load` | 使用短时 preview token 恢复世界 | `persistence.read` + `persistence.write` + `scene.write` + `scene.spawn` |

v1 只允许四种可逆操作：`spawn`、`set_transform`、`set_active`、`set_properties`。暂不开放任意 `AddComponent`、`RemoveComponent`、方法反射调用、文件路径或 `eval`。

`sceneRevision` 只描述 RPC 可配置、可持久化的世界状态。物理、动画和每帧 Transform 抖动不应逐帧推进它；项目系统若在 RPC 外修改了持久状态，应调用 `NotifyExternalPersistentMutation`。

Catalog Prefab 的 root 必须保持 inactive。Runtime 先实例化 inactive 模板、分配正式 object ID、注册、应用 Transform，并在整批可逆命令执行成功后才统一激活。因此项目 `Awake`/`OnEnable` 能看到正式 ID，不会看到模板 ID。生命周期回调仍不得执行不可逆的文件、网络或业务副作用：Unity 无法对任意回调副作用做通用事务回滚。

默认只索引 `sandboxRoot`。只有项目显式启用 `includeLoadedScenes`，Controller 才会合并所有已加载 scene 的 identity。active 动态对象可通过 `OnEnable` 自动发现；销毁会自动清理。场景加载后才创建且始终 inactive 的对象不会产生 `OnEnable`，项目应显式调用 `TryRegister`，再调用 `NotifyExternalPersistentMutation`。

## 类 VaM 内容模型

建议把裸 `GameObject` 提升为一个明确的运行时实体（类似 Atom）：

```text
Runtime entity
├─ stable objectId
├─ prefabId / content schema version
├─ transform and parent
├─ explicit property adapters
├─ behavior instance + bounded state
└─ persistence policy
```

内置内容先使用直接引用的 Prefab Catalog。后续用户内容使用 Addressables/AssetBundles 或专用 glTF、纹理、音频导入器。Unity 官方将 Addressables 定位为基于 AssetBundle、负责依赖和位置管理的运行时内容系统；AssetBundle 还是平台相关产物，Windows 与 Android 内容需要分别构建。[Unity Runtime Asset Management](https://docs.unity3d.com/Manual/assets-managing-runtime.html)、[AssetBundle 平台注意事项](https://docs.unity3d.com/Manual/assetbundles-platforms.html)

场景存档现在使用独立、版本化的 `brainregion.world.v1`。它只记录 sandbox root 下的基础实体和 Catalog 生成实体、父子关系、局部 Transform、active 状态，以及 adapter 明确标为 `Persistent && !ReadOnly` 的属性；不会写 `.unity` 场景、ScriptableObject、任意 MonoBehaviour 内存、物理/动画状态或外部文件路径。Unity 文档也明确说明部署后的 build 不能用 ScriptableObject 保存数据。[Unity ScriptableObject](https://docs.unity3d.com/6000.1/Documentation/Manual/class-ScriptableObject.html)

每个 slot 是 `Application.persistentDataPath/BrainRegion/Worlds/<slot>.brworld.json` 下的单一封装文件，slot 只允许 1–64 位 ASCII 字母、数字、下划线和连字符。当前限制为 32 个 slot、每个 1 MiB、总量 16 MiB、256 个实体和 2048 个属性。保存使用同目录临时文件、强制刷新和原子替换；覆盖可带 `expectedSlotDigest` 做比较并交换，防止误覆盖更新后的存档。摘要用于发现损坏和意外改写，不是签名，不能抵御能同时篡改文档与摘要的攻击者。

`loadPreview` 会先校验 product、build、scene、Catalog schema、基础对象身份、Prefab 白名单、父子图和所有持久化属性，返回与主体、连接 epoch、revision、mutation ID 和文档摘要绑定的短时 token，且不修改场景。`load` 在主线程再次校验后恢复基础对象、删除多余的 Catalog 实例并按原 object ID 重建缺失实例；失败会逆序回滚，回滚不完整时进入 degraded 状态并推进 dirty revision。成功加载形成外部历史屏障，旧 preview 和 Undo 不再覆盖新世界。

当前 JSON 序列化和最多 1 MiB 的文件读写是用户显式触发、配额受限的同步主线程操作。Windows smoke 已验证正确性，但在真实 VR 中启用前仍应把文件读取、摘要和序列化移到后台，只把最终验证与世界应用留在 Unity 主线程；Android/Quest 的持久化路径、原子替换语义和掉电恢复也必须单独实机验证。

## “实时改脚本”的可行定义

当前 IL2CPP 设置下应分三层：

1. 实时改参数、对象和数据驱动动作——本阶段已建立基础。
2. 预编译 Behavior Graph/节点系统——节点代码随游戏构建，图和参数可热替换。
3. 可选 AOT-safe 解释器——例如受限 Lua/自有 bytecode VM；只开放生成的 host API，并限制每帧指令、时间、内存、协程、文件和网络。

新增 C# API、原生插件或不在构建中的组件仍需重新打包。即使 Windows Player 改用 Mono 可以降低技术限制，也不应让未受信任内容获得任意 .NET、文件、网络或进程权限。

## 后续接入顺序

1. Windows 命名管道客户端、challenge/HMAC proof、有界 reader/writer queue 和重连已完成；Unity 6000.3 EditMode 测试已验证 Rust/C# golden proof、过期/越权 challenge、帧边界和队列背压。
2. 独立最小 Windows x64 IL2CPP Player 已完成真实读写 smoke：高熵随机密钥配对、`runtime/info`、`scene/hierarchy`、属性范围拒绝、preview 零副作用、apply、stale revision 拒绝、同主体重连后的精确幂等 replay、Undo，以及 Undo 后拒绝旧成功回放均通过。
3. 故障闭环已完成：同一事务先写 counter、再注入 adapter 写失败，实际 Player 会恢复 counter、保持 revision 不变并永久拒绝复用该 mutation ID；另一个事务在最后一次写入时主动断开管道，daemon 返回 `outcome=unknown, retryable=false`，Player 以新 epoch 重连后保留已提交 revision，并只用原 `clientMutationId + previewToken + expectedRevision` 精确回放得到幂等 receipt，没有二次修改。
4. Catalog 自动生成、属性 binding codegen 和 High managed stripping smoke 已完成：EditMode 验证生成源码确定性、无反射成员访问、GUID prefabId 与 tag 排序，并拒绝 active Catalog root；实际 Player 验证生成整数属性的 write/Undo，以及 Catalog prefab 的 staging/spawn/Undo。probe 证明 Prefab 的 `Awake` 和 `OnEnable` 读取到正式 object ID。
5. 运行时生命周期闭环已完成：打包 Player 实测 active 动态对象自动注册和销毁清理；显式开启 `includeLoadedScenes` 后，additive scene root 会加入和退出 registry；每次应用侧生命周期变化形成一次外部 mutation barrier，没有残留死引用。
6. `WorldDocument v1` 已完成：实际 Player 验证原子 save、slot list、preview 零副作用、摘要 CAS、精确幂等 replay、属性恢复，以及按原 ID 重建已删除的 Catalog Prefab。
7. 下一阶段建议先把持久化序列化/I/O 移出 Unity 主线程，并补齐异常中断恢复测试；再经你确认把 package 作为本地 UPM dependency 接入 VR 项目。随后做 Android ARM64/Quest、Addressables provider 和 Behavior Graph；文本脚本解释器最后接入。

独立 smoke 项目位于 `unity/SmokeProjects/WindowsScenePipePlayer/`。它只用于集成验证，不是 VR 项目模板；构建输出位于 `target/`，不会提交到仓库。构建和测试命令见该目录的 `README.md`。

生产 build 默认不自动连接。配对、远程写、内容安装、脚本逻辑和外部电脑调试必须拆成独立 capability，并留下审计 receipt。
