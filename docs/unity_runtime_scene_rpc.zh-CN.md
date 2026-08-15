# Unity Player 运行时开放编辑与 Scene RPC v1

状态：第二阶段核心纵切已实现，尚未接入实际 VR 项目或启用网络传输。

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
- `RuntimeSceneController`：层级、检查、Prefab 创建、Transform/active/显式属性修改；
- `scene/preview -> scene/apply`：预览 token 绑定 revision 和 mutation ID；
- 单调 `sceneRevision`、幂等 mutation receipt、失败回滚和运行时 Undo；
- `SceneRpcDispatcher`：认证主体、连接 epoch、逐方法 capability、1 MiB 帧上限、有界队列、deadline、每帧数量/时间预算；
- `RuntimeLogBuffer`：线程安全、容量受限的 Player 日志轮询。
- 构建校验器：拒绝场景对象 ID 缺失/重复和无效 Prefab Catalog；菜单可为已加载场景分配缺失 ID。

协议 schema 与跨语言示例位于：

```text
schemas/scene-rpc/v1/
```

Rust 侧已经嵌入同一 schema，并提供严格 DTO 与验证：

```powershell
cargo run --locked -p brainregiond -- scene-schema
```

## Runtime Scene RPC v1

Unity Player 是主动连接方；正式传输将在后续实现。连接成功后先发送 `runtime/register`，其中包含新的进程实例 ID、会话 ID、build ID、Unity/platform 信息、revision 和 capabilities。进程重启必须产生新 session；重连后 daemon 必须让旧 pending 请求失败并重新取 snapshot。

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

v1 只允许四种可逆操作：`spawn`、`set_transform`、`set_active`、`set_properties`。暂不开放任意 `AddComponent`、`RemoveComponent`、方法反射调用、文件路径或 `eval`。

`sceneRevision` 只描述 RPC 可配置、可持久化的世界状态。物理、动画和每帧 Transform 抖动不应逐帧推进它；项目系统若在 RPC 外修改了持久状态，应调用 `NotifyExternalPersistentMutation`。

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

场景存档应是独立、版本化的 `WorldDocument`，只记录 sandbox root 下的实体、父子关系、Prefab ID 和允许持久化的属性 override。不要尝试在 Player 中写 `.unity` 场景或 ScriptableObject asset；Unity 文档明确说明部署后的 build 不能用 ScriptableObject 保存数据。[Unity ScriptableObject](https://docs.unity3d.com/6000.1/Documentation/Manual/class-ScriptableObject.html)

## “实时改脚本”的可行定义

当前 IL2CPP 设置下应分三层：

1. 实时改参数、对象和数据驱动动作——本阶段已建立基础。
2. 预编译 Behavior Graph/节点系统——节点代码随游戏构建，图和参数可热替换。
3. 可选 AOT-safe 解释器——例如受限 Lua/自有 bytecode VM；只开放生成的 host API，并限制每帧指令、时间、内存、协程、文件和网络。

新增 C# API、原生插件或不在构建中的组件仍需重新打包。即使 Windows Player 改用 Mono 可以降低技术限制，也不应让未受信任内容获得任意 .NET、文件、网络或进程权限。

## 后续接入顺序

1. 在 `brainregiond` 实现 Runtime peer registry 与双向会话：Windows PCVR 优先当前用户命名管道，Android/Quest 使用 Player 主动发起的 WSS + 配对。
2. 用 mock Player 完成 register、hierarchy、preview/apply、stale revision、断线重连测试。
3. 再把 package 作为本地 UPM dependency 接入 VR 项目，先做 Windows IL2CPP Development build smoke；确认后再做 Android ARM64/Quest。
4. 补 Catalog 生成、属性 binding codegen 和实际 IL2CPP stripping smoke。
5. 增加 `WorldDocument` 存档、Addressables provider 和 Behavior Graph；文本脚本解释器最后接入。

生产 build 默认不自动连接。配对、远程写、内容安装、脚本逻辑和外部电脑调试必须拆成独立 capability，并留下审计 receipt。
