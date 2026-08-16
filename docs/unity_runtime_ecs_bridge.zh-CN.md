# Unity Runtime Bridge 的 ECS 混合接入决策

## 结论

BrainRegion 不应为了接入当前 VR 项目而把控制面、网络、权限或整个 Scene RPC 改写成 ECS。推荐结构是：

- GameObject/MonoBehaviour 继续承载 XR Rig、交互器、UI、音频和少量可编辑场景对象；
- Entities 继续承载 XPBD 毛发、布料、粒子、约束和其他大规模数据并行模拟；
- 一个可选的 Entities provider 只把“角色、发型、布料、附件”等逻辑 owner 投影成 Scene RPC 虚拟对象；
- RPC 写操作进入有界逻辑命令队列，再由固定更新组中的 System 写入 ECB；
- 网络线程、后台文件线程绝不直接访问 `EntityManager`、`EntityQuery`、NativeContainer 或 Unity 对象。

基础 `com.brainregion.runtime-bridge` 包继续不依赖 `com.unity.entities`。Entities 适配应放进独立 companion assembly/package，避免让没有 DOTS 的 Player 增加依赖、代码体积和 stripping 风险。

## 对真实 VR 项目的只读审计

项目根：`D:\Unity\My Project\VR Project`，本阶段未修改。

- Unity `6000.3.20f1`；
- `com.unity.entities 1.4.8`；
- 已使用 `SubScene`、`ISystem`、`SystemBase`、`IJobEntity`、Baker 与 Begin/End Initialization ECB；
- XPBD 毛发和布料有独立 Runtime Assembly、批量粒子/约束 job 和托管 Presentation bridge；
- `XpbdHairRuntimeIdentity`、`XpbdClothRuntimeIdentity` 已保存 `AssetId` 与 `QualityProfileId`；
- 内容系统已有后台 Provider Fetch、owner 线程 `PumpCommits` 和 ECB 提交边界。

最后一点可以直接作为 BrainRegion 的参考模式：后台只处理不可变字节/JSON；主线程或 ECS owner system 在安全点把经过验证的逻辑命令写入 ECB。

## 不应暴露的对象

以下状态不能成为 Scene RPC 层级节点，也不应进入 WorldDocument：

- 每个毛发/布料粒子；
- 距离、弯曲、碰撞等约束实体；
- 求解器临时 buffer、lambda、速度和预测位置；
- 原始 `Entity.Index/Version`；
- 每帧变化的 `LocalToWorld`、渲染池槽位和 job 临时统计。

它们数量大、每帧变化，而且身份只在当前 World 内有效。把它们映射为 RPC 对象会导致 hierarchy、JSON、主线程同步点和存档体积同时失控。

建议只暴露逻辑聚合：

```text
character:<stable-instance-id>
├─ hair:<part-id>        assetId / qualityProfile / enabled / attachment
├─ cloth:<part-id>       assetId / qualityProfile / enabled / attachment
└─ accessory:<part-id>   content identity / local transform / visible
```

## 身份与 revision

协议身份必须是构建或业务层分配的稳定字符串，不能使用原始 `Entity`。建议新增类似下面的 opt-in 组件，或从现有角色 session + part ID 生成同等稳定的键：

```csharp
public struct BrainRegionRuntimeEntityId : IComponentData
{
    public FixedString128Bytes Value;
}
```

`Entity` 只作为 provider 内部的当前帧缓存；World 重建、SubScene reload 或 structural change 后必须重新解析。`sceneRevision` 只在 RPC 可编辑配置、内容挂载或持久化状态改变时推进，不能跟随模拟帧、粒子位置或动画变化。

## 读取路径

不应在 `SceneRpcDispatcher.Update` 中临时执行 EntityQuery 并调用 `CompleteDependency()`，这会把 Agent 读取变成模拟同步栅栏。推荐新增一个只读桥接 System：

1. 在已知的 simulation/presentation 安全点查询少量逻辑 owner；
2. 复制成有界、托管或不可变的双缓冲 snapshot；
3. snapshot 带 `providerRevision`、`observedFrame`、`schemaVersion` 和 dropped/overflow 状态；
4. Scene RPC 只读上一份完整 snapshot，不等待当前帧 jobs。

高频 profiler/统计使用定长结构和增量轮询；层级与属性描述只在 provider revision 变化时重建。

## 写入与恢复路径

RPC 不直接调用 `EntityManager.SetComponentData` 或立即做 structural change。建议：

1. 网络边界完成认证、capability、DTO 与 deadline 校验；
2. Unity 主线程把逻辑命令放进固定容量队列；
3. companion System 在约定更新组消费命令并写入 ECB；
4. ECB playback 后生成包含 mutation ID、实际 revision 和结果摘要的 receipt；
5. Dispatcher 在后续帧完成原 RPC；超时或断线后的写操作结果标记为 unknown，客户端只能用同一 mutation ID 查明，不能自动换 ID 重试。

当前项目应优先复用 `XpbdCharacterHostLoadCoordinator` 和 Hair attachment transaction，而不是绕过它们直接创建底层实体。存档只记录 content key、稳定 part ID、quality profile、附件目标和明确 opt-in 的配置；加载时重新走现有 Provider + coordinator + ECB 流程。

## WorldDocument 与线程边界

基础 Runtime Bridge 已把目录扫描、严格 JSON 读取、SHA-256、编码和原子写入迁移到后台串行 worker。以下步骤仍必须在 Unity/ECS 安全点：

- 从 GameObject adapter 或 provider snapshot 捕获逻辑状态；
- 校验当前 build/catalog/provider schema；
- 创建加载计划；
- 应用 GameObject 事务或把 ECS 命令写入 ECB；
- playback 后登记 receipt 和推进 revision。

因此 ECS provider 应输出小型、不可变的逻辑 snapshot，而不是让后台线程读取 Entity World。

## 实施顺序

1. 保持基础包无 Entities 依赖，先完成后台 WorldDocument I/O 和故障语义；
2. 在独立 companion assembly 实现只读 owner snapshot，先接角色、发型和布料目录；
3. 增加逻辑配置写入和 receipt，复用项目现有 coordinator/ECB；
4. 将 provider 的可持久化逻辑状态纳入 WorldDocument 扩展区；
5. 在 Windows PCVR 和 Android/Quest 分别测量主线程耗时、同步点、队列背压、IL2CPP stripping 和 SubScene reload；
6. 通过性能门槛后才开放远程写，粒子/约束级调试只提供聚合统计或专用 Development capability。
