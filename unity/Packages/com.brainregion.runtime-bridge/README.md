# BrainRegion Runtime Bridge

This package is the Unity Player-side core for `brainregion.scene.v1`. It is
intended for packaged applications, including Windows PCVR and Android/Quest
builds. Its Runtime assembly contains no `UnityEditor` reference and does not
depend on an Editor MCP plugin. The separate Editor assembly is only a Player-
build validator and is never referenced by Runtime.

The first slice deliberately exposes a small, reversible surface:

- enumerate registered runtime objects and inspect explicitly exposed fields;
- instantiate prefabs from an application-owned catalog;
- preview and then atomically apply transform, active-state, and exposed-property changes;
- undo the most recent Scene RPC transaction;
- enqueue JSON-RPC work from a transport and execute it under a main-thread budget.

No network listener is enabled by this package. A transport must call
`SceneRpcDispatcher.TryEnqueue(AuthenticatedPeerContext, ...)`. The transport
must authenticate and pair the connection before constructing that immutable
context; passing no context is rejected. Keeping transport separate lets a
Windows build use a user-scoped named pipe while an Android headset uses an
outbound WSS connection without changing scene semantics.

`connectionEpoch` must increase when one principal reconnects. A request queued
for an older epoch is rejected before it can touch Unity state. The dispatcher
enforces these grants independently of the transport:

| Methods/operation | Required capability |
|---|---|
| `runtime/info`, `scene/hierarchy`, `object/inspect`, `prefab/list` | `scene.read` |
| `scene/preview`, `scene/apply` | `scene.write` |
| a preview containing `spawn`, and its apply/replay | `scene.spawn` in addition to `scene.write` |
| `history/undo` | `scene.undo` |
| `logs/poll` | `logs.read` |

A preview token is bound to its authenticated principal and connection epoch;
another connection must preview again. Applied mutation receipts remain bound
to the principal so an authenticated reconnect can perform a safe idempotent
replay. `RpcObjectIdentity.allowRemoteChanges` defaults to false, so project
content must explicitly opt into mutation.

Before building, the Editor validator rejects missing/duplicate authored object
IDs and invalid prefab catalogs. Use **Tools > BrainRegion > Assign Missing
Runtime IDs**, then save the modified scenes.

JSON envelope and typed DTO parsing now happen in `TryEnqueue`, before the
request reaches the Unity main-thread queue. Completion callbacks still run on
the main thread and must only enqueue bytes for a non-blocking transport writer.
When the dispatcher is disabled it stops accepting work and completes every
pending request with `dispatcher_stopped`. Network threads obtain registration
text through `TryGetCachedRegistrationNotification`; only main-thread code may
call `BuildRegistrationNotification`.

Arbitrary reflection, method invocation, component injection, and C# evaluation
are intentionally absent. Project components opt in by deriving from
`RpcPropertyAdapterBehaviour` and implementing typed validation/read/write.

Before a transport is enabled, the integration still needs Player tests for
scene unload/reload, dynamic project-owned object registration, and prefab
`Awake`/`OnEnable` staging. Those lifecycle paths are not an authenticated
network surface yet.
