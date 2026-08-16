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

No network listener is enabled by this package. The opt-in
`WindowsScenePipeTransport` is a Player-side client for brainregiond's
current-user named pipe; it is disabled until `Connect()` is called or
`connectOnEnable` is explicitly selected. Android/WSS is not implemented.
Custom transports must call
`SceneRpcDispatcher.TryEnqueue(AuthenticatedPeerContext, ...)` after pairing;
passing no context is rejected.

The Windows transport obtains its pipe name from
`BRAINREGIOND_SCENE_PIPE_NAME` (or the serialized non-secret fallback) and its
32..4096 UTF-8 byte secret from `BRAINREGIOND_SCENE_PAIRING_SECRET` or
`SetPairingSecret`. Never serialize the secret into a scene or prefab. It
strictly parses the one-time challenge, creates an HMAC-SHA256 registration,
and uses only the server-granted capabilities bound into that proof. Pipe I/O
runs off the Unity main thread; responses enter a bounded writer queue and
overload closes the connection instead of blocking a frame.

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

The default application catalog can be generated instead of maintained by hand.
Add the `BrainRegionRuntimePrefab` asset label to each allowed prefab, keep exactly
one `RpcObjectIdentity` on its root, and keep that root inactive. The inactive
template is a safety boundary: spawn assigns the final object ID, registers the
object, and applies its transform before the transaction activates it and permits
project `Awake`/`OnEnable` callbacks. Active catalog roots fail validation and
block the Player build. Then run **Tools > BrainRegion > Rebuild Runtime Prefab
Catalog**. The generated asset is written to
`Assets/BrainRegionGenerated/RuntimePrefabCatalog.asset`; each wire `prefabId`
uses the prefab asset GUID, so moving or renaming the asset does not change its
identity. Other sorted Unity asset labels become catalog tags. The same generator
runs before Player builds when labeled sources or the generated catalog exist.

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
For simple public fields or properties, mark a public `MonoBehaviour` with
`[RpcBindingTarget(componentKey, typeId)]` and selected writable members with
`[RpcExposedProperty(propertyId)]`, then run **Tools > BrainRegion > Generate
Runtime Property Bindings**. The Editor uses reflection only while generating;
the emitted `IRpcPropertyAdapter` code performs direct typed member access in the
Player. v1 generation supports `bool`, `int`, `float`, `double`, and bounded
`string` members. Complex state and side effects still require a hand-written
adapter with explicit rollback behavior.

Active application-created `RpcObjectIdentity` instances are reconciled on the
main thread after `OnEnable`; destroyed objects are removed without retaining a
dead Unity reference. `RuntimeSceneController.includeLoadedScenes` is an explicit,
default-off opt-in for indexing identities in additive loaded scenes. Each batch
of external lifecycle changes invalidates previews, clears incompatible Undo
history, and advances `sceneRevision` once. Inactive objects created after scene
load cannot signal `OnEnable`; their owner must call `TryRegister` and then
`NotifyExternalPersistentMutation` explicitly.

Prefab activation is intentionally the final transaction step. Project
`Awake`/`OnEnable` code may observe the assigned ID, but it must not perform
irreversible filesystem, network, or application side effects: Unity does not
provide a general rollback mechanism for arbitrary lifecycle callbacks.

The package includes EditMode tests for the Rust/C# pairing golden vector,
expired and over-granted challenges, JSONL frame bounds, writer backpressure,
deterministic binding source, and GUID-based catalog generation. The repository
smoke Player also exercises generated property write/Undo, generated catalog
spawn/Undo, pre-activation identity staging, active dynamic object registration
and destruction cleanup, and additive scene load/unload under Windows IL2CPP
with High managed stripping. Product integration still needs platform-specific
PlayMode coverage and a real Windows/Android VR build after package import.
