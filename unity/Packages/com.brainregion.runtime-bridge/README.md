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

The package includes EditMode tests for the Rust/C# pairing golden vector,
expired and over-granted challenges, JSONL frame bounds, and writer backpressure.
Before write capabilities are enabled in a product, the integration still needs
Player tests for scene unload/reload, dynamic project-owned object registration,
prefab `Awake`/`OnEnable` staging, and a real Windows IL2CPP pipe smoke.
