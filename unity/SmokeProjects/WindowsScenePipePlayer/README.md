# Windows Scene Pipe Smoke Player

This minimal Unity 6000.0+ project builds a Windows x64 IL2CPP Development
Player with Release IL2CPP compilation and High managed stripping. It contains
the real `com.brainregion.runtime-bridge` package and is an integration fixture,
not the VR project or a product template.

Build it from PowerShell:

```powershell
./build-windows-il2cpp.ps1 `
  -UnityEditor 'C:\Program Files\Unity\Hub\Editor\6000.0.59f2\Editor\Unity.exe'
```

The build script first runs the package binding generator in a separate Unity
process, then starts a clean compilation/build process and prints the generated
executable path. It also creates a temporary labeled prefab, generates the
application catalog, validates the catalog reference in Unity's build-scene
copy, and removes only those temporary source assets after the Player finishes.

Run the package EditMode tests in the same pinned project:

```powershell
./test-editmode.ps1 `
  -UnityEditor 'C:\Program Files\Unity\Hub\Editor\6000.0.59f2\Editor\Unity.exe'
```

Then set the Player path in `BRAINREGIOND_UNITY_SMOKE_PLAYER` and run the
ignored Rust integration test:

```powershell
$env:BRAINREGIOND_UNITY_SMOKE_PLAYER = `
  (Resolve-Path '../../../target/unity-scene-pipe-smoke/BrainRegionScenePipeSmoke.exe')
cargo test --locked -p brainregiond --test unity_player_windows `
  -- --ignored --test-threads=1
```

The generated scene contains one stable, explicitly writable object with an
AOT-safe integer property adapter. Two additional test-only properties inject a
late adapter write failure or close the pipe during the last write. The Rust test
creates a fresh current-user pipe and random pairing secret, launches the
packaged Player, and verifies property bounds, preview isolation, apply,
stale-revision rejection, exact idempotent replay after reconnect, and Undo. It
also proves that a failed multi-property write is rolled back without advancing
the revision, and that a committed apply whose response was lost is reported as
an unknown non-retryable outcome and can be confirmed by replaying only the exact
same mutation request after reconnect. The test kills only the child Player it
launched. It additionally writes and undoes an attribute-generated property,
lists the generated GUID-based prefab catalog, spawns that prefab, and undoes the
spawn while running the High-stripping IL2CPP build. The catalog template is
inactive; a read-only probe proves its `Awake` and `OnEnable` callbacks observe
the final spawned object ID. Test-only application lifecycle commands then prove
that an active dynamic identity is automatically registered and cleaned up, and
that an additive scene identity joins and leaves the registry with one external
revision barrier per lifecycle change.

The same end-to-end test saves both the base world and a world containing a
spawned catalog prefab, lists the fixed slots, rejects invalid slots and stale
slot digests, proves load preview has no side effects, restores a changed
persistent property, and recreates a deleted prefab with its original object ID.
Exact save/load replays return their receipts without applying the mutation a
second time.

The fixture currently pins `6000.0.59f2`, the installed editor with Windows
IL2CPP support on the verification machine. The package itself is also compiled
and tested under Unity `6000.3.20f1`; use an exact 6000.3 IL2CPP module before
the real VR project build.
