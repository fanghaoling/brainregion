# Windows Scene Pipe Smoke Player

This minimal Unity 6000.0+ project builds a Windows x64 IL2CPP Development
Player that contains the real `com.brainregion.runtime-bridge` package. It is
an integration fixture, not the VR project and not a product template.

Build it from PowerShell:

```powershell
./build-windows-il2cpp.ps1 `
  -UnityEditor 'C:\Program Files\Unity\Hub\Editor\6000.0.59f2\Editor\Unity.exe'
```

The build command prints the generated executable path.

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
AOT-safe integer property adapter. The Rust test creates a fresh current-user
pipe and random pairing secret, launches the packaged Player, verifies property
bounds, preview isolation, apply, stale-revision rejection, exact idempotent
replay after reconnect, and Undo. It also verifies that an undone mutation cannot
replay as success, and kills only the child Player it launched.

The fixture currently pins `6000.0.59f2`, the installed editor with Windows
IL2CPP support on the verification machine. The package itself is also compiled
and tested under Unity `6000.3.20f1`; use an exact 6000.3 IL2CPP module before
the real VR project build.
