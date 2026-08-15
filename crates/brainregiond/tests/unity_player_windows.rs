#![cfg(windows)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use brainregiond::config::{PairingSecret, ScenePipeConfig};
use brainregiond::scene_peer::{SceneMethod, ScenePeerHandle, ScenePeerRegistry};
use brainregiond::scene_pipe::ScenePipeListener;
use brainregiond::scene_rpc::{RuntimeStatus, SceneCapability};
use serde_json::json;
use tokio::process::{Child, Command};
use tokio::time::Instant;

const PRINCIPAL_ID: &str = "unity-il2cpp-smoke";
const PLAYER_STARTUP_TIMEOUT: Duration = Duration::from_secs(60);
const PLAYER_RECONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

fn unique_pipe_name() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after Unix epoch")
        .as_nanos();
    format!("brainregion.scene.unity.{}.{}", std::process::id(), nanos)
}

fn random_pairing_secret() -> String {
    let mut raw = [0_u8; 32];
    getrandom::fill(&mut raw).expect("OS randomness must be available");
    URL_SAFE_NO_PAD.encode(raw)
}

fn configured_player_path() -> PathBuf {
    let path = PathBuf::from(
        env::var_os("BRAINREGIOND_UNITY_SMOKE_PLAYER")
            .expect("BRAINREGIOND_UNITY_SMOKE_PLAYER must point to the built smoke Player"),
    );
    assert!(path.is_absolute(), "smoke Player path must be absolute");
    assert!(path.is_file(), "smoke Player does not exist: {path:?}");
    path
}

fn start_player(path: &Path, pipe_name: &str, secret: &str, log_path: &Path) -> Child {
    let mut command = Command::new(path);
    command
        .current_dir(path.parent().expect("Player must have a parent directory"))
        .args(["-batchmode", "-nographics", "-logFile"])
        .arg(log_path)
        .env("BRAINREGIOND_SCENE_PIPE_NAME", pipe_name)
        .env("BRAINREGIOND_SCENE_PAIRING_SECRET", secret)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    command
        .spawn()
        .expect("could not launch Unity smoke Player")
}

async fn wait_for_peer(
    registry: &ScenePeerRegistry,
    minimum_epoch_exclusive: u64,
    timeout: Duration,
    player: &mut Child,
    player_log: &Path,
) -> ScenePeerHandle {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = player
            .try_wait()
            .expect("could not inspect Unity smoke Player")
        {
            panic!(
                "Unity smoke Player exited before registration with {status}; log: {player_log:?}"
            );
        }
        if let Some(handle) = registry.get(PRINCIPAL_ID).await
            && handle.connection_epoch() > minimum_epoch_exclusive
        {
            return handle;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for Unity smoke Player epoch > {minimum_epoch_exclusive}; log: {player_log:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a locally built Windows IL2CPP smoke Player"]
async fn packaged_il2cpp_player_registers_calls_and_reconnects() {
    let player_path = configured_player_path();
    let player_log = player_path.with_file_name("BrainRegionScenePipeSmoke.e2e.log");
    let pipe_name = unique_pipe_name();
    let secret = random_pairing_secret();
    let registry = ScenePeerRegistry::default();
    let config = ScenePipeConfig {
        name: pipe_name.clone(),
        principal_id: PRINCIPAL_ID.to_owned(),
        pairing_secret: PairingSecret::new(secret.as_bytes()).unwrap(),
        granted_capabilities: vec![SceneCapability::SceneRead],
        max_connections: 2,
        authentication_timeout: Duration::from_secs(10),
    };
    let listener = ScenePipeListener::start(config, registry.clone()).unwrap();
    let mut player = start_player(&player_path, &pipe_name, &secret, &player_log);

    let first = wait_for_peer(
        &registry,
        0,
        PLAYER_STARTUP_TIMEOUT,
        &mut player,
        &player_log,
    )
    .await;
    let first_snapshot = first.snapshot();
    assert_eq!(first_snapshot.runtime_status, RuntimeStatus::Ready);
    assert_eq!(
        first_snapshot.granted_capabilities,
        vec![SceneCapability::SceneRead]
    );

    let info = first
        .request(SceneMethod::RuntimeInfo, json!({}), REQUEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(info["protocolVersion"], "brainregion.scene.v1");
    assert_eq!(info["status"], "ready");
    assert_eq!(info["platform"], "WindowsPlayer");
    assert_eq!(info["product"], "BrainRegion Scene Pipe Smoke");
    assert_eq!(info["sceneRevision"], 0);

    let hierarchy = first
        .request(SceneMethod::Hierarchy, json!({}), REQUEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(hierarchy["sceneRevision"], 0);
    assert_eq!(hierarchy["nodes"].as_array().map(Vec::len), Some(1));

    let first_epoch = first.connection_epoch();
    first.close();
    drop(first);

    let second = wait_for_peer(
        &registry,
        first_epoch,
        PLAYER_RECONNECT_TIMEOUT,
        &mut player,
        &player_log,
    )
    .await;
    let second_snapshot = second.snapshot();
    assert!(second_snapshot.connection_epoch > first_epoch);
    assert_eq!(second_snapshot.instance_id, first_snapshot.instance_id);
    assert_eq!(second_snapshot.session_id, first_snapshot.session_id);
    let reconnected_info = second
        .request(SceneMethod::RuntimeInfo, json!({}), REQUEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(reconnected_info["instanceId"], first_snapshot.instance_id);
    assert_eq!(reconnected_info["sessionId"], first_snapshot.session_id);

    registry.close_all().await;
    listener.shutdown().await.unwrap();
    if player.try_wait().unwrap().is_none() {
        player.start_kill().unwrap();
    }
    tokio::time::timeout(Duration::from_secs(10), player.wait())
        .await
        .expect("Unity smoke Player did not exit after kill")
        .expect("could not wait for Unity smoke Player");
}
