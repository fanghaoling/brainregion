#![cfg(windows)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use brainregiond::config::{PairingSecret, ScenePipeConfig};
use brainregiond::error::BrainregiondError;
use brainregiond::scene_peer::{SceneMethod, ScenePeerHandle, ScenePeerRegistry};
use brainregiond::scene_pipe::ScenePipeListener;
use brainregiond::scene_rpc::{RuntimeStatus, SceneCapability};
use serde_json::{Value, json};
use tokio::process::{Child, Command};
use tokio::time::Instant;

const PRINCIPAL_ID: &str = "unity-il2cpp-smoke";
const PLAYER_STARTUP_TIMEOUT: Duration = Duration::from_secs(60);
const PLAYER_RECONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const SMOKE_OBJECT_ID: &str = "smoke-object-01";
const SMOKE_COMPONENT_ID: &str = "smoke-object-01/smoke";

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

async fn inspect_counter(peer: &ScenePeerHandle) -> i64 {
    let inspected = peer
        .request(
            SceneMethod::Inspect,
            json!({
                "objectId": SMOKE_OBJECT_ID,
                "componentIds": [SMOKE_COMPONENT_ID],
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(
        inspected["object"]["components"][0]["componentId"],
        SMOKE_COMPONENT_ID
    );
    assert_eq!(
        inspected["object"]["components"][0]["properties"][0]["descriptor"]["propertyId"],
        "counter"
    );
    inspected["object"]["components"][0]["properties"][0]["value"]
        .as_i64()
        .expect("smoke counter must be an integer")
}

fn assert_upstream_error(error: BrainregiondError, code: i64, reason: &str) -> Value {
    let BrainregiondError::Upstream(payload) = error else {
        panic!("expected Runtime Scene RPC upstream error, got {error}");
    };
    assert_eq!(payload["code"], code);
    assert_eq!(payload["data"]["reason"], reason);
    payload
}

fn assert_disconnected_unknown_outcome(error: BrainregiondError) {
    match error {
        BrainregiondError::Upstream(payload) => {
            assert_eq!(payload["code"], -32011);
            assert_eq!(payload["data"]["outcome"], "unknown");
            assert_eq!(payload["data"]["retryable"], false);
        }
        error => panic!("expected disconnected/unknown apply outcome, got {error}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a locally built Windows IL2CPP smoke Player"]
async fn packaged_il2cpp_player_executes_transaction_and_reconnects() {
    let player_path = configured_player_path();
    let player_log = player_path.with_file_name("BrainRegionScenePipeSmoke.e2e.log");
    let pipe_name = unique_pipe_name();
    let secret = random_pairing_secret();
    let registry = ScenePeerRegistry::default();
    let config = ScenePipeConfig {
        name: pipe_name.clone(),
        principal_id: PRINCIPAL_ID.to_owned(),
        pairing_secret: PairingSecret::new(secret.as_bytes()).unwrap(),
        granted_capabilities: vec![
            SceneCapability::SceneRead,
            SceneCapability::SceneWrite,
            SceneCapability::SceneUndo,
        ],
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
        vec![
            SceneCapability::SceneRead,
            SceneCapability::SceneUndo,
            SceneCapability::SceneWrite,
        ]
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
    assert_eq!(hierarchy["nodes"][0]["objectId"], SMOKE_OBJECT_ID);
    assert_eq!(inspect_counter(&first).await, 1);

    let invalid_value = first
        .request(
            SceneMethod::Preview,
            json!({
                "expectedRevision": 0,
                "clientMutationId": "smoke-invalid-01",
                "commands": [{
                    "kind": "set_properties",
                    "objectId": SMOKE_OBJECT_ID,
                    "componentId": SMOKE_COMPONENT_ID,
                    "changes": [{"propertyId": "counter", "value": 101}],
                }],
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap_err();
    let invalid_payload =
        assert_upstream_error(invalid_value, -32013, "property_validation_failed");
    assert_eq!(invalid_payload["data"]["propertyId"], "counter");
    assert_eq!(inspect_counter(&first).await, 1);

    let mutation_id = "smoke-counter-01";
    let preview = first
        .request(
            SceneMethod::Preview,
            json!({
                "expectedRevision": 0,
                "clientMutationId": mutation_id,
                "commands": [{
                    "kind": "set_properties",
                    "objectId": SMOKE_OBJECT_ID,
                    "componentId": SMOKE_COMPONENT_ID,
                    "changes": [{"propertyId": "counter", "value": 7}],
                }],
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(preview["baseRevision"], 0);
    assert_eq!(preview["clientMutationId"], mutation_id);
    let preview_token = preview["previewToken"]
        .as_str()
        .expect("preview must return a token")
        .to_owned();

    // Preview is validation-only and must not mutate the Player.
    assert_eq!(inspect_counter(&first).await, 1);

    let apply_params = json!({
        "previewToken": preview_token,
        "expectedRevision": 0,
        "clientMutationId": mutation_id,
    });
    let applied = first
        .request(SceneMethod::Apply, apply_params.clone(), REQUEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(applied["sceneRevision"], 1);
    assert_eq!(applied["clientMutationId"], mutation_id);
    assert_eq!(applied["idempotentReplay"], false);
    let undo_id = applied["undoId"]
        .as_str()
        .expect("apply must return an undo id")
        .to_owned();
    assert_eq!(inspect_counter(&first).await, 7);

    let stale_error = first
        .request(
            SceneMethod::Preview,
            json!({
                "expectedRevision": 0,
                "clientMutationId": "smoke-stale-01",
                "commands": [{
                    "kind": "set_properties",
                    "objectId": SMOKE_OBJECT_ID,
                    "componentId": SMOKE_COMPONENT_ID,
                    "changes": [{"propertyId": "counter", "value": 9}],
                }],
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap_err();
    let stale_payload = assert_upstream_error(stale_error, -32010, "revision_conflict");
    assert_eq!(stale_payload["data"]["expectedRevision"], 0);
    assert_eq!(stale_payload["data"]["actualRevision"], 1);

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
    assert_eq!(second_snapshot.scene_revision, 1);
    let reconnected_info = second
        .request(SceneMethod::RuntimeInfo, json!({}), REQUEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(reconnected_info["instanceId"], first_snapshot.instance_id);
    assert_eq!(reconnected_info["sessionId"], first_snapshot.session_id);
    assert_eq!(reconnected_info["sceneRevision"], 1);

    // An exact apply replay by the same principal survives a new connection epoch
    // and confirms the already-committed outcome without applying it twice.
    let replayed = second
        .request(SceneMethod::Apply, apply_params.clone(), REQUEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(replayed["sceneRevision"], 1);
    assert_eq!(replayed["clientMutationId"], mutation_id);
    assert_eq!(replayed["undoId"], undo_id);
    assert_eq!(replayed["idempotentReplay"], true);
    assert_eq!(inspect_counter(&second).await, 7);

    let undone = second
        .request(
            SceneMethod::Undo,
            json!({"expectedRevision": 1, "undoId": undo_id}),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(undone["sceneRevision"], 2);
    assert_eq!(undone["undoneClientMutationId"], mutation_id);
    assert_eq!(inspect_counter(&second).await, 1);

    let replay_after_undo = second
        .request(SceneMethod::Apply, apply_params, REQUEST_TIMEOUT)
        .await
        .unwrap_err();
    assert_upstream_error(replay_after_undo, -32016, "mutation_was_undone");

    // A later property failure must reverse the earlier counter write, keep the
    // revision unchanged, and permanently consume that mutation ID.
    let rollback_mutation_id = "smoke-rollback-01";
    let rollback_preview = second
        .request(
            SceneMethod::Preview,
            json!({
                "expectedRevision": 2,
                "clientMutationId": rollback_mutation_id,
                "commands": [{
                    "kind": "set_properties",
                    "objectId": SMOKE_OBJECT_ID,
                    "componentId": SMOKE_COMPONENT_ID,
                    "changes": [
                        {"propertyId": "counter", "value": 13},
                        {"propertyId": "inject_write_failure", "value": true},
                    ],
                }],
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    let rollback_apply_params = json!({
        "previewToken": rollback_preview["previewToken"],
        "expectedRevision": 2,
        "clientMutationId": rollback_mutation_id,
    });
    let rolled_back = second
        .request(
            SceneMethod::Apply,
            rollback_apply_params.clone(),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap_err();
    let rolled_back_payload = assert_upstream_error(rolled_back, -32013, "property_write_failed");
    assert_eq!(rolled_back_payload["data"]["mutationStatus"], "rolledback");
    assert_eq!(rolled_back_payload["data"]["lastKnownRevision"], 2);
    assert_eq!(inspect_counter(&second).await, 1);

    let rolled_back_replay = second
        .request(SceneMethod::Apply, rollback_apply_params, REQUEST_TIMEOUT)
        .await
        .unwrap_err();
    assert_upstream_error(
        rolled_back_replay,
        -32013,
        "mutation_previously_rolled_back",
    );

    // The final test-only property disconnects the pipe during the last write.
    // The client cannot know whether apply committed, so it must not invent a new
    // mutation ID and retry. Reconnect and replay the exact request to query the
    // session receipt instead.
    let unknown_mutation_id = "smoke-response-lost-01";
    let unknown_preview = second
        .request(
            SceneMethod::Preview,
            json!({
                "expectedRevision": 2,
                "clientMutationId": unknown_mutation_id,
                "commands": [{
                    "kind": "set_properties",
                    "objectId": SMOKE_OBJECT_ID,
                    "componentId": SMOKE_COMPONENT_ID,
                    "changes": [
                        {"propertyId": "counter", "value": 23},
                        {"propertyId": "disconnect_after_write", "value": true},
                    ],
                }],
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    let unknown_apply_params = json!({
        "previewToken": unknown_preview["previewToken"],
        "expectedRevision": 2,
        "clientMutationId": unknown_mutation_id,
    });
    let second_epoch = second.connection_epoch();
    let unknown_outcome = second
        .request(
            SceneMethod::Apply,
            unknown_apply_params.clone(),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap_err();
    assert_disconnected_unknown_outcome(unknown_outcome);
    drop(second);

    let third = wait_for_peer(
        &registry,
        second_epoch,
        PLAYER_RECONNECT_TIMEOUT,
        &mut player,
        &player_log,
    )
    .await;
    assert_eq!(third.snapshot().scene_revision, 3);
    assert_eq!(inspect_counter(&third).await, 23);

    let confirmed = third
        .request(SceneMethod::Apply, unknown_apply_params, REQUEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(confirmed["sceneRevision"], 3);
    assert_eq!(confirmed["clientMutationId"], unknown_mutation_id);
    assert_eq!(confirmed["idempotentReplay"], true);
    let unknown_undo_id = confirmed["undoId"]
        .as_str()
        .expect("confirmed apply must retain its undo id");
    let unknown_undone = third
        .request(
            SceneMethod::Undo,
            json!({"expectedRevision": 3, "undoId": unknown_undo_id}),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(unknown_undone["sceneRevision"], 4);
    assert_eq!(
        unknown_undone["undoneClientMutationId"],
        unknown_mutation_id
    );
    assert_eq!(inspect_counter(&third).await, 1);

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
