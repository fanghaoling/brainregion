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
const GENERATED_COMPONENT_ID: &str = "smoke-object-01/generated";

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

async fn inspect_property(
    peer: &ScenePeerHandle,
    object_id: &str,
    component_id: &str,
    property_id: &str,
) -> Value {
    let inspected = peer
        .request(
            SceneMethod::Inspect,
            json!({
                "objectId": object_id,
                "componentIds": [component_id],
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    let component = inspected["object"]["components"]
        .as_array()
        .and_then(|components| {
            components
                .iter()
                .find(|component| component["componentId"] == component_id)
        })
        .expect("requested smoke component must be present");
    component["properties"]
        .as_array()
        .and_then(|properties| {
            properties
                .iter()
                .find(|property| property["descriptor"]["propertyId"] == property_id)
        })
        .expect("requested smoke property must be present")["value"]
        .clone()
}

async fn inspect_counter(peer: &ScenePeerHandle) -> i64 {
    inspect_property(peer, SMOKE_OBJECT_ID, SMOKE_COMPONENT_ID, "counter")
        .await
        .as_i64()
        .expect("smoke counter must be an integer")
}

async fn inspect_generated_brightness(peer: &ScenePeerHandle) -> i64 {
    inspect_property(peer, SMOKE_OBJECT_ID, GENERATED_COMPONENT_ID, "brightness")
        .await
        .as_i64()
        .expect("generated brightness must be an integer")
}

async fn wait_for_scene_revision(peer: &ScenePeerHandle, expected: i64) {
    let deadline = Instant::now() + REQUEST_TIMEOUT;
    loop {
        let info = peer
            .request(SceneMethod::RuntimeInfo, json!({}), REQUEST_TIMEOUT)
            .await
            .unwrap();
        if info["sceneRevision"] == expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for scene revision {expected}; latest info: {info:#?}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn apply_lifecycle_command(
    peer: &ScenePeerHandle,
    expected_revision: i64,
    mutation_id: &str,
    command: &str,
) {
    let preview = peer
        .request(
            SceneMethod::Preview,
            json!({
                "expectedRevision": expected_revision,
                "clientMutationId": mutation_id,
                "commands": [{
                    "kind": "set_properties",
                    "objectId": SMOKE_OBJECT_ID,
                    "componentId": SMOKE_COMPONENT_ID,
                    "changes": [{"propertyId": "lifecycle_command", "value": command}],
                }],
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    let applied = peer
        .request(
            SceneMethod::Apply,
            json!({
                "previewToken": preview["previewToken"],
                "expectedRevision": expected_revision,
                "clientMutationId": mutation_id,
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(applied["sceneRevision"], expected_revision + 1);
    wait_for_scene_revision(peer, expected_revision + 2).await;
}

async fn hierarchy_names(peer: &ScenePeerHandle) -> Vec<String> {
    let hierarchy = peer
        .request(
            SceneMethod::Hierarchy,
            json!({"includeInactive": true}),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    hierarchy["nodes"]
        .as_array()
        .expect("hierarchy nodes must be an array")
        .iter()
        .map(|node| {
            node["name"]
                .as_str()
                .expect("hierarchy node name must be a string")
                .to_owned()
        })
        .collect()
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
            SceneCapability::PersistenceRead,
            SceneCapability::PersistenceWrite,
            SceneCapability::SceneRead,
            SceneCapability::SceneSpawn,
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
            SceneCapability::PersistenceRead,
            SceneCapability::PersistenceWrite,
            SceneCapability::SceneRead,
            SceneCapability::SceneSpawn,
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
    assert_eq!(inspect_generated_brightness(&first).await, 2);

    let prefab_list = first
        .request(SceneMethod::PrefabList, json!({}), REQUEST_TIMEOUT)
        .await
        .unwrap();
    assert!(
        prefab_list["schemaVersion"]
            .as_str()
            .is_some_and(|version| version.starts_with("1-")),
        "unexpected generated prefab catalog: {prefab_list:#?}"
    );
    assert_eq!(prefab_list["entries"].as_array().map(Vec::len), Some(1));
    assert_eq!(
        prefab_list["entries"][0]["displayName"],
        "Generated Runtime Prefab"
    );
    assert_eq!(prefab_list["entries"][0]["tags"], json!(["smoke"]));
    let generated_prefab_id = prefab_list["entries"][0]["prefabId"]
        .as_str()
        .expect("generated catalog entry must have a prefab id")
        .to_owned();

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

    // Generated direct-access bindings must survive High managed stripping in
    // the packaged IL2CPP Player and remain reversible.
    let generated_mutation_id = "smoke-generated-binding-01";
    let generated_preview = third
        .request(
            SceneMethod::Preview,
            json!({
                "expectedRevision": 4,
                "clientMutationId": generated_mutation_id,
                "commands": [{
                    "kind": "set_properties",
                    "objectId": SMOKE_OBJECT_ID,
                    "componentId": GENERATED_COMPONENT_ID,
                    "changes": [{"propertyId": "brightness", "value": 8}],
                }],
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    let generated_applied = third
        .request(
            SceneMethod::Apply,
            json!({
                "previewToken": generated_preview["previewToken"],
                "expectedRevision": 4,
                "clientMutationId": generated_mutation_id,
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(generated_applied["sceneRevision"], 5);
    assert_eq!(inspect_generated_brightness(&third).await, 8);
    let generated_undone = third
        .request(
            SceneMethod::Undo,
            json!({
                "expectedRevision": 5,
                "undoId": generated_applied["undoId"],
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(generated_undone["sceneRevision"], 6);
    assert_eq!(inspect_generated_brightness(&third).await, 2);

    // Spawn through the generated catalog's opaque, GUID-derived prefab ID.
    let spawn_mutation_id = "smoke-generated-catalog-01";
    let spawn_preview = third
        .request(
            SceneMethod::Preview,
            json!({
                "expectedRevision": 6,
                "clientMutationId": spawn_mutation_id,
                "commands": [{
                    "kind": "spawn",
                    "tempId": "tmp:generated-prefab",
                    "prefabId": generated_prefab_id,
                }],
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    let spawned = third
        .request(
            SceneMethod::Apply,
            json!({
                "previewToken": spawn_preview["previewToken"],
                "expectedRevision": 6,
                "clientMutationId": spawn_mutation_id,
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(spawned["sceneRevision"], 7);
    let spawned_object_id = spawned["tempIdMap"]["tmp:generated-prefab"]
        .as_str()
        .expect("spawn must map the temporary id")
        .to_owned();
    let staging_component_id = format!("{spawned_object_id}/staging");
    assert_eq!(
        inspect_property(
            &third,
            &spawned_object_id,
            &staging_component_id,
            "awake_identity",
        )
        .await,
        spawned_object_id
    );
    assert_eq!(
        inspect_property(
            &third,
            &spawned_object_id,
            &staging_component_id,
            "enable_identity",
        )
        .await,
        spawned_object_id
    );
    let hierarchy_with_spawn = third
        .request(SceneMethod::Hierarchy, json!({}), REQUEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(
        hierarchy_with_spawn["nodes"].as_array().map(Vec::len),
        Some(2)
    );
    let spawned_world = third
        .request(
            SceneMethod::PersistenceSave,
            json!({
                "slot": "integration-spawned",
                "expectedRevision": 7,
                "clientMutationId": "smoke-world-save-spawned-01",
                "metadata": {"label": "World with generated prefab"},
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(spawned_world["savedRevision"], 7);
    let spawn_undone = third
        .request(
            SceneMethod::Undo,
            json!({"expectedRevision": 7, "undoId": spawned["undoId"]}),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(spawn_undone["sceneRevision"], 8);
    let final_hierarchy = third
        .request(SceneMethod::Hierarchy, json!({}), REQUEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(final_hierarchy["nodes"].as_array().map(Vec::len), Some(1));

    // Active objects created by application code are discovered automatically,
    // and their destruction forms one external-mutation revision barrier.
    apply_lifecycle_command(&third, 8, "smoke-lifecycle-create-01", "create_dynamic").await;
    assert!(
        hierarchy_names(&third)
            .await
            .contains(&"BrainRegion Dynamic Lifecycle Object".to_owned())
    );
    apply_lifecycle_command(&third, 10, "smoke-lifecycle-destroy-01", "destroy_dynamic").await;
    assert!(
        !hierarchy_names(&third)
            .await
            .contains(&"BrainRegion Dynamic Lifecycle Object".to_owned())
    );

    // With includeLoadedScenes explicitly enabled, additive scene roots join and
    // leave the same registry without retaining dead Unity object references.
    apply_lifecycle_command(&third, 12, "smoke-additive-load-01", "load_additive").await;
    assert!(
        hierarchy_names(&third)
            .await
            .contains(&"BrainRegion Additive Lifecycle Object".to_owned())
    );
    apply_lifecycle_command(&third, 14, "smoke-additive-unload-01", "unload_additive").await;
    assert!(
        !hierarchy_names(&third)
            .await
            .contains(&"BrainRegion Additive Lifecycle Object".to_owned())
    );

    // WorldDocument slots are hashed and saved atomically. Only properties marked
    // persistent are captured; load is a revision-bound preview/apply mutation.
    let invalid_slot = third
        .request(
            SceneMethod::PersistenceSave,
            json!({
                "slot": "../escape",
                "expectedRevision": 16,
                "clientMutationId": "smoke-world-invalid-slot-01",
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap_err();
    assert_upstream_error(invalid_slot, -32020, "invalid_persistence_params");

    let save_params = json!({
        "slot": "integration-smoke",
        "expectedRevision": 16,
        "clientMutationId": "smoke-world-save-01",
        "metadata": {"label": "IL2CPP integration world"},
    });
    let saved = third
        .request(
            SceneMethod::PersistenceSave,
            save_params.clone(),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(saved["slot"], "integration-smoke");
    assert_eq!(saved["savedRevision"], 16);
    assert_eq!(saved["idempotentReplay"], false);
    assert!(
        saved["digest"]
            .as_str()
            .is_some_and(|digest| digest.starts_with("sha256:") && digest.len() == 71)
    );
    let save_replay = third
        .request(SceneMethod::PersistenceSave, save_params, REQUEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(save_replay["digest"], saved["digest"]);
    assert_eq!(save_replay["idempotentReplay"], true);
    let digest_conflict = third
        .request(
            SceneMethod::PersistenceSave,
            json!({
                "slot": "integration-smoke",
                "expectedRevision": 16,
                "clientMutationId": "smoke-world-save-conflict-01",
                "expectedSlotDigest": format!("sha256:{}", "0".repeat(64)),
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap_err();
    assert_upstream_error(digest_conflict, -32020, "slot_digest_conflict");

    let listed = third
        .request(SceneMethod::PersistenceList, json!({}), REQUEST_TIMEOUT)
        .await
        .unwrap();
    assert!(listed["slots"].as_array().is_some_and(|slots| {
        slots
            .iter()
            .any(|slot| slot["slot"] == "integration-smoke" && slot["digest"] == saved["digest"])
    }));

    let post_save_preview = third
        .request(
            SceneMethod::Preview,
            json!({
                "expectedRevision": 16,
                "clientMutationId": "smoke-post-save-change-01",
                "commands": [{
                    "kind": "set_properties",
                    "objectId": SMOKE_OBJECT_ID,
                    "componentId": SMOKE_COMPONENT_ID,
                    "changes": [{"propertyId": "counter", "value": 41}],
                }],
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    let post_save_applied = third
        .request(
            SceneMethod::Apply,
            json!({
                "previewToken": post_save_preview["previewToken"],
                "expectedRevision": 16,
                "clientMutationId": "smoke-post-save-change-01",
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(post_save_applied["sceneRevision"], 17);
    assert_eq!(inspect_counter(&third).await, 41);

    let load_preview = third
        .request(
            SceneMethod::PersistenceLoadPreview,
            json!({
                "slot": "integration-smoke",
                "expectedRevision": 17,
                "clientMutationId": "smoke-world-load-01",
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(load_preview["baseRevision"], 17);
    assert_eq!(load_preview["summary"]["entities"], 1);
    assert_eq!(load_preview["summary"]["create"], 0);
    // Preview is side-effect free.
    assert_eq!(inspect_counter(&third).await, 41);

    let load_params = json!({
        "previewToken": load_preview["previewToken"],
        "expectedRevision": 17,
        "clientMutationId": "smoke-world-load-01",
    });
    let loaded = third
        .request(
            SceneMethod::PersistenceLoad,
            load_params.clone(),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(loaded["sceneRevision"], 18);
    assert_eq!(loaded["digest"], saved["digest"]);
    assert_eq!(loaded["idempotentReplay"], false);
    assert_eq!(inspect_counter(&third).await, 1);
    let load_replay = third
        .request(SceneMethod::PersistenceLoad, load_params, REQUEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(load_replay["sceneRevision"], 18);
    assert_eq!(load_replay["idempotentReplay"], true);
    assert_eq!(inspect_counter(&third).await, 1);

    // A second document was saved while the generated prefab existed. Loading it
    // after Undo must recreate the exact objectId from the catalog and still stage
    // Awake/OnEnable behind identity assignment.
    let spawned_load_preview = third
        .request(
            SceneMethod::PersistenceLoadPreview,
            json!({
                "slot": "integration-spawned",
                "expectedRevision": 18,
                "clientMutationId": "smoke-world-load-spawned-01",
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(spawned_load_preview["summary"]["create"], 1);
    let spawned_loaded = third
        .request(
            SceneMethod::PersistenceLoad,
            json!({
                "previewToken": spawned_load_preview["previewToken"],
                "expectedRevision": 18,
                "clientMutationId": "smoke-world-load-spawned-01",
            }),
            REQUEST_TIMEOUT,
        )
        .await
        .unwrap();
    assert_eq!(spawned_loaded["sceneRevision"], 19);
    assert_eq!(hierarchy_names(&third).await.len(), 2);
    assert_eq!(
        inspect_property(
            &third,
            &spawned_object_id,
            &staging_component_id,
            "awake_identity",
        )
        .await,
        spawned_object_id
    );
    assert_eq!(
        inspect_property(
            &third,
            &spawned_object_id,
            &staging_component_id,
            "enable_identity",
        )
        .await,
        spawned_object_id
    );

    registry.close_all().await;
    listener.shutdown().await.unwrap();
    if player.try_wait().unwrap().is_none() {
        player.start_kill().unwrap();
    }
    tokio::time::timeout(Duration::from_secs(30), player.wait())
        .await
        .expect("Unity smoke Player did not exit after kill")
        .expect("could not wait for Unity smoke Player");
}
