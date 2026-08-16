use std::time::Duration;

use brainregiond::BrainregiondError;
use brainregiond::protocol::read_bounded_line_async;
use brainregiond::scene_peer::{
    SceneMethod, ScenePeerAuth, ScenePeerRegistry, ScenePeerState, accept_scene_peer,
};
use brainregiond::scene_rpc::{MAX_SCENE_FRAME_BYTES, SceneCapability};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

const TEST_TIMEOUT: Duration = Duration::from_secs(2);

fn registration(session_id: &str, revision: u64) -> Value {
    let mut value: Value = serde_json::from_str(include_str!(
        "../../../schemas/scene-rpc/v1/examples/runtime-register.json"
    ))
    .unwrap();
    value["params"]["sessionId"] = json!(session_id);
    value["params"]["instanceId"] = json!(format!("instance-{session_id}"));
    value["params"]["sceneRevision"] = json!(revision);
    value
}

async fn write_frame<W: AsyncWrite + Unpin>(writer: &mut W, value: &Value) {
    let mut encoded = serde_json::to_vec(value).unwrap();
    encoded.push(b'\n');
    writer.write_all(&encoded).await.unwrap();
    writer.flush().await.unwrap();
}

async fn read_frame<R: AsyncRead + Unpin>(reader: &mut BufReader<R>) -> Option<Value> {
    read_bounded_line_async(reader, MAX_SCENE_FRAME_BYTES)
        .await
        .unwrap()
        .map(|line| serde_json::from_str(&line).unwrap())
}

#[tokio::test]
async fn registers_calls_and_tracks_scene_change_notifications() {
    let registry = ScenePeerRegistry::default();
    let auth = ScenePeerAuth::new("paired-player", [SceneCapability::SceneRead]).unwrap();
    let (daemon_stream, player_stream) = tokio::io::duplex(64 * 1024);

    let player = tokio::spawn(async move {
        let (read_half, mut write_half) = tokio::io::split(player_stream);
        let mut reader = BufReader::new(read_half);
        write_frame(&mut write_half, &registration("session-a", 0)).await;

        let request = read_frame(&mut reader).await.unwrap();
        assert_eq!(request["method"], "runtime/info");
        assert!(request["deadlineUnixMs"].as_u64().unwrap() > 1_000_000);
        write_frame(
            &mut write_half,
            &json!({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"status": "ready", "sceneRevision": 0}
            }),
        )
        .await;
        write_frame(
            &mut write_half,
            &json!({
                "jsonrpc": "2.0",
                "method": "scene/changed",
                "params": {
                    "sceneRevision": 1,
                    "clientMutationId": "mutation-1",
                    "summary": "set lamp intensity"
                }
            }),
        )
        .await;
        write_frame(
            &mut write_half,
            &json!({
                "jsonrpc": "2.0",
                "method": "scene/changed",
                "params": {
                    "sceneRevision": 2,
                    "summary": "external persistent mutation"
                }
            }),
        )
        .await;

        assert!(read_frame(&mut reader).await.is_none());
    });

    let handle = accept_scene_peer(registry.clone(), auth, daemon_stream, TEST_TIMEOUT)
        .await
        .unwrap();
    let mut events = handle.subscribe_events();
    let result = handle
        .request(SceneMethod::RuntimeInfo, json!({}), TEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(result["status"], "ready");

    let event = tokio::time::timeout(TEST_TIMEOUT, events.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(event.scene_revision, 1);
    assert_eq!(event.client_mutation_id.as_deref(), Some("mutation-1"));
    let event = tokio::time::timeout(TEST_TIMEOUT, events.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(event.scene_revision, 2);
    assert_eq!(event.client_mutation_id, None);
    assert_eq!(handle.snapshot().scene_revision, 2);
    assert_eq!(registry.snapshots().await.len(), 1);

    handle.close();
    tokio::time::timeout(TEST_TIMEOUT, player)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn newer_connection_epoch_supersedes_the_previous_session() {
    let registry = ScenePeerRegistry::default();
    let auth = ScenePeerAuth::new("paired-player", [SceneCapability::SceneRead]).unwrap();

    let (daemon_a, player_a) = tokio::io::duplex(16 * 1024);
    let player_a = tokio::spawn(async move {
        let (read_half, mut write_half) = tokio::io::split(player_a);
        let mut reader = BufReader::new(read_half);
        write_frame(&mut write_half, &registration("session-a", 4)).await;
        assert!(read_frame(&mut reader).await.is_none());
    });
    let first = accept_scene_peer(registry.clone(), auth.clone(), daemon_a, TEST_TIMEOUT)
        .await
        .unwrap();

    let (daemon_b, player_b) = tokio::io::duplex(16 * 1024);
    let player_b = tokio::spawn(async move {
        let (read_half, mut write_half) = tokio::io::split(player_b);
        let mut reader = BufReader::new(read_half);
        write_frame(&mut write_half, &registration("session-b", 5)).await;
        assert!(read_frame(&mut reader).await.is_none());
    });
    let second = accept_scene_peer(registry.clone(), auth, daemon_b, TEST_TIMEOUT)
        .await
        .unwrap();

    assert_eq!(first.connection_epoch(), 1);
    assert_eq!(first.state(), ScenePeerState::Superseded);
    assert_eq!(second.connection_epoch(), 2);
    let snapshots = registry.snapshots().await;
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].session_id, "session-b");

    second.close();
    tokio::time::timeout(TEST_TIMEOUT, player_a)
        .await
        .unwrap()
        .unwrap();
    tokio::time::timeout(TEST_TIMEOUT, player_b)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn timed_out_request_is_not_retried_and_late_response_is_ignored() {
    let registry = ScenePeerRegistry::default();
    let auth = ScenePeerAuth::new("slow-player", [SceneCapability::SceneRead]).unwrap();
    let (daemon_stream, player_stream) = tokio::io::duplex(64 * 1024);

    let player = tokio::spawn(async move {
        let (read_half, mut write_half) = tokio::io::split(player_stream);
        let mut reader = BufReader::new(read_half);
        write_frame(&mut write_half, &registration("slow-session", 0)).await;

        let first = read_frame(&mut reader).await.unwrap();
        tokio::time::sleep(Duration::from_millis(120)).await;
        write_frame(
            &mut write_half,
            &json!({"jsonrpc": "2.0", "id": first["id"], "result": {"late": true}}),
        )
        .await;

        let second = read_frame(&mut reader).await.unwrap();
        write_frame(
            &mut write_half,
            &json!({"jsonrpc": "2.0", "id": second["id"], "result": {"ok": true}}),
        )
        .await;
        assert!(read_frame(&mut reader).await.is_none());
    });

    let handle = accept_scene_peer(registry, auth, daemon_stream, TEST_TIMEOUT)
        .await
        .unwrap();
    let error = handle
        .request(
            SceneMethod::RuntimeInfo,
            json!({}),
            Duration::from_millis(20),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("timed out"));

    let result = handle
        .request(SceneMethod::RuntimeInfo, json!({}), TEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(result["ok"], true);
    assert_eq!(handle.state(), ScenePeerState::Connected);

    handle.close();
    tokio::time::timeout(TEST_TIMEOUT, player)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn capability_checks_and_disconnects_fail_closed() {
    let registry = ScenePeerRegistry::default();
    let auth = ScenePeerAuth::new("read-only-player", [SceneCapability::SceneRead]).unwrap();
    let (daemon_stream, player_stream) = tokio::io::duplex(32 * 1024);

    let player = tokio::spawn(async move {
        let (read_half, mut write_half) = tokio::io::split(player_stream);
        let mut reader = BufReader::new(read_half);
        write_frame(&mut write_half, &registration("read-only-session", 0)).await;
        let request = read_frame(&mut reader).await.unwrap();
        assert_eq!(request["method"], "runtime/info");
        // Dropping the stream makes the pending request outcome explicit.
    });

    let handle = accept_scene_peer(registry, auth, daemon_stream, TEST_TIMEOUT)
        .await
        .unwrap();
    let denied = handle
        .request(
            SceneMethod::Preview,
            json!({"expectedRevision": 0, "clientMutationId": "m", "commands": []}),
            TEST_TIMEOUT,
        )
        .await
        .unwrap_err();
    match denied {
        BrainregiondError::Upstream(fault) => {
            assert_eq!(fault["code"], -32002);
            assert_eq!(fault["data"]["retryable"], false);
        }
        other => panic!("expected structured capability denial, got {other}"),
    }

    let disconnected = handle
        .request(SceneMethod::RuntimeInfo, json!({}), TEST_TIMEOUT)
        .await
        .unwrap_err();
    assert!(disconnected.to_string().contains("EOF") || disconnected.to_string().contains("ended"));
    assert_eq!(handle.state(), ScenePeerState::Disconnected);
    player.await.unwrap();
}

#[tokio::test]
async fn mutating_request_disconnect_is_an_unknown_non_retryable_outcome() {
    let registry = ScenePeerRegistry::default();
    let auth = ScenePeerAuth::new("write-player", [SceneCapability::SceneWrite]).unwrap();
    let (daemon_stream, player_stream) = tokio::io::duplex(32 * 1024);

    let player = tokio::spawn(async move {
        let (read_half, mut write_half) = tokio::io::split(player_stream);
        let mut reader = BufReader::new(read_half);
        write_frame(&mut write_half, &registration("write-session", 0)).await;
        let request = read_frame(&mut reader).await.unwrap();
        assert_eq!(request["method"], "scene/apply");
        // The Player may have committed before the stream disappeared.
    });

    let handle = accept_scene_peer(registry, auth, daemon_stream, TEST_TIMEOUT)
        .await
        .unwrap();
    let error = handle
        .request(
            SceneMethod::Apply,
            json!({
                "previewToken": "preview-1",
                "expectedRevision": 0,
                "clientMutationId": "mutation-1",
            }),
            TEST_TIMEOUT,
        )
        .await
        .unwrap_err();
    match error {
        BrainregiondError::Upstream(fault) => {
            assert_eq!(fault["code"], -32011);
            assert_eq!(fault["data"]["outcome"], "unknown");
            assert_eq!(fault["data"]["retryable"], false);
        }
        other => panic!("expected structured unknown outcome, got {other}"),
    }
    player.await.unwrap();
}

#[tokio::test]
async fn rejects_policy_grants_that_runtime_did_not_advertise() {
    let registry = ScenePeerRegistry::default();
    let auth = ScenePeerAuth::new("limited-player", [SceneCapability::LogsRead]).unwrap();
    let (daemon_stream, mut player_stream) = tokio::io::duplex(16 * 1024);
    let mut limited = registration("limited-session", 0);
    limited["params"]["capabilities"] = json!(["scene.read"]);
    write_frame(&mut player_stream, &limited).await;

    let error = accept_scene_peer(registry, auth, daemon_stream, TEST_TIMEOUT)
        .await
        .err()
        .expect("unsupported grant must be rejected");
    assert!(error.to_string().contains("unsupported"));
}
