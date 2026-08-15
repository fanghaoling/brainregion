#![cfg(windows)]

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use brainregiond::config::{PairingSecret, ScenePipeConfig};
use brainregiond::protocol::read_bounded_line_async;
use brainregiond::scene_pairing::{PairingChallengeNotification, build_pairing_proof};
use brainregiond::scene_peer::{SceneMethod, ScenePeerRegistry};
use brainregiond::scene_pipe::ScenePipeListener;
use brainregiond::scene_rpc::{
    MAX_SCENE_FRAME_BYTES, RuntimeRegistrationNotification, SceneCapability,
};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::windows::named_pipe::{ClientOptions, ServerOptions};

const TEST_TIMEOUT: Duration = Duration::from_secs(3);
const SECRET: &[u8] = b"0123456789abcdef0123456789abcdef";

fn unique_pipe_name() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("brainregion.scene.test.{}.{}", std::process::id(), nanos)
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
async fn current_user_pipe_pairs_and_proxies_a_runtime_request() {
    let registry = ScenePeerRegistry::default();
    let config = ScenePipeConfig {
        name: unique_pipe_name(),
        principal_id: "pipe-test-player".to_owned(),
        pairing_secret: PairingSecret::new(SECRET).unwrap(),
        granted_capabilities: vec![SceneCapability::SceneRead],
        max_connections: 2,
        authentication_timeout: TEST_TIMEOUT,
    };
    let listener = ScenePipeListener::start(config, registry.clone()).unwrap();
    let pipe_path = listener.pipe_path().to_owned();

    let collision = ServerOptions::new()
        .first_pipe_instance(true)
        .create(&pipe_path)
        .unwrap_err();
    assert!(matches!(
        collision.kind(),
        std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::AlreadyExists
    ));

    let player_path = pipe_path.clone();
    let player = tokio::spawn(async move {
        let stream = ClientOptions::new().open(player_path).unwrap();
        let (read_half, mut write_half) = tokio::io::split(stream);
        let mut reader = BufReader::new(read_half);
        let challenge: PairingChallengeNotification =
            serde_json::from_value(read_frame(&mut reader).await.unwrap()).unwrap();

        let fixture = include_str!("../../../schemas/scene-rpc/v1/examples/runtime-register.json");
        let mut registration: RuntimeRegistrationNotification =
            serde_json::from_str(fixture).unwrap();
        registration.params.instance_id = "pipe-instance".to_owned();
        registration.params.session_id = "pipe-session".to_owned();
        registration.params.capabilities = vec![SceneCapability::SceneRead];
        registration.params.pairing_proof =
            Some(build_pairing_proof(SECRET, &challenge.params, &registration.params).unwrap());
        write_frame(
            &mut write_half,
            &serde_json::to_value(registration).unwrap(),
        )
        .await;

        let request = read_frame(&mut reader).await.unwrap();
        assert_eq!(request["method"], "runtime/info");
        write_frame(
            &mut write_half,
            &json!({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"status": "ready", "sceneRevision": 0}
            }),
        )
        .await;
        assert!(read_frame(&mut reader).await.is_none());
    });

    let handle = tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            if let Some(handle) = registry.get("pipe-test-player").await {
                break handle;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let result = handle
        .request(SceneMethod::RuntimeInfo, json!({}), TEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(result["status"], "ready");

    registry.close_all().await;
    tokio::time::timeout(TEST_TIMEOUT, player)
        .await
        .unwrap()
        .unwrap();
    listener.shutdown().await.unwrap();
}
