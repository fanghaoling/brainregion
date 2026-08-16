use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use brainregiond::config::DaemonConfig;
use brainregiond::server::probe;
use brainregiond::supervisor::McpSupervisor;

/// Real repository smoke test. It is opt-in because it requires the Python
/// virtual environment and imports the full BrainRegion application.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires the repository Python .venv"]
async fn connects_to_the_real_brainregion_mcp_worker() {
    let config = DaemonConfig::discover().expect("discover daemon configuration");
    let mut supervisor = McpSupervisor::connect(&config)
        .await
        .expect("connect to Python MCP worker");

    let report = probe(&mut supervisor).await.expect("probe BrainRegion MCP");
    assert_eq!(report["ok"], true);
    assert_eq!(report["status"], "ready");
    assert_eq!(report["mcp"]["protocolVersion"], "2025-11-25");
    assert_eq!(report["mcp"]["brainregion"]["name"], "brainregion");
    assert!(report["mcp"]["brainregion"]["version"].is_string());
    assert!(report["mcp"]["toolCount"].as_u64().unwrap_or_default() > 0);

    supervisor.shutdown().await.expect("shutdown MCP worker");
}

/// Regression test for Tokio's non-cancellable stdio implementation: the
/// daemon must exit after its shutdown response even while the parent keeps
/// the stdin pipe handle open.
#[test]
#[ignore = "requires the repository Python .venv"]
fn shutdown_exits_while_parent_keeps_stdin_open() {
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace root");
    let mut child = Command::new(env!("CARGO_BIN_EXE_brainregiond"))
        .arg("serve")
        .current_dir(repository)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("start brainregiond");
    let mut stdin = child.stdin.take().expect("daemon stdin");
    let mut stdout = BufReader::new(child.stdout.take().expect("daemon stdout"));

    let mut ready = String::new();
    stdout.read_line(&mut ready).expect("read ready message");
    let ready: serde_json::Value = serde_json::from_str(&ready).expect("parse ready message");
    assert_eq!(ready["method"], "daemon/ready");

    writeln!(
        stdin,
        "{{\"jsonrpc\":\"2.0\",\"id\":\"shutdown-1\",\"method\":\"daemon/shutdown\"}}"
    )
    .expect("write shutdown request");
    stdin.flush().expect("flush shutdown request");

    let mut response = String::new();
    stdout
        .read_line(&mut response)
        .expect("read shutdown response");
    let response: serde_json::Value =
        serde_json::from_str(&response).expect("parse shutdown response");
    assert_eq!(response["id"], "shutdown-1");
    assert_eq!(response["result"]["accepted"], true);

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = child.try_wait().expect("poll daemon status") {
            assert!(status.success(), "daemon exited with {status}");
            break;
        }
        if Instant::now() >= deadline {
            drop(stdin);
            let _ = child.kill();
            let _ = child.wait();
            panic!("daemon stayed alive while the parent kept stdin open");
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    // Keep this handle alive until after observing process exit; dropping it
    // earlier would hide the regression this test protects against.
    drop(stdin);
}
