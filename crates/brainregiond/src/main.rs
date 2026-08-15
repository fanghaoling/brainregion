use std::env;
use std::process::ExitCode;

use brainregiond::config::{DaemonConfig, RunMode, usage};
use brainregiond::scene_peer::ScenePeerRegistry;
use brainregiond::scene_pipe::ScenePipeListener;
use brainregiond::server::{probe, serve_stdio_until_with_scene_peers, termination_signal};
use brainregiond::supervisor::McpSupervisor;
use brainregiond::{CONTROL_SCHEMA_JSON, DAEMON_NAME, DAEMON_VERSION, Result, SCENE_SCHEMA_JSON};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{DAEMON_NAME}: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    let mut config = DaemonConfig::discover()?;
    config.apply_args(env::args_os().skip(1))?;

    match config.mode {
        RunMode::Help => {
            println!("{}", usage());
            return Ok(());
        }
        RunMode::Version => {
            println!("{DAEMON_NAME} {DAEMON_VERSION}");
            return Ok(());
        }
        RunMode::Schema => {
            println!("{CONTROL_SCHEMA_JSON}");
            return Ok(());
        }
        RunMode::SceneSchema => {
            println!("{SCENE_SCHEMA_JSON}");
            return Ok(());
        }
        RunMode::Serve | RunMode::Probe => {}
    }

    eprintln!(
        "{DAEMON_NAME} {DAEMON_VERSION} starting MCP child {:?}",
        config.mcp.program
    );
    let mut termination = Box::pin(termination_signal());
    let mut supervisor = tokio::select! {
        result = McpSupervisor::connect(&config) => result?,
        signal_result = &mut termination => {
            signal_result?;
            // RMCP drops an initializing child asynchronously; keep the runtime
            // alive briefly so its kill task can be polled.
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            return Ok(());
        }
    };

    match config.mode {
        RunMode::Probe => {
            let probe_result: Result<Option<String>> = tokio::select! {
                result = async {
                    let report = probe(&mut supervisor).await?;
                    serde_json::to_string_pretty(&report).map(Some).map_err(Into::into)
                } => result,
                signal_result = &mut termination => {
                    signal_result.map(|()| None).map_err(Into::into)
                }
            };
            let cleanup_result = supervisor.shutdown().await;
            if let Some(rendered) = combine_probe_and_cleanup(probe_result, cleanup_result)? {
                println!("{rendered}");
            }
        }
        RunMode::Serve => {
            let scene_peers = ScenePeerRegistry::default();
            let scene_pipe = match config.scene_pipe.clone() {
                Some(pipe_config) => {
                    match ScenePipeListener::start(pipe_config, scene_peers.clone()) {
                        Ok(listener) => {
                            eprintln!(
                                "{DAEMON_NAME}: Runtime scene pipe listening at {}",
                                listener.pipe_path()
                            );
                            Some(listener)
                        }
                        Err(error) => {
                            let cleanup = supervisor.shutdown().await;
                            return combine_probe_and_cleanup(Err(error), cleanup);
                        }
                    }
                }
                None => None,
            };

            let serve_result = if let Some(listener) = &scene_pipe {
                let listener_failure = listener.failure_signal();
                let shutdown = async move {
                    tokio::select! {
                        result = &mut termination => result,
                        result = listener_failure => result,
                    }
                };
                serve_stdio_until_with_scene_peers(supervisor, scene_peers, shutdown).await
            } else {
                serve_stdio_until_with_scene_peers(supervisor, scene_peers, termination).await
            };
            let pipe_cleanup = match scene_pipe {
                Some(listener) => listener.shutdown().await,
                None => Ok(()),
            };
            combine_probe_and_cleanup(serve_result, pipe_cleanup)?;
        }
        RunMode::Schema | RunMode::SceneSchema | RunMode::Help | RunMode::Version => unreachable!(),
    }
    Ok(())
}

fn combine_probe_and_cleanup<T>(operation: Result<T>, cleanup: Result<()>) -> Result<T> {
    match (operation, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(cleanup_error)) => Err(brainregiond::BrainregiondError::Protocol(
            format!("{error}; MCP cleanup also failed: {cleanup_error}"),
        )),
    }
}
