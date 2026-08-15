use std::future::Future;
use std::io;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use rmcp::RoleClient;
use rmcp::service::{RxJsonRpcMessage, TxJsonRpcMessage};
use rmcp::transport::Transport;
use rmcp::transport::async_rw::AsyncRwTransport;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

const GRACEFUL_EXIT_TIMEOUT: Duration = Duration::from_secs(3);
const FORCED_EXIT_TIMEOUT: Duration = Duration::from_secs(3);

/// A cloneable owner for the Python process handle. The handle remains inside
/// the controller while any async wait is in progress, so cancelling a future
/// cannot lose the ability to kill and reap the child later.
#[derive(Clone)]
pub(crate) struct ChildController {
    child: Arc<Mutex<Option<Child>>>,
}

impl ChildController {
    fn new(child: Child) -> Self {
        Self {
            child: Arc::new(Mutex::new(Some(child))),
        }
    }

    pub(crate) async fn graceful_reap(&self) -> io::Result<()> {
        let mut child = self.child.lock().await;
        let Some(process) = child.as_mut() else {
            return Ok(());
        };

        match tokio::time::timeout(GRACEFUL_EXIT_TIMEOUT, process.wait()).await {
            Ok(Ok(_status)) => {
                child.take();
                return Ok(());
            }
            Ok(Err(error)) => return Err(error),
            Err(_) => {}
        }

        force_process_exit(process).await?;
        child.take();
        Ok(())
    }

    pub(crate) async fn force_reap(&self) -> io::Result<()> {
        let mut child = self.child.lock().await;
        let Some(process) = child.as_mut() else {
            return Ok(());
        };

        force_process_exit(process).await?;
        child.take();
        Ok(())
    }

    #[cfg(test)]
    async fn is_reaped(&self) -> bool {
        self.child.lock().await.is_none()
    }
}

async fn force_process_exit(process: &mut Child) -> io::Result<()> {
    match tokio::time::timeout(FORCED_EXIT_TIMEOUT, process.kill()).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(kill_error)) => match process.try_wait()? {
            Some(_) => Ok(()),
            None => Err(kill_error),
        },
        Err(_) => Err(io::Error::new(
            io::ErrorKind::TimedOut,
            format!(
                "child did not exit within {} ms after kill",
                FORCED_EXIT_TIMEOUT.as_millis()
            ),
        )),
    }
}

pub(crate) struct SupervisedChildTransport {
    io: AsyncRwTransport<RoleClient, ChildStdout, ChildStdin>,
    controller: ChildController,
}

impl SupervisedChildTransport {
    pub(crate) async fn spawn(mut command: Command) -> io::Result<(Self, ChildController)> {
        let mut child = command
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = match child.stdin.take() {
            Some(stdin) => stdin,
            None => {
                let _ = child.kill().await;
                return Err(io::Error::other("MCP child stdin pipe is unavailable"));
            }
        };
        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                drop(stdin);
                let _ = child.kill().await;
                return Err(io::Error::other("MCP child stdout pipe is unavailable"));
            }
        };

        let controller = ChildController::new(child);
        let transport = Self {
            io: AsyncRwTransport::new_client(stdout, stdin),
            controller: controller.clone(),
        };
        Ok((transport, controller))
    }
}

impl Transport<RoleClient> for SupervisedChildTransport {
    type Error = io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        self.io.send(item)
    }

    fn receive(&mut self) -> impl Future<Output = Option<RxJsonRpcMessage<RoleClient>>> + Send {
        self.io.receive()
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        self.io.close().await?;
        self.controller.graceful_reap().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    fn sleeping_command() -> Command {
        let mut command = Command::new("powershell.exe");
        command.args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Start-Sleep -Seconds 30",
        ]);
        command
    }

    #[cfg(unix)]
    fn sleeping_command() -> Command {
        let mut command = Command::new("sleep");
        command.arg("30");
        command
    }

    #[tokio::test]
    async fn controller_can_reap_a_transport_dropped_during_initialization() {
        let (transport, controller) = SupervisedChildTransport::spawn(sleeping_command())
            .await
            .unwrap();
        drop(transport);

        tokio::time::timeout(Duration::from_secs(2), controller.force_reap())
            .await
            .expect("forced cleanup should be bounded")
            .unwrap();
        assert!(controller.is_reaped().await);
    }
}
