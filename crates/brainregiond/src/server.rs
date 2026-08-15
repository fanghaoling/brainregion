use std::collections::VecDeque;
use std::future::Future;
use std::io;
use std::time::Instant;

use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufRead, AsyncWrite, AsyncWriteExt, BufReader, DuplexStream};

use crate::error::{BrainregiondError, Result};
use crate::protocol::{
    MAX_CONTROL_FRAME_BYTES, MAX_CONTROL_OUTPUT_BYTES, Request, RpcFault, failure, notification,
    parse_request, read_bounded_line_async, success,
};
use crate::supervisor::{BrainregionPing, McpBackend, McpSupervisor};
use crate::{CONTROL_PROTOCOL_VERSION, DAEMON_NAME, DAEMON_VERSION};

const MAX_PENDING_REQUESTS: usize = 32;

struct ControlServer<B> {
    backend: B,
    started_at: Instant,
    request_count: u64,
}

impl<B: McpBackend> ControlServer<B> {
    fn new(backend: B) -> Self {
        Self {
            backend,
            started_at: Instant::now(),
            request_count: 0,
        }
    }

    fn ready_notification(&self) -> Value {
        notification(
            "daemon/ready",
            json!({
                "protocolVersion": CONTROL_PROTOCOL_VERSION,
                "daemon": {
                    "name": DAEMON_NAME,
                    "version": DAEMON_VERSION,
                },
                "status": self.backend.state().as_str(),
                "mcp": self.backend.metadata(),
            }),
        )
    }

    async fn dispatch(&mut self, request: Request) -> (Value, bool) {
        self.request_count = self.request_count.saturating_add(1);
        let id = request.id;

        if matches!(
            request.method.as_str(),
            "daemon/info" | "daemon/health" | "mcp/tools/list" | "daemon/shutdown"
        ) && !request.params.as_object().is_some_and(Map::is_empty)
        {
            return (
                failure(
                    id,
                    RpcFault::invalid_params(format!(
                        "{} params must be an empty object",
                        request.method
                    )),
                ),
                false,
            );
        }

        match request.method.as_str() {
            "daemon/info" => (
                success(
                    id,
                    json!({
                        "protocolVersion": CONTROL_PROTOCOL_VERSION,
                        "daemon": {
                            "name": DAEMON_NAME,
                            "version": DAEMON_VERSION,
                            "uptimeMs": self.started_at.elapsed().as_millis(),
                            "requestCount": self.request_count,
                        },
                        "status": self.backend.state().as_str(),
                        "mcp": self.backend.metadata(),
                    }),
                ),
                false,
            ),
            "daemon/health" => match self.backend.ping().await {
                Ok(ping) => (success(id, self.health_value(ping)), false),
                Err(error) => (failure(id, RpcFault::internal(error.to_string())), false),
            },
            "mcp/tools/list" => match self.backend.list_tools_value().await {
                Ok(value) => (success(id, value), false),
                Err(error) => (failure(id, RpcFault::internal(error.to_string())), false),
            },
            "mcp/tools/call" => match parse_tool_call_params(request.params) {
                Ok((name, arguments)) => match self.backend.call_tool_value(&name, arguments).await
                {
                    Ok(value) => (success(id, value), false),
                    Err(error) => (failure(id, tool_call_fault(error)), false),
                },
                Err(fault) => (failure(id, fault), false),
            },
            "daemon/shutdown" => {
                let response = match self.backend.shutdown().await {
                    Ok(()) => success(id, json!({"accepted": true})),
                    Err(error) => failure(id, RpcFault::internal(error.to_string())),
                };
                (response, true)
            }
            method => (failure(id, RpcFault::method_not_found(method)), false),
        }
    }

    fn health_value(&self, ping: BrainregionPing) -> Value {
        let metadata = self.backend.metadata();
        json!({
            "ok": true,
            "status": self.backend.state().as_str(),
            "daemon": {
                "name": DAEMON_NAME,
                "version": DAEMON_VERSION,
                "uptimeMs": self.started_at.elapsed().as_millis(),
            },
            "mcp": {
                "protocolVersion": metadata.protocol_version,
                "serverInfo": metadata.server_info,
                "toolCount": metadata.tool_count,
                "brainregion": ping,
            },
        })
    }

    #[cfg(test)]
    async fn serve<R, W>(&mut self, reader: R, writer: W) -> Result<()>
    where
        R: AsyncBufRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        self.serve_until(reader, writer, std::future::pending())
            .await
    }

    async fn serve_until<R, W, S>(
        &mut self,
        mut reader: R,
        mut writer: W,
        shutdown_signal: S,
    ) -> Result<()>
    where
        R: AsyncBufRead + Unpin,
        W: AsyncWrite + Unpin,
        S: Future<Output = io::Result<()>>,
    {
        tokio::pin!(shutdown_signal);

        let run_result = async {
            write_message(&mut writer, &self.ready_notification()).await?;
            let mut pending = VecDeque::new();

            loop {
                let request = if let Some(request) = pending.pop_front() {
                    request
                } else {
                    let request = tokio::select! {
                        signal_result = &mut shutdown_signal => {
                            signal_result?;
                            break;
                        }
                        request = read_next_request(&mut reader, &mut writer) => request?,
                    };
                    let Some(request) = request else {
                        break;
                    };
                    request
                };

                let interrupted_id = request.id.clone();
                let interrupted_method = request.method.clone();
                let dispatch_outcome = if interrupted_method == "daemon/shutdown" {
                    // Once bounded MCP cleanup starts, never drop that future. A
                    // second shutdown or OS signal cannot turn detached cleanup
                    // into a false "stopped" result.
                    DispatchEvent::Completed(self.dispatch(request).await)
                } else {
                    let dispatch = self.dispatch(request);
                    tokio::pin!(dispatch);

                    loop {
                        let event = tokio::select! {
                            biased;
                            signal_result = &mut shutdown_signal => {
                                signal_result?;
                                DispatchEvent::Interrupted
                            }
                            result = &mut dispatch => DispatchEvent::Completed(result),
                            request = read_next_request(&mut reader, &mut writer) => {
                                match request? {
                                    Some(request) if is_shutdown_request(&request) => {
                                        DispatchEvent::Shutdown(request)
                                    }
                                    Some(request) => {
                                        if pending.len() < MAX_PENDING_REQUESTS {
                                            pending.push_back(request);
                                        } else {
                                            write_message(
                                                &mut writer,
                                                &failure(request.id, RpcFault {
                                                    code: -32003,
                                                    message: "control request queue is full".to_owned(),
                                                    data: Some(json!({"retryable": true})),
                                                }),
                                            ).await?;
                                        }
                                        continue;
                                    }
                                    None => DispatchEvent::Interrupted,
                                }
                            }
                        };
                        break event;
                    }
                };

                match dispatch_outcome {
                    DispatchEvent::Completed((response, should_shutdown)) => {
                        write_message(&mut writer, &response).await?;
                        if should_shutdown {
                            break;
                        }
                    }
                    DispatchEvent::Shutdown(shutdown_request) => {
                        write_message(
                            &mut writer,
                            &failure(
                                interrupted_id,
                                interrupted_fault(&interrupted_method),
                            ),
                        )
                        .await?;
                        for pending_request in pending.drain(..) {
                            write_message(
                                &mut writer,
                                &failure(
                                    pending_request.id,
                                    interrupted_fault(&pending_request.method),
                                ),
                            )
                            .await?;
                        }
                        let (response, _) = self.dispatch(shutdown_request).await;
                        write_message(&mut writer, &response).await?;
                        break;
                    }
                    DispatchEvent::Interrupted => {
                        break;
                    }
                }
            }
            Ok(())
        }
        .await;

        let cleanup_result = if self.backend.state() == crate::supervisor::McpState::Stopped {
            Ok(())
        } else {
            self.backend.shutdown().await
        };
        combine_run_and_cleanup(run_result, cleanup_result)
    }
}

enum DispatchEvent {
    Completed((Value, bool)),
    Shutdown(Request),
    Interrupted,
}

fn is_shutdown_request(request: &Request) -> bool {
    request.method == "daemon/shutdown" && request.params.as_object().is_some_and(Map::is_empty)
}

fn interrupted_fault(method: &str) -> RpcFault {
    RpcFault {
        code: -32002,
        message: format!("request {method:?} was interrupted by daemon shutdown"),
        data: Some(json!({
            "outcome": "unknown",
            "retryable": false,
        })),
    }
}

async fn read_next_request<R, W>(reader: &mut R, writer: &mut W) -> Result<Option<Request>>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    loop {
        let line = match read_bounded_line_async(reader, MAX_CONTROL_FRAME_BYTES).await {
            Ok(Some(line)) => line,
            Ok(None) => return Ok(None),
            Err(error) if error.kind() == io::ErrorKind::InvalidData => {
                write_message(
                    writer,
                    &failure(Value::Null, RpcFault::parse(error.to_string())),
                )
                .await?;
                return Ok(None);
            }
            Err(error) => return Err(BrainregiondError::Io(error)),
        };
        if line.trim().is_empty() {
            continue;
        }

        match parse_request(&line) {
            Ok(request) => return Ok(Some(request)),
            Err(error) => {
                write_message(writer, &failure(error.id, error.fault)).await?;
            }
        }
    }
}

fn tool_call_fault(error: BrainregiondError) -> RpcFault {
    match error {
        error @ BrainregiondError::Timeout { .. } => RpcFault {
            code: -32001,
            message: format!("{error}; cancellation was requested but completion is unknown"),
            data: Some(json!({
                "outcome": "unknown",
                "retryable": false,
            })),
        },
        error => RpcFault::internal(error.to_string()),
    }
}

fn parse_tool_call_params(
    params: Value,
) -> std::result::Result<(String, Map<String, Value>), RpcFault> {
    let object = params
        .as_object()
        .ok_or_else(|| RpcFault::invalid_params("mcp/tools/call params must be an object"))?;
    if let Some(field) = object
        .keys()
        .find(|field| !matches!(field.as_str(), "name" | "arguments"))
    {
        return Err(RpcFault::invalid_params(format!(
            "unexpected mcp/tools/call field {field:?}"
        )));
    }
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| RpcFault::invalid_params("mcp/tools/call requires a non-empty name"))?
        .to_owned();
    let arguments = match object.get("arguments") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(arguments)) => arguments.clone(),
        Some(_) => {
            return Err(RpcFault::invalid_params(
                "mcp/tools/call arguments must be an object",
            ));
        }
    };
    Ok((name, arguments))
}

async fn write_message<W: AsyncWrite + Unpin>(writer: &mut W, value: &Value) -> Result<()> {
    let mut encoded = serde_json::to_vec(value)?;
    if encoded.len() > MAX_CONTROL_OUTPUT_BYTES {
        let Some(id) = value.get("id") else {
            return Err(BrainregiondError::Protocol(format!(
                "control output exceeds {MAX_CONTROL_OUTPUT_BYTES} bytes"
            )));
        };
        encoded = serde_json::to_vec(&failure(
            id.clone(),
            RpcFault::internal(format!(
                "control result exceeds {MAX_CONTROL_OUTPUT_BYTES} bytes"
            )),
        ))?;
    }
    encoded.push(b'\n');
    writer.write_all(&encoded).await?;
    writer.flush().await?;
    Ok(())
}

fn combine_run_and_cleanup(run: Result<()>, cleanup: Result<()>) -> Result<()> {
    match (run, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(cleanup_error)) => Err(BrainregiondError::Protocol(format!(
            "{error}; MCP cleanup also failed: {cleanup_error}"
        ))),
    }
}

pub async fn serve_stdio(supervisor: McpSupervisor) -> Result<()> {
    serve_stdio_until(supervisor, termination_signal()).await
}

pub async fn serve_stdio_until<S>(mut supervisor: McpSupervisor, shutdown_signal: S) -> Result<()>
where
    S: Future<Output = io::Result<()>>,
{
    let (stdin, forwarder) = match detached_stdin_reader() {
        Ok(reader) => reader,
        Err(error) => {
            let cleanup = supervisor.shutdown().await;
            return combine_run_and_cleanup(Err(error), cleanup);
        }
    };
    let stdout = tokio::io::stdout();
    let result = ControlServer::new(supervisor)
        .serve_until(stdin, stdout, shutdown_signal)
        .await;
    forwarder.abort();
    let _ = forwarder.await;
    result
}

#[cfg(not(unix))]
pub async fn termination_signal() -> io::Result<()> {
    tokio::signal::ctrl_c().await
}

#[cfg(unix)]
pub async fn termination_signal() -> io::Result<()> {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! {
        result = tokio::signal::ctrl_c() => result,
        _ = terminate.recv() => Ok(()),
    }
}

fn detached_stdin_reader() -> Result<(BufReader<DuplexStream>, tokio::task::JoinHandle<()>)> {
    const READ_CHUNK_BYTES: usize = 8 * 1024;
    const READ_QUEUE_CHUNKS: usize = 8;

    let (sender, mut receiver) = tokio::sync::mpsc::channel::<Vec<u8>>(READ_QUEUE_CHUNKS);
    std::thread::Builder::new()
        .name("brainregiond-stdin".to_owned())
        .spawn(move || {
            use std::io::Read;

            let stdin = io::stdin();
            let mut stdin = stdin.lock();
            let mut buffer = [0_u8; READ_CHUNK_BYTES];
            loop {
                let count = match stdin.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(count) => count,
                };
                if sender.blocking_send(buffer[..count].to_vec()).is_err() {
                    break;
                }
            }
        })?;

    let (mut producer, consumer) = tokio::io::duplex(READ_CHUNK_BYTES * 2);
    let forwarder = tokio::spawn(async move {
        while let Some(chunk) = receiver.recv().await {
            if producer.write_all(&chunk).await.is_err() {
                return;
            }
        }
        let _ = producer.shutdown().await;
    });
    Ok((BufReader::new(consumer), forwarder))
}

pub async fn probe(supervisor: &mut McpSupervisor) -> Result<Value> {
    let ping = supervisor.ping().await?;
    let metadata = supervisor.metadata();
    Ok(json!({
        "ok": true,
        "status": supervisor.state().as_str(),
        "daemon": {
            "name": DAEMON_NAME,
            "version": DAEMON_VERSION,
            "protocolVersion": CONTROL_PROTOCOL_VERSION,
        },
        "mcp": {
            "protocolVersion": metadata.protocol_version,
            "serverInfo": metadata.server_info,
            "toolCount": metadata.tool_count,
            "brainregion": ping,
        }
    }))
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::{Context, Poll};

    use tokio::io::AsyncWrite;

    use super::*;
    use crate::supervisor::{McpMetadata, McpState};

    struct FakeBackend {
        metadata: McpMetadata,
        stopped: Arc<AtomicBool>,
    }

    impl FakeBackend {
        fn new(stopped: Arc<AtomicBool>) -> Self {
            let ping = BrainregionPing {
                ok: true,
                name: "brainregion".to_owned(),
                legacy_name: Some("brain_region".to_owned()),
                version: "0.2.0".to_owned(),
            };
            Self {
                metadata: McpMetadata {
                    protocol_version: "2025-11-25".to_owned(),
                    server_info: json!({"name": "brainregion", "version": "1.28.1"}),
                    capabilities: json!({"tools": {"listChanged": false}}),
                    tool_count: 2,
                    brainregion: ping,
                },
                stopped,
            }
        }
    }

    impl McpBackend for FakeBackend {
        fn state(&self) -> McpState {
            if self.stopped.load(Ordering::SeqCst) {
                McpState::Stopped
            } else {
                McpState::Ready
            }
        }

        fn metadata(&self) -> &McpMetadata {
            &self.metadata
        }

        async fn list_tools_value(&mut self) -> Result<Value> {
            Ok(json!({"tools": [{"name": "ping"}, {"name": "review_document"}]}))
        }

        async fn call_tool_value(
            &mut self,
            name: &str,
            arguments: Map<String, Value>,
        ) -> Result<Value> {
            Ok(json!({"name": name, "arguments": arguments, "isError": false}))
        }

        async fn ping(&mut self) -> Result<BrainregionPing> {
            Ok(self.metadata.brainregion.clone())
        }

        async fn shutdown(&mut self) -> Result<()> {
            self.stopped.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    struct SlowBackend(FakeBackend);

    impl McpBackend for SlowBackend {
        fn state(&self) -> McpState {
            self.0.state()
        }

        fn metadata(&self) -> &McpMetadata {
            self.0.metadata()
        }

        async fn list_tools_value(&mut self) -> Result<Value> {
            self.0.list_tools_value().await
        }

        async fn call_tool_value(
            &mut self,
            _name: &str,
            _arguments: Map<String, Value>,
        ) -> Result<Value> {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            unreachable!("the shutdown request must cancel the slow call")
        }

        async fn ping(&mut self) -> Result<BrainregionPing> {
            self.0.ping().await
        }

        async fn shutdown(&mut self) -> Result<()> {
            self.0.shutdown().await
        }
    }

    struct SlowShutdownBackend {
        inner: FakeBackend,
        calls: Arc<AtomicUsize>,
        completed: Arc<AtomicBool>,
    }

    impl McpBackend for SlowShutdownBackend {
        fn state(&self) -> McpState {
            self.inner.state()
        }

        fn metadata(&self) -> &McpMetadata {
            self.inner.metadata()
        }

        async fn list_tools_value(&mut self) -> Result<Value> {
            self.inner.list_tools_value().await
        }

        async fn call_tool_value(
            &mut self,
            name: &str,
            arguments: Map<String, Value>,
        ) -> Result<Value> {
            self.inner.call_tool_value(name, arguments).await
        }

        async fn ping(&mut self) -> Result<BrainregionPing> {
            self.inner.ping().await
        }

        async fn shutdown(&mut self) -> Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            self.completed.store(true, Ordering::SeqCst);
            self.inner.shutdown().await
        }
    }

    #[tokio::test]
    async fn serves_ready_health_tool_call_and_shutdown() {
        let stopped = Arc::new(AtomicBool::new(false));
        let backend = FakeBackend::new(stopped.clone());
        let input = concat!(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"daemon/health\"}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"mcp/tools/call\",\"params\":{\"name\":\"ping\",\"arguments\":{}}}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"daemon/shutdown\"}\n"
        );
        let mut output = Vec::new();

        ControlServer::new(backend)
            .serve(input.as_bytes(), &mut output)
            .await
            .unwrap();

        let messages: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0]["method"], "daemon/ready");
        assert_eq!(messages[1]["result"]["ok"], true);
        assert_eq!(messages[2]["result"]["name"], "ping");
        assert_eq!(messages[3]["result"]["accepted"], true);
        assert!(stopped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn returns_json_rpc_errors_without_crashing() {
        let stopped = Arc::new(AtomicBool::new(false));
        let backend = FakeBackend::new(stopped.clone());
        let input = concat!(
            "not-json\n",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"missing\"}\n"
        );
        let mut output = Vec::new();

        ControlServer::new(backend)
            .serve(input.as_bytes(), &mut output)
            .await
            .unwrap();

        let messages: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(messages[1]["error"]["code"], -32700);
        assert_eq!(messages[2]["error"]["code"], -32601);
        assert!(stopped.load(Ordering::SeqCst));
    }

    struct BrokenWriter;

    impl AsyncWrite for BrokenWriter {
        fn poll_write(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
            _buffer: &[u8],
        ) -> Poll<io::Result<usize>> {
            Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "test writer closed",
            )))
        }

        fn poll_flush(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn cleans_up_backend_when_ready_write_fails() {
        let stopped = Arc::new(AtomicBool::new(false));
        let backend = FakeBackend::new(stopped.clone());

        let error = ControlServer::new(backend)
            .serve(&b""[..], BrokenWriter)
            .await
            .unwrap_err();

        assert!(matches!(error, BrainregiondError::Io(_)));
        assert!(stopped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn shutdown_preempts_a_slow_tool_call() {
        let stopped = Arc::new(AtomicBool::new(false));
        let backend = SlowBackend(FakeBackend::new(stopped.clone()));
        let input = concat!(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"mcp/tools/call\",\"params\":{\"name\":\"slow_mutation\"}}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"daemon/shutdown\"}\n"
        );
        let mut output = Vec::new();

        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            ControlServer::new(backend).serve(input.as_bytes(), &mut output),
        )
        .await
        .expect("shutdown should not wait for the tool timeout")
        .unwrap();

        let messages: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(messages[1]["id"], 1);
        assert_eq!(messages[1]["error"]["code"], -32002);
        assert_eq!(messages[2]["id"], 2);
        assert_eq!(messages[2]["result"]["accepted"], true);
        assert!(stopped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn stdin_eof_cancels_a_slow_tool_call() {
        let stopped = Arc::new(AtomicBool::new(false));
        let backend = SlowBackend(FakeBackend::new(stopped.clone()));
        let input = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"mcp/tools/call\",\"params\":{\"name\":\"slow_mutation\"}}\n";
        let mut output = Vec::new();

        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            ControlServer::new(backend).serve(input.as_bytes(), &mut output),
        )
        .await
        .expect("stdin EOF should not wait for the tool timeout")
        .unwrap();

        assert_eq!(String::from_utf8(output).unwrap().lines().count(), 1);
        assert!(stopped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn shutdown_cleanup_is_not_interrupted_by_a_signal() {
        let stopped = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicBool::new(false));
        let backend = SlowShutdownBackend {
            inner: FakeBackend::new(stopped.clone()),
            calls: calls.clone(),
            completed: completed.clone(),
        };
        let input = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"daemon/shutdown\"}\n";
        let mut output = Vec::new();
        let signal = async {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            Ok(())
        };

        ControlServer::new(backend)
            .serve_until(input.as_bytes(), &mut output, signal)
            .await
            .unwrap();

        let messages: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(messages[1]["result"]["accepted"], true);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(completed.load(Ordering::SeqCst));
        assert!(stopped.load(Ordering::SeqCst));
    }

    #[test]
    fn tool_timeout_is_reported_as_an_unknown_non_retryable_outcome() {
        let fault = tool_call_fault(BrainregiondError::Timeout {
            operation: "MCP tools/call mutate".to_owned(),
            timeout: std::time::Duration::from_secs(1),
        });

        assert_eq!(fault.code, -32001);
        assert_eq!(fault.data.as_ref().unwrap()["outcome"], "unknown");
        assert_eq!(fault.data.as_ref().unwrap()["retryable"], false);
    }
}
