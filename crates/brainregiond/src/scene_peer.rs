//! Transport-neutral session handling for packaged-game Runtime Scene RPC peers.
//!
//! A concrete listener must authenticate and pair the connection before calling
//! [`accept_scene_peer`]. The session layer then owns registration validation,
//! bounded JSONL framing, request correlation, deadlines, notifications, and
//! connection-epoch replacement. Named pipes and WSS adapters can share this
//! state machine without exposing a network listener from the core crate.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex, broadcast, mpsc, oneshot, watch};
use tokio::time::{Instant, MissedTickBehavior};

use crate::error::{BrainregiondError, Result};
use crate::protocol::read_bounded_line_async;
use crate::scene_rpc::{
    MAX_JSON_SAFE_INTEGER, MAX_SCENE_FRAME_BYTES, RuntimeRegistration,
    RuntimeRegistrationNotification, RuntimeStatus, SceneCapability, SceneChangedNotification,
};

const REQUEST_QUEUE_CAPACITY: usize = 128;
const MAX_PENDING_REQUESTS: usize = 128;
const EVENT_QUEUE_CAPACITY: usize = 128;
const MAX_RETIRED_REQUEST_IDS: usize = 256;
const PENDING_SWEEP_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Clone, Debug)]
pub struct ScenePeerAuth {
    principal_id: String,
    granted_capabilities: Vec<SceneCapability>,
}

impl ScenePeerAuth {
    /// Authentication is deliberately external to this constructor. A transport
    /// may only create this value after pairing/credential verification succeeds.
    pub fn new(
        principal_id: impl Into<String>,
        capabilities: impl IntoIterator<Item = SceneCapability>,
    ) -> Result<Self> {
        let principal_id = principal_id.into();
        validate_wire_identifier("principalId", &principal_id, 128)?;

        let mut unique = HashSet::new();
        let mut granted_capabilities = Vec::new();
        for capability in capabilities {
            if unique.insert(capability) {
                granted_capabilities.push(capability);
            }
        }
        granted_capabilities.sort_by_key(|capability| capability_name(*capability));

        Ok(Self {
            principal_id,
            granted_capabilities,
        })
    }

    pub fn principal_id(&self) -> &str {
        &self.principal_id
    }

    pub fn granted_capabilities(&self) -> &[SceneCapability] {
        &self.granted_capabilities
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenePeerState {
    Connected,
    Superseded,
    Disconnected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SceneMethod {
    RuntimeInfo,
    Hierarchy,
    Inspect,
    PrefabList,
    Preview,
    Apply,
    Undo,
    LogsPoll,
}

impl SceneMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RuntimeInfo => "runtime/info",
            Self::Hierarchy => "scene/hierarchy",
            Self::Inspect => "object/inspect",
            Self::PrefabList => "prefab/list",
            Self::Preview => "scene/preview",
            Self::Apply => "scene/apply",
            Self::Undo => "history/undo",
            Self::LogsPoll => "logs/poll",
        }
    }

    pub fn from_wire_name(method: &str) -> Option<Self> {
        match method {
            "runtime/info" => Some(Self::RuntimeInfo),
            "scene/hierarchy" => Some(Self::Hierarchy),
            "object/inspect" => Some(Self::Inspect),
            "prefab/list" => Some(Self::PrefabList),
            "scene/preview" => Some(Self::Preview),
            "scene/apply" => Some(Self::Apply),
            "history/undo" => Some(Self::Undo),
            "logs/poll" => Some(Self::LogsPoll),
            _ => None,
        }
    }

    fn required_capability(self) -> SceneCapability {
        match self {
            Self::RuntimeInfo | Self::Hierarchy | Self::Inspect | Self::PrefabList => {
                SceneCapability::SceneRead
            }
            Self::Preview | Self::Apply => SceneCapability::SceneWrite,
            Self::Undo => SceneCapability::SceneUndo,
            Self::LogsPoll => SceneCapability::LogsRead,
        }
    }

    fn may_mutate_scene(self) -> bool {
        matches!(self, Self::Apply | Self::Undo)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScenePeerSnapshot {
    pub principal_id: String,
    pub connection_epoch: u64,
    pub instance_id: String,
    pub session_id: String,
    pub build_id: String,
    pub scene_id: String,
    pub scene_revision: u64,
    pub runtime_status: RuntimeStatus,
    pub runtime_error: Option<String>,
    pub supported_capabilities: Vec<SceneCapability>,
    pub granted_capabilities: Vec<SceneCapability>,
    pub connection_state: ScenePeerState,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScenePeerEvent {
    pub method: String,
    pub scene_revision: u64,
    pub client_mutation_id: Option<String>,
    pub summary: String,
}

#[derive(Clone, Default)]
pub struct ScenePeerRegistry {
    inner: Arc<Mutex<RegistryState>>,
}

#[derive(Default)]
struct RegistryState {
    next_epoch_by_principal: HashMap<String, u64>,
    active_by_principal: HashMap<String, Arc<ScenePeerInner>>,
}

#[derive(Clone)]
pub struct ScenePeerHandle {
    inner: Arc<ScenePeerInner>,
}

struct ScenePeerInner {
    principal_id: String,
    connection_epoch: u64,
    registration: RuntimeRegistration,
    granted_capabilities: Vec<SceneCapability>,
    observed_revision: AtomicU64,
    next_request_id: AtomicU64,
    commands: mpsc::Sender<PeerCommand>,
    lifecycle: watch::Sender<ScenePeerState>,
    events: broadcast::Sender<ScenePeerEvent>,
}

struct PeerCommand {
    id: String,
    method: SceneMethod,
    params: Value,
    deadline_unix_ms: u64,
    deadline_at: Instant,
    response: oneshot::Sender<PeerReply>,
}

struct PendingRequest {
    deadline_at: Instant,
    response: oneshot::Sender<PeerReply>,
}

enum PeerReply {
    Success(Value),
    UpstreamError(Value),
    Rejected(String),
    Disconnected(String),
    TimedOut,
}

impl ScenePeerRegistry {
    pub async fn get(&self, principal_id: &str) -> Option<ScenePeerHandle> {
        let state = self.inner.lock().await;
        state
            .active_by_principal
            .get(principal_id)
            .filter(|inner| inner.state() == ScenePeerState::Connected)
            .cloned()
            .map(|inner| ScenePeerHandle { inner })
    }

    pub async fn snapshots(&self) -> Vec<ScenePeerSnapshot> {
        let state = self.inner.lock().await;
        let mut snapshots: Vec<_> = state
            .active_by_principal
            .values()
            .filter(|inner| inner.state() == ScenePeerState::Connected)
            .map(|inner| inner.snapshot())
            .collect();
        snapshots.sort_by(|left, right| left.principal_id.cmp(&right.principal_id));
        snapshots
    }

    pub async fn close_all(&self) {
        let peers = {
            let mut state = self.inner.lock().await;
            state
                .active_by_principal
                .drain()
                .map(|(_, peer)| peer)
                .collect::<Vec<_>>()
        };
        for peer in peers {
            peer.set_state(ScenePeerState::Disconnected);
        }
    }

    async fn attach(
        &self,
        auth: ScenePeerAuth,
        mut registration: RuntimeRegistration,
    ) -> Result<(
        ScenePeerHandle,
        mpsc::Receiver<PeerCommand>,
        watch::Receiver<ScenePeerState>,
    )> {
        // Pairing proofs are credential material for a concrete transport. The
        // session registry never retains or exposes them after authentication.
        registration.pairing_proof = None;

        let (commands, command_receiver) = mpsc::channel(REQUEST_QUEUE_CAPACITY);
        let (lifecycle, lifecycle_receiver) = watch::channel(ScenePeerState::Connected);
        let (events, _) = broadcast::channel(EVENT_QUEUE_CAPACITY);

        let mut state = self.inner.lock().await;
        let next_epoch = state
            .next_epoch_by_principal
            .entry(auth.principal_id.clone())
            .or_insert(0);
        *next_epoch = next_epoch.checked_add(1).ok_or_else(|| {
            BrainregiondError::Protocol(format!(
                "Runtime Scene RPC connection epoch exhausted for principal {:?}",
                auth.principal_id
            ))
        })?;
        let connection_epoch = *next_epoch;

        let inner = Arc::new(ScenePeerInner {
            principal_id: auth.principal_id.clone(),
            connection_epoch,
            observed_revision: AtomicU64::new(registration.scene_revision),
            next_request_id: AtomicU64::new(1),
            registration,
            granted_capabilities: auth.granted_capabilities,
            commands,
            lifecycle,
            events,
        });
        let previous = state
            .active_by_principal
            .insert(auth.principal_id, Arc::clone(&inner));
        drop(state);

        if let Some(previous) = previous {
            previous.set_state(ScenePeerState::Superseded);
        }

        Ok((
            ScenePeerHandle {
                inner: Arc::clone(&inner),
            },
            command_receiver,
            lifecycle_receiver,
        ))
    }

    async fn remove_if_current(&self, principal_id: &str, connection_epoch: u64) {
        let mut state = self.inner.lock().await;
        let is_current = state
            .active_by_principal
            .get(principal_id)
            .is_some_and(|inner| inner.connection_epoch == connection_epoch);
        if is_current {
            state.active_by_principal.remove(principal_id);
        }
    }
}

impl ScenePeerHandle {
    pub fn principal_id(&self) -> &str {
        &self.inner.principal_id
    }

    pub fn connection_epoch(&self) -> u64 {
        self.inner.connection_epoch
    }

    pub fn state(&self) -> ScenePeerState {
        self.inner.state()
    }

    pub fn snapshot(&self) -> ScenePeerSnapshot {
        self.inner.snapshot()
    }

    pub fn subscribe_state(&self) -> watch::Receiver<ScenePeerState> {
        self.inner.lifecycle.subscribe()
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<ScenePeerEvent> {
        self.inner.events.subscribe()
    }

    pub fn close(&self) {
        self.inner.set_state(ScenePeerState::Disconnected);
    }

    pub async fn request(
        &self,
        method: SceneMethod,
        params: Value,
        timeout: Duration,
    ) -> Result<Value> {
        if !params.is_object() {
            return Err(BrainregiondError::Protocol(
                "Runtime Scene RPC params must be an object".to_owned(),
            ));
        }
        if timeout.as_millis() == 0 {
            return Err(BrainregiondError::Config(
                "Runtime Scene RPC timeout must be at least one millisecond".to_owned(),
            ));
        }
        if self.state() != ScenePeerState::Connected {
            return Err(scene_rpc_error(
                -32011,
                format!(
                    "Runtime Scene RPC peer {:?} is not connected",
                    self.principal_id()
                ),
                true,
            ));
        }

        self.require_capability(method.required_capability())?;
        if method == SceneMethod::Preview && preview_contains_spawn(&params) {
            self.require_capability(SceneCapability::SceneSpawn)?;
        }

        let (deadline_unix_ms, deadline_at) = request_deadline(timeout)?;
        let sequence = self.inner.next_request_id.fetch_add(1, Ordering::Relaxed);
        if sequence == 0 || sequence > MAX_JSON_SAFE_INTEGER {
            return Err(BrainregiondError::Protocol(
                "Runtime Scene RPC request id space is exhausted".to_owned(),
            ));
        }
        let id = format!("br:{}:{sequence}", self.connection_epoch());
        let (response, receiver) = oneshot::channel();
        let command = PeerCommand {
            id,
            method,
            params,
            deadline_unix_ms,
            deadline_at,
            response,
        };

        match self.inner.commands.try_send(command) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                return Err(scene_rpc_error(
                    -32003,
                    "Runtime Scene RPC request queue is full",
                    true,
                ));
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(scene_rpc_error(
                    -32011,
                    "Runtime Scene RPC peer disconnected before enqueue",
                    true,
                ));
            }
        }

        let reply = match tokio::time::timeout(timeout, receiver).await {
            Ok(Ok(reply)) => reply,
            Ok(Err(_)) => {
                return Err(scene_request_disconnected_error(
                    method,
                    "Runtime Scene RPC session ended before responding",
                ));
            }
            Err(_) => {
                return Err(BrainregiondError::Timeout {
                    operation: method.as_str().to_owned(),
                    timeout,
                });
            }
        };

        match reply {
            PeerReply::Success(result) => Ok(result),
            PeerReply::UpstreamError(error) => Err(BrainregiondError::Upstream(error)),
            PeerReply::Rejected(message) => Err(scene_rpc_error(-32003, message, true)),
            PeerReply::Disconnected(message) => {
                Err(scene_request_disconnected_error(method, message))
            }
            PeerReply::TimedOut => Err(BrainregiondError::Timeout {
                operation: method.as_str().to_owned(),
                timeout,
            }),
        }
    }

    fn require_capability(&self, capability: SceneCapability) -> Result<()> {
        let supported = self.inner.registration.capabilities.contains(&capability);
        let granted = self.inner.granted_capabilities.contains(&capability);
        if supported && granted {
            return Ok(());
        }
        Err(scene_rpc_error(
            -32002,
            format!(
                "Runtime Scene RPC capability {:?} is not both supported and granted",
                capability_name(capability)
            ),
            false,
        ))
    }
}

impl ScenePeerInner {
    fn state(&self) -> ScenePeerState {
        *self.lifecycle.borrow()
    }

    fn set_state(&self, state: ScenePeerState) {
        self.lifecycle.send_replace(state);
    }

    fn snapshot(&self) -> ScenePeerSnapshot {
        ScenePeerSnapshot {
            principal_id: self.principal_id.clone(),
            connection_epoch: self.connection_epoch,
            instance_id: self.registration.instance_id.clone(),
            session_id: self.registration.session_id.clone(),
            build_id: self.registration.build_id.clone(),
            scene_id: self.registration.scene_id.clone(),
            scene_revision: self.observed_revision.load(Ordering::Acquire),
            runtime_status: self.registration.status,
            runtime_error: self.registration.error.clone(),
            supported_capabilities: self.registration.capabilities.clone(),
            granted_capabilities: self.granted_capabilities.clone(),
            connection_state: self.state(),
        }
    }
}

/// Accept an already authenticated bidirectional byte stream.
///
/// The first frame must be `runtime/register`. This function intentionally does
/// not listen on a socket, verify a pairing proof, or choose an OS transport.
pub async fn accept_scene_peer<S>(
    registry: ScenePeerRegistry,
    auth: ScenePeerAuth,
    stream: S,
    registration_timeout: Duration,
) -> Result<ScenePeerHandle>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    if registration_timeout.as_millis() == 0 {
        return Err(BrainregiondError::Config(
            "Runtime peer registration timeout must be at least one millisecond".to_owned(),
        ));
    }

    let (read_half, write_half) = tokio::io::split(stream);
    let mut reader = BufReader::new(read_half);
    let registration_line = tokio::time::timeout(
        registration_timeout,
        read_bounded_line_async(&mut reader, MAX_SCENE_FRAME_BYTES),
    )
    .await
    .map_err(|_| BrainregiondError::Timeout {
        operation: "runtime/register".to_owned(),
        timeout: registration_timeout,
    })??
    .ok_or_else(|| {
        BrainregiondError::Protocol(
            "Runtime Scene RPC peer closed before runtime/register".to_owned(),
        )
    })?;

    let registration: RuntimeRegistrationNotification = serde_json::from_str(&registration_line)?;
    accept_registered_scene_peer(registry, auth, reader, write_half, registration).await
}

pub(crate) async fn accept_registered_scene_peer<R, W>(
    registry: ScenePeerRegistry,
    auth: ScenePeerAuth,
    reader: BufReader<R>,
    writer: W,
    registration: RuntimeRegistrationNotification,
) -> Result<ScenePeerHandle>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    registration
        .validate()
        .map_err(BrainregiondError::Protocol)?;
    for capability in auth.granted_capabilities() {
        if !registration.params.capabilities.contains(capability) {
            return Err(BrainregiondError::Protocol(format!(
                "paired policy grants unsupported Runtime Scene RPC capability {:?}",
                capability_name(*capability)
            )));
        }
    }

    let (handle, commands, lifecycle) = registry.attach(auth, registration.params).await?;
    let task_handle = handle.clone();
    tokio::spawn(async move {
        run_peer_session(registry, task_handle, reader, writer, commands, lifecycle).await;
    });
    Ok(handle)
}

async fn run_peer_session<R, W>(
    registry: ScenePeerRegistry,
    handle: ScenePeerHandle,
    mut reader: BufReader<R>,
    mut writer: W,
    mut commands: mpsc::Receiver<PeerCommand>,
    mut lifecycle: watch::Receiver<ScenePeerState>,
) where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut pending: HashMap<String, PendingRequest> = HashMap::new();
    let mut retired_ids = RetiredRequestIds::default();
    let mut sweep = tokio::time::interval(PENDING_SWEEP_INTERVAL);
    sweep.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let disconnect_reason = 'session: loop {
        tokio::select! {
            biased;
            changed = lifecycle.changed() => {
                if changed.is_err() || *lifecycle.borrow() != ScenePeerState::Connected {
                    break 'session match *lifecycle.borrow() {
                        ScenePeerState::Superseded => {
                            "Runtime Scene RPC connection was superseded by a newer epoch".to_owned()
                        }
                        _ => "Runtime Scene RPC connection was closed".to_owned(),
                    };
                }
            }
            _ = sweep.tick() => {
                let now = Instant::now();
                let expired: Vec<_> = pending
                    .iter()
                    .filter(|(_, request)| request.deadline_at <= now)
                    .map(|(id, _)| id.clone())
                    .collect();
                for id in expired {
                    if let Some(request) = pending.remove(&id) {
                        retired_ids.insert(id);
                        let _ = request.response.send(PeerReply::TimedOut);
                    }
                }
            }
            incoming = read_bounded_line_async(&mut reader, MAX_SCENE_FRAME_BYTES) => {
                match incoming {
                    Ok(Some(line)) => {
                        if let Err(message) = process_incoming(
                            &line,
                            &handle.inner,
                            &mut pending,
                            &mut retired_ids,
                        ) {
                            break 'session message;
                        }
                    }
                    Ok(None) => {
                        break 'session "Runtime Scene RPC peer reached EOF".to_owned();
                    }
                    Err(error) => {
                        break 'session format!("Runtime Scene RPC read failed: {error}");
                    }
                }
            }
            command = commands.recv() => {
                let Some(command) = command else {
                    break 'session "Runtime Scene RPC request channel closed".to_owned();
                };
                if handle.state() != ScenePeerState::Connected {
                    let reason = "Runtime Scene RPC connection is no longer current".to_owned();
                    let _ = command.response.send(PeerReply::Disconnected(
                        reason.clone(),
                    ));
                    break 'session reason;
                }
                if pending.len() >= MAX_PENDING_REQUESTS {
                    let _ = command.response.send(PeerReply::Rejected(
                        "Runtime Scene RPC pending request limit reached".to_owned(),
                    ));
                    continue;
                }
                if command.deadline_at <= Instant::now() {
                    let _ = command.response.send(PeerReply::TimedOut);
                    continue;
                }

                let request = json!({
                    "jsonrpc": "2.0",
                    "id": command.id,
                    "method": command.method.as_str(),
                    "deadlineUnixMs": command.deadline_unix_ms,
                    "params": command.params,
                });
                let request_id = request["id"].as_str().unwrap_or_default().to_owned();
                let write = write_json_line(&mut writer, &request);
                enum WriteRace {
                    Completed(Result<()>),
                    Closed(String),
                    TimedOut,
                }
                let write_race = tokio::select! {
                    biased;
                    changed = lifecycle.changed() => {
                        WriteRace::Closed(if changed.is_err() {
                            "Runtime Scene RPC lifecycle channel closed while writing".to_owned()
                        } else {
                            "Runtime Scene RPC connection closed while writing".to_owned()
                        })
                    }
                    _ = tokio::time::sleep_until(command.deadline_at) => {
                        WriteRace::TimedOut
                    }
                    result = write => WriteRace::Completed(result),
                };
                match write_race {
                    WriteRace::Closed(reason) => {
                        let _ = command.response.send(PeerReply::Disconnected(reason.clone()));
                        break 'session reason;
                    }
                    WriteRace::TimedOut => {
                        let _ = command.response.send(PeerReply::TimedOut);
                        break 'session
                            "Runtime Scene RPC write exceeded its deadline; stream discarded"
                                .to_owned();
                    }
                    WriteRace::Completed(Err(error)) => {
                        let reason = format!("Runtime Scene RPC write failed: {error}");
                        let _ = command.response.send(PeerReply::Disconnected(reason.clone()));
                        break 'session reason;
                    }
                    WriteRace::Completed(Ok(())) => {}
                }
                pending.insert(
                    request_id,
                    PendingRequest {
                        deadline_at: command.deadline_at,
                        response: command.response,
                    },
                );
            }
        }
    };

    for (_, request) in pending.drain() {
        let _ = request
            .response
            .send(PeerReply::Disconnected(disconnect_reason.clone()));
    }
    if handle.state() == ScenePeerState::Connected {
        handle.inner.set_state(ScenePeerState::Disconnected);
    }
    registry
        .remove_if_current(handle.principal_id(), handle.connection_epoch())
        .await;
}

fn process_incoming(
    line: &str,
    inner: &ScenePeerInner,
    pending: &mut HashMap<String, PendingRequest>,
    retired_ids: &mut RetiredRequestIds,
) -> std::result::Result<(), String> {
    let value: Value = serde_json::from_str(line)
        .map_err(|error| format!("Runtime Scene RPC returned invalid JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "Runtime Scene RPC message must be an object".to_owned())?;
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err("Runtime Scene RPC message has an invalid jsonrpc version".to_owned());
    }

    if let Some(id) = object.get("id") {
        let id = id
            .as_str()
            .ok_or_else(|| "Runtime Scene RPC response id must be a string".to_owned())?;
        validate_wire_identifier("response id", id, 128).map_err(|error| error.to_string())?;
        let has_result = object.contains_key("result");
        let has_error = object.contains_key("error");
        if has_result == has_error
            || object
                .keys()
                .any(|field| !matches!(field.as_str(), "jsonrpc" | "id" | "result" | "error"))
        {
            return Err("Runtime Scene RPC response shape is invalid".to_owned());
        }

        let Some(request) = pending.remove(id) else {
            if retired_ids.remove(id) {
                return Ok(());
            }
            return Err(format!(
                "Runtime Scene RPC returned an unknown response id {id:?}"
            ));
        };
        let reply = if has_result {
            if !object["result"].is_object() {
                return Err("Runtime Scene RPC result must be an object".to_owned());
            }
            PeerReply::Success(object["result"].clone())
        } else {
            let error = object["error"].as_object().ok_or_else(|| {
                "Runtime Scene RPC error response must contain an object".to_owned()
            })?;
            if error
                .keys()
                .any(|field| !matches!(field.as_str(), "code" | "message" | "data"))
                || error.get("code").and_then(Value::as_i64).is_none()
                || error
                    .get("message")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
                || error.get("data").is_some_and(|data| !data.is_object())
            {
                return Err("Runtime Scene RPC error response must contain an object".to_owned());
            }
            PeerReply::UpstreamError(object["error"].clone())
        };
        let _ = request.response.send(reply);
        return Ok(());
    }

    if object
        .keys()
        .any(|field| !matches!(field.as_str(), "jsonrpc" | "method" | "params"))
        || object.get("method").and_then(Value::as_str) != Some("scene/changed")
        || !object.get("params").is_some_and(Value::is_object)
    {
        return Err("Runtime Scene RPC sent an unsupported notification".to_owned());
    }

    let notification: SceneChangedNotification = serde_json::from_value(value)
        .map_err(|error| format!("invalid scene/changed notification: {error}"))?;
    notification.validate()?;
    let previous_revision = inner.observed_revision.load(Ordering::Acquire);
    if notification.params.scene_revision <= previous_revision {
        return Err(format!(
            "scene/changed revision {} is not newer than {}",
            notification.params.scene_revision, previous_revision
        ));
    }
    inner
        .observed_revision
        .store(notification.params.scene_revision, Ordering::Release);
    let _ = inner.events.send(ScenePeerEvent {
        method: notification.method,
        scene_revision: notification.params.scene_revision,
        client_mutation_id: notification.params.client_mutation_id,
        summary: notification.params.summary,
    });
    Ok(())
}

async fn write_json_line<W: AsyncWrite + Unpin>(writer: &mut W, value: &Value) -> Result<()> {
    let encoded = serde_json::to_vec(value)?;
    if encoded.len() > MAX_SCENE_FRAME_BYTES {
        return Err(BrainregiondError::Protocol(format!(
            "Runtime Scene RPC frame exceeds {MAX_SCENE_FRAME_BYTES} bytes"
        )));
    }
    writer.write_all(&encoded).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

fn request_deadline(timeout: Duration) -> Result<(u64, Instant)> {
    let now_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            BrainregiondError::Protocol(format!("system clock is before epoch: {error}"))
        })?
        .as_millis();
    let timeout_ms = timeout.as_millis();
    let deadline_unix_ms = now_unix_ms
        .checked_add(timeout_ms)
        .filter(|deadline| *deadline <= u128::from(MAX_JSON_SAFE_INTEGER))
        .ok_or_else(|| {
            BrainregiondError::Config(
                "Runtime Scene RPC deadline exceeds the JSON safe-integer range".to_owned(),
            )
        })? as u64;
    let deadline_at = Instant::now().checked_add(timeout).ok_or_else(|| {
        BrainregiondError::Config("Runtime Scene RPC monotonic deadline overflowed".to_owned())
    })?;
    Ok((deadline_unix_ms, deadline_at))
}

fn preview_contains_spawn(params: &Value) -> bool {
    params
        .get("commands")
        .and_then(Value::as_array)
        .is_some_and(|commands| {
            commands
                .iter()
                .any(|command| command.get("kind").and_then(Value::as_str) == Some("spawn"))
        })
}

fn scene_rpc_error(code: i64, message: impl Into<String>, retryable: bool) -> BrainregiondError {
    BrainregiondError::Upstream(json!({
        "code": code,
        "message": message.into(),
        "data": {"retryable": retryable},
    }))
}

fn scene_request_disconnected_error(
    method: SceneMethod,
    message: impl Into<String>,
) -> BrainregiondError {
    let outcome_unknown = method.may_mutate_scene();
    let mut data = json!({"retryable": !outcome_unknown});
    if outcome_unknown {
        data["outcome"] = json!("unknown");
    }
    BrainregiondError::Upstream(json!({
        "code": -32011,
        "message": message.into(),
        "data": data,
    }))
}

fn validate_wire_identifier(name: &str, value: &str, maximum: usize) -> Result<()> {
    if value.is_empty() || value.len() > maximum {
        return Err(BrainregiondError::Protocol(format!(
            "{name} must contain 1..{maximum} bytes"
        )));
    }
    if value
        .bytes()
        .any(|byte| !(byte.is_ascii_alphanumeric() || b"._:/-".contains(&byte)))
    {
        return Err(BrainregiondError::Protocol(format!(
            "{name} contains an unsupported character"
        )));
    }
    Ok(())
}

fn capability_name(capability: SceneCapability) -> &'static str {
    match capability {
        SceneCapability::SceneRead => "scene.read",
        SceneCapability::SceneWrite => "scene.write",
        SceneCapability::SceneSpawn => "scene.spawn",
        SceneCapability::SceneUndo => "scene.undo",
        SceneCapability::LogsRead => "logs.read",
    }
}

#[derive(Default)]
struct RetiredRequestIds {
    order: VecDeque<String>,
    values: HashSet<String>,
}

impl RetiredRequestIds {
    fn insert(&mut self, id: String) {
        if self.values.insert(id.clone()) {
            self.order.push_back(id);
        }
        while self.order.len() > MAX_RETIRED_REQUEST_IDS {
            if let Some(oldest) = self.order.pop_front() {
                self.values.remove(&oldest);
            }
        }
    }

    fn remove(&mut self, id: &str) -> bool {
        self.values.remove(id)
    }
}
