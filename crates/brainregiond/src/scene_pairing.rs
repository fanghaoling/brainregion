//! One-time challenge authentication for Runtime Scene RPC transports.
//!
//! A transport sends a fresh random challenge before `runtime/register`. The
//! Player proves possession of a pre-shared secret with HMAC-SHA256 over the
//! nonce, configured principal, and every security-relevant registration field.
//! The nonce is scoped to one connection, so a captured registration cannot be
//! replayed on a later named-pipe or WSS connection.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

use crate::config::{PairingSecret, ScenePipeConfig};
use crate::error::{BrainregiondError, Result};
use crate::protocol::read_bounded_line_async;
use crate::scene_peer::{
    ScenePeerAuth, ScenePeerHandle, ScenePeerRegistry, accept_registered_scene_peer,
};
use crate::scene_rpc::{
    MAX_JSON_SAFE_INTEGER, MAX_SCENE_FRAME_BYTES, RuntimeRegistration,
    RuntimeRegistrationNotification, RuntimeStatus, SceneCapability,
};

pub const PAIRING_PROTOCOL_VERSION: &str = "brainregion.scene.pairing.v1";
pub const PAIRING_ALGORITHM: &str = "hmac-sha256";
const NONCE_BYTES: usize = 32;
const PROOF_PREFIX: &str = "hmac-sha256.";

type HmacSha256 = Hmac<Sha256>;

#[derive(Clone, Debug)]
pub struct PairingPolicy {
    principal_id: String,
    secret: PairingSecret,
    granted_capabilities: Vec<SceneCapability>,
    authentication_timeout: Duration,
}

impl From<&ScenePipeConfig> for PairingPolicy {
    fn from(config: &ScenePipeConfig) -> Self {
        Self {
            principal_id: config.principal_id.clone(),
            secret: config.pairing_secret.clone(),
            granted_capabilities: config.granted_capabilities.clone(),
            authentication_timeout: config.authentication_timeout,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PairingChallengeNotification {
    pub jsonrpc: String,
    pub method: String,
    pub params: PairingChallenge,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PairingChallenge {
    pub protocol_version: String,
    pub algorithm: String,
    pub nonce: String,
    pub expires_unix_ms: u64,
    pub principal_id: String,
}

impl PairingChallengeNotification {
    fn issue(principal_id: &str, timeout: Duration) -> Result<Self> {
        let mut nonce = [0_u8; NONCE_BYTES];
        getrandom::fill(&mut nonce).map_err(|error| {
            BrainregiondError::Protocol(format!("could not generate pairing nonce: {error}"))
        })?;
        let expires_unix_ms = unix_time_ms()?
            .checked_add(timeout.as_millis())
            .filter(|value| *value <= u128::from(MAX_JSON_SAFE_INTEGER))
            .ok_or_else(|| {
                BrainregiondError::Config(
                    "pairing challenge expiry exceeds the JSON safe-integer range".to_owned(),
                )
            })? as u64;
        Ok(Self {
            jsonrpc: "2.0".to_owned(),
            method: "runtime/challenge".to_owned(),
            params: PairingChallenge {
                protocol_version: PAIRING_PROTOCOL_VERSION.to_owned(),
                algorithm: PAIRING_ALGORITHM.to_owned(),
                nonce: URL_SAFE_NO_PAD.encode(nonce),
                expires_unix_ms,
                principal_id: principal_id.to_owned(),
            },
        })
    }
}

/// Authenticate one bidirectional stream, then transfer it to the normal peer
/// session. Concrete listeners still own OS access control and connection limits.
pub async fn authenticate_scene_peer<S>(
    registry: ScenePeerRegistry,
    policy: PairingPolicy,
    stream: S,
) -> Result<ScenePeerHandle>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let challenge =
        PairingChallengeNotification::issue(&policy.principal_id, policy.authentication_timeout)?;
    let (read_half, mut write_half) = tokio::io::split(stream);
    let mut reader = BufReader::new(read_half);
    write_json_line(&mut write_half, &serde_json::to_value(&challenge)?).await?;

    let registration_line = tokio::time::timeout(
        policy.authentication_timeout,
        read_bounded_line_async(&mut reader, MAX_SCENE_FRAME_BYTES),
    )
    .await
    .map_err(|_| BrainregiondError::Timeout {
        operation: "Runtime Scene RPC pairing".to_owned(),
        timeout: policy.authentication_timeout,
    })??
    .ok_or_else(|| {
        BrainregiondError::Protocol("Runtime Scene RPC peer closed during pairing".to_owned())
    })?;
    let registration: RuntimeRegistrationNotification = serde_json::from_str(&registration_line)?;
    registration
        .validate()
        .map_err(BrainregiondError::Protocol)?;
    verify_pairing_proof(
        policy.secret.expose(),
        &challenge.params,
        &registration.params,
    )?;

    let auth = ScenePeerAuth::new(policy.principal_id, policy.granted_capabilities)?;
    accept_registered_scene_peer(registry, auth, reader, write_half, registration).await
}

/// Produce the exact proof format expected in `runtime/register.pairingProof`.
/// This is public so mock clients and future Unity transports can share golden
/// vectors; production callers must keep `secret` outside logs and serialized DTOs.
pub fn build_pairing_proof(
    secret: &[u8],
    challenge: &PairingChallenge,
    registration: &RuntimeRegistration,
) -> Result<String> {
    let mut mac = HmacSha256::new_from_slice(secret).map_err(|_| {
        BrainregiondError::Config("pairing secret is not valid for HMAC-SHA256".to_owned())
    })?;
    mac.update(&canonical_pairing_payload(challenge, registration));
    Ok(format!(
        "{PROOF_PREFIX}{}",
        URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes())
    ))
}

fn verify_pairing_proof(
    secret: &[u8],
    challenge: &PairingChallenge,
    registration: &RuntimeRegistration,
) -> Result<()> {
    let encoded = registration
        .pairing_proof
        .as_deref()
        .and_then(|proof| proof.strip_prefix(PROOF_PREFIX))
        .ok_or_else(pairing_rejected)?;
    let supplied = URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|_| pairing_rejected())?;
    let mut mac = HmacSha256::new_from_slice(secret).map_err(|_| pairing_rejected())?;
    mac.update(&canonical_pairing_payload(challenge, registration));
    mac.verify_slice(&supplied).map_err(|_| pairing_rejected())
}

fn canonical_pairing_payload(
    challenge: &PairingChallenge,
    registration: &RuntimeRegistration,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(512);
    append_field(&mut payload, challenge.protocol_version.as_bytes());
    append_field(&mut payload, challenge.algorithm.as_bytes());
    append_field(&mut payload, challenge.nonce.as_bytes());
    append_field(
        &mut payload,
        challenge.expires_unix_ms.to_string().as_bytes(),
    );
    append_field(&mut payload, challenge.principal_id.as_bytes());
    append_field(&mut payload, registration.protocol_version.as_bytes());
    append_field(&mut payload, registration.instance_id.as_bytes());
    append_field(&mut payload, registration.session_id.as_bytes());
    append_field(&mut payload, registration.build_id.as_bytes());
    append_field(&mut payload, registration.unity_version.as_bytes());
    append_field(&mut payload, registration.platform.as_bytes());
    append_field(&mut payload, registration.product.as_bytes());
    append_field(&mut payload, registration.scene_id.as_bytes());
    append_field(
        &mut payload,
        registration.scene_revision.to_string().as_bytes(),
    );
    append_field(
        &mut payload,
        match registration.status {
            RuntimeStatus::Ready => b"ready",
            RuntimeStatus::Degraded => b"degraded",
        },
    );
    match &registration.error {
        Some(error) => {
            payload.push(1);
            append_field(&mut payload, error.as_bytes());
        }
        None => payload.push(0),
    }
    let mut capabilities = registration.capabilities.clone();
    capabilities.sort_by_key(|capability| capability_name(*capability));
    append_field(&mut payload, capabilities.len().to_string().as_bytes());
    for capability in capabilities {
        append_field(&mut payload, capability_name(capability).as_bytes());
    }
    payload
}

fn append_field(payload: &mut Vec<u8>, value: &[u8]) {
    payload.extend_from_slice(value.len().to_string().as_bytes());
    payload.push(b':');
    payload.extend_from_slice(value);
    payload.push(b'\n');
}

fn pairing_rejected() -> BrainregiondError {
    BrainregiondError::Protocol("Runtime Scene RPC pairing proof was rejected".to_owned())
}

fn unix_time_ms() -> Result<u128> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            BrainregiondError::Protocol(format!("system clock is before epoch: {error}"))
        })?
        .as_millis())
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

async fn write_json_line<W: AsyncWrite + Unpin>(
    writer: &mut W,
    value: &serde_json::Value,
) -> Result<()> {
    let encoded = serde_json::to_vec(value)?;
    if encoded.len() > MAX_SCENE_FRAME_BYTES {
        return Err(BrainregiondError::Protocol(format!(
            "Runtime pairing frame exceeds {MAX_SCENE_FRAME_BYTES} bytes"
        )));
    }
    writer.write_all(&encoded).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    fn registration() -> RuntimeRegistration {
        let fixture = include_str!("../../../schemas/scene-rpc/v1/examples/runtime-register.json");
        let notification: RuntimeRegistrationNotification = serde_json::from_str(fixture).unwrap();
        notification.params
    }

    fn challenge() -> PairingChallenge {
        PairingChallenge {
            protocol_version: PAIRING_PROTOCOL_VERSION.to_owned(),
            algorithm: PAIRING_ALGORITHM.to_owned(),
            nonce: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".to_owned(),
            expires_unix_ms: 1_800_000_000_000,
            principal_id: "unity-local".to_owned(),
        }
    }

    #[test]
    fn proof_is_deterministic_and_binds_registration_fields() {
        let secret = b"0123456789abcdef0123456789abcdef";
        let challenge = challenge();
        let mut registration = registration();
        let proof = build_pairing_proof(secret, &challenge, &registration).unwrap();
        assert_eq!(
            proof,
            "hmac-sha256.S4MPoLHckkQeWS8FToy1MyvK0ZMpFxzqrkd32zFpnSA"
        );
        registration.pairing_proof = Some(proof);
        verify_pairing_proof(secret, &challenge, &registration).unwrap();

        registration.scene_id = "OtherScene".to_owned();
        assert!(verify_pairing_proof(secret, &challenge, &registration).is_err());

        let mut replay_registration = self::registration();
        replay_registration.pairing_proof =
            Some(build_pairing_proof(secret, &challenge, &replay_registration).unwrap());
        let mut replay_challenge = challenge;
        replay_challenge.nonce = "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB".to_owned();
        assert!(verify_pairing_proof(secret, &replay_challenge, &replay_registration).is_err());
    }

    #[test]
    fn challenge_shape_does_not_contain_secret_material() {
        let challenge =
            PairingChallengeNotification::issue("unity-local", Duration::from_secs(10)).unwrap();
        let value: Value = serde_json::to_value(challenge).unwrap();
        assert_eq!(value["method"], "runtime/challenge");
        assert_eq!(value["params"]["algorithm"], "hmac-sha256");
        assert_eq!(value["params"]["nonce"].as_str().unwrap().len(), 43);
    }

    #[tokio::test]
    async fn wrong_secret_is_rejected_before_registry_attachment() {
        let registry = ScenePeerRegistry::default();
        let policy = PairingPolicy {
            principal_id: "unity-local".to_owned(),
            secret: PairingSecret::new(b"0123456789abcdef0123456789abcdef").unwrap(),
            granted_capabilities: vec![SceneCapability::SceneRead],
            authentication_timeout: Duration::from_secs(1),
        };
        let (server_stream, client_stream) = tokio::io::duplex(16 * 1024);
        let client = tokio::spawn(async move {
            let (read_half, mut write_half) = tokio::io::split(client_stream);
            let mut reader = BufReader::new(read_half);
            let challenge_line = read_bounded_line_async(&mut reader, MAX_SCENE_FRAME_BYTES)
                .await
                .unwrap()
                .unwrap();
            let challenge: PairingChallengeNotification =
                serde_json::from_str(&challenge_line).unwrap();
            let mut registration = registration();
            registration.pairing_proof = Some(
                build_pairing_proof(
                    b"wrong-secret-wrong-secret-000000",
                    &challenge.params,
                    &registration,
                )
                .unwrap(),
            );
            let notification = RuntimeRegistrationNotification {
                jsonrpc: "2.0".to_owned(),
                method: "runtime/register".to_owned(),
                params: registration,
            };
            write_json_line(
                &mut write_half,
                &serde_json::to_value(notification).unwrap(),
            )
            .await
            .unwrap();
        });

        let error = authenticate_scene_peer(registry.clone(), policy, server_stream)
            .await
            .err()
            .expect("wrong secret must be rejected");
        assert!(error.to_string().contains("pairing proof was rejected"));
        assert!(registry.snapshots().await.is_empty());
        client.await.unwrap();
    }
}
