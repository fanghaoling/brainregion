//! Optional Windows named-pipe transport for packaged Unity Players.
//!
//! The listener is disabled unless configured. On Windows each pipe instance is
//! created with a protected DACL granting access only to the daemon's current
//! user SID, rejects remote clients, and then performs the one-time HMAC pairing
//! handshake before attaching the stream to [`crate::scene_peer`].

use crate::config::ScenePipeConfig;
use crate::error::{BrainregiondError, Result};
use crate::scene_peer::ScenePeerRegistry;

#[cfg(windows)]
mod platform {
    use std::ffi::c_void;
    use std::io;
    use std::mem::size_of;
    use std::ptr;
    use std::sync::Arc;

    use tokio::net::windows::named_pipe::{NamedPipeServer, ServerOptions};
    use tokio::sync::{Semaphore, watch};
    use tokio::task::{JoinHandle, JoinSet};
    use windows_sys::Win32::Foundation::{
        CloseHandle, ERROR_INSUFFICIENT_BUFFER, HANDLE, LocalFree,
    };
    use windows_sys::Win32::Security::Authorization::{
        ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
        SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::{
        GetTokenInformation, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};
    use windows_sys::core::PWSTR;

    use super::*;
    use crate::scene_pairing::{PairingPolicy, authenticate_scene_peer};
    use crate::scene_peer::ScenePeerState;

    pub struct ScenePipeListener {
        pipe_path: String,
        shutdown: watch::Sender<bool>,
        failure: watch::Receiver<Option<String>>,
        task: Option<JoinHandle<Result<()>>>,
    }

    impl ScenePipeListener {
        pub fn start(config: ScenePipeConfig, registry: ScenePeerRegistry) -> Result<Self> {
            let pipe_path = format!(r"\\.\pipe\{}", config.name);
            let security = PipeSecurity::for_current_user()?;
            let initial = create_pipe(&pipe_path, &config, &security, true)?;
            let (shutdown, shutdown_receiver) = watch::channel(false);
            let (failure_sender, failure) = watch::channel(None);
            let task_path = pipe_path.clone();
            let task = tokio::spawn(async move {
                let result = run_listener(
                    task_path,
                    config,
                    registry,
                    security,
                    initial,
                    shutdown_receiver,
                )
                .await;
                if let Err(error) = &result {
                    failure_sender.send_replace(Some(error.to_string()));
                }
                result
            });
            Ok(Self {
                pipe_path,
                shutdown,
                failure,
                task: Some(task),
            })
        }

        pub fn pipe_path(&self) -> &str {
            &self.pipe_path
        }

        pub fn failure_signal(
            &self,
        ) -> impl std::future::Future<Output = io::Result<()>> + Send + 'static {
            let mut failure = self.failure.clone();
            async move {
                loop {
                    if let Some(message) = failure.borrow().clone() {
                        return Err(io::Error::other(message));
                    }
                    failure.changed().await.map_err(|_| {
                        io::Error::other("Runtime scene pipe listener stopped unexpectedly")
                    })?;
                }
            }
        }

        pub async fn shutdown(mut self) -> Result<()> {
            self.shutdown.send_replace(true);
            let Some(task) = self.task.take() else {
                return Ok(());
            };
            task.await.map_err(|error| {
                BrainregiondError::Protocol(format!(
                    "Runtime scene pipe listener task failed: {error}"
                ))
            })?
        }
    }

    impl Drop for ScenePipeListener {
        fn drop(&mut self) {
            self.shutdown.send_replace(true);
            if let Some(task) = self.task.take() {
                task.abort();
            }
        }
    }

    async fn run_listener(
        pipe_path: String,
        config: ScenePipeConfig,
        registry: ScenePeerRegistry,
        security: PipeSecurity,
        mut listening: NamedPipeServer,
        mut shutdown: watch::Receiver<bool>,
    ) -> Result<()> {
        let permits = Arc::new(Semaphore::new(config.max_connections));
        let policy = PairingPolicy::from(&config);
        let mut connections = JoinSet::new();

        loop {
            let permit = tokio::select! {
                biased;
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        break;
                    }
                    continue;
                }
                permit = Arc::clone(&permits).acquire_owned() => {
                    permit.map_err(|_| BrainregiondError::Protocol(
                        "Runtime scene pipe connection limiter closed".to_owned()
                    ))?
                }
            };

            let connect_result = tokio::select! {
                biased;
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        drop(permit);
                        break;
                    }
                    continue;
                }
                result = listening.connect() => result,
            };
            connect_result?;

            let connected = listening;
            listening = create_pipe(&pipe_path, &config, &security, false)?;
            let connection_registry = registry.clone();
            let connection_policy = policy.clone();
            connections.spawn(async move {
                let _permit = permit;
                match authenticate_scene_peer(connection_registry, connection_policy, connected)
                    .await
                {
                    Ok(handle) => {
                        let mut state = handle.subscribe_state();
                        while *state.borrow() == ScenePeerState::Connected {
                            if state.changed().await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(error) => {
                        eprintln!("brainregiond: Runtime scene pipe connection rejected: {error}");
                    }
                }
            });

            while connections.try_join_next().is_some() {}
        }

        drop(listening);
        connections.abort_all();
        while connections.join_next().await.is_some() {}
        Ok(())
    }

    fn create_pipe(
        pipe_path: &str,
        config: &ScenePipeConfig,
        security: &PipeSecurity,
        first_instance: bool,
    ) -> Result<NamedPipeServer> {
        let maximum_instances = config.max_connections.checked_add(1).ok_or_else(|| {
            BrainregiondError::Config("scene pipe maximum connection count overflowed".to_owned())
        })?;
        let mut options = ServerOptions::new();
        options
            .first_pipe_instance(first_instance)
            .reject_remote_clients(true)
            .max_instances(maximum_instances)
            .in_buffer_size(64 * 1024)
            .out_buffer_size(64 * 1024);
        let mut attributes = security.attributes();
        // SAFETY: attributes points to a valid SECURITY_ATTRIBUTES whose owned,
        // immutable descriptor outlives this synchronous CreateNamedPipeW call.
        unsafe {
            options
                .create_with_security_attributes_raw(
                    pipe_path,
                    (&mut attributes as *mut SECURITY_ATTRIBUTES).cast::<c_void>(),
                )
                .map_err(Into::into)
        }
    }

    struct PipeSecurity {
        descriptor: *mut c_void,
    }

    // SAFETY: the descriptor returned by LocalAlloc is self-relative and is only
    // read while creating pipe instances. Ownership remains unique in this type.
    unsafe impl Send for PipeSecurity {}

    impl PipeSecurity {
        fn for_current_user() -> Result<Self> {
            let sid = current_user_sid_string()?;
            let sddl = format!("O:{sid}D:P(A;;GA;;;{sid})");
            let encoded: Vec<u16> = sddl.encode_utf16().chain(Some(0)).collect();
            let mut descriptor = ptr::null_mut();
            // SAFETY: encoded is NUL-terminated and descriptor is a valid output pointer.
            let converted = unsafe {
                ConvertStringSecurityDescriptorToSecurityDescriptorW(
                    encoded.as_ptr(),
                    SDDL_REVISION_1,
                    &mut descriptor,
                    ptr::null_mut(),
                )
            };
            if converted == 0 {
                return Err(io::Error::last_os_error().into());
            }
            Ok(Self { descriptor })
        }

        fn attributes(&self) -> SECURITY_ATTRIBUTES {
            SECURITY_ATTRIBUTES {
                nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: self.descriptor,
                bInheritHandle: 0,
            }
        }
    }

    impl Drop for PipeSecurity {
        fn drop(&mut self) {
            if !self.descriptor.is_null() {
                // SAFETY: descriptor was allocated by LocalAlloc through the
                // ConvertStringSecurityDescriptor API and is owned by this type.
                unsafe {
                    LocalFree(self.descriptor);
                }
            }
        }
    }

    struct OwnedHandle(HANDLE);

    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: handle was returned by OpenProcessToken and is uniquely owned.
                unsafe {
                    CloseHandle(self.0);
                }
            }
        }
    }

    fn current_user_sid_string() -> Result<String> {
        let mut raw_token = ptr::null_mut();
        // SAFETY: raw_token is a valid output pointer and pseudo process handle is valid.
        if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut raw_token) } == 0 {
            return Err(io::Error::last_os_error().into());
        }
        let token = OwnedHandle(raw_token);

        let mut required = 0_u32;
        // SAFETY: a null buffer with size 0 is the documented size-query call.
        let first =
            unsafe { GetTokenInformation(token.0, TokenUser, ptr::null_mut(), 0, &mut required) };
        if first != 0
            || required == 0
            || io::Error::last_os_error().raw_os_error() != Some(ERROR_INSUFFICIENT_BUFFER as i32)
        {
            return Err(io::Error::last_os_error().into());
        }
        let words = (required as usize).div_ceil(size_of::<usize>());
        let mut storage = vec![0_usize; words];
        // SAFETY: storage is aligned and has at least `required` writable bytes.
        if unsafe {
            GetTokenInformation(
                token.0,
                TokenUser,
                storage.as_mut_ptr().cast(),
                required,
                &mut required,
            )
        } == 0
        {
            return Err(io::Error::last_os_error().into());
        }
        // SAFETY: successful TokenUser query initialized a TOKEN_USER at storage start.
        let token_user = unsafe { &*storage.as_ptr().cast::<TOKEN_USER>() };
        let mut sid_text: PWSTR = ptr::null_mut();
        // SAFETY: token_user SID remains alive in storage and sid_text is an output pointer.
        if unsafe { ConvertSidToStringSidW(token_user.User.Sid, &mut sid_text) } == 0 {
            return Err(io::Error::last_os_error().into());
        }
        let sid = OwnedLocalString(sid_text);
        sid.to_string()
    }

    struct OwnedLocalString(PWSTR);

    impl OwnedLocalString {
        fn to_string(&self) -> Result<String> {
            let mut length = 0_usize;
            // SAFETY: pointer is a NUL-terminated UTF-16 string returned by Win32.
            unsafe {
                while *self.0.add(length) != 0 {
                    length += 1;
                }
                String::from_utf16(std::slice::from_raw_parts(self.0, length)).map_err(|error| {
                    BrainregiondError::Protocol(format!(
                        "current user SID is not valid UTF-16: {error}"
                    ))
                })
            }
        }
    }

    impl Drop for OwnedLocalString {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: string was allocated by LocalAlloc through ConvertSidToStringSidW.
                unsafe {
                    LocalFree(self.0.cast());
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn current_user_descriptor_has_one_explicit_allow_ace() {
            let sid = current_user_sid_string().unwrap();
            assert!(sid.starts_with("S-1-"));
            let sddl = format!("O:{sid}D:P(A;;GA;;;{sid})");
            assert_eq!(sddl.matches("(A;;GA;;;").count(), 1);
            PipeSecurity::for_current_user().unwrap();
        }

        #[test]
        fn security_attributes_are_not_inheritable() {
            let security = PipeSecurity::for_current_user().unwrap();
            let attributes = security.attributes();
            assert_eq!(
                attributes.nLength as usize,
                std::mem::size_of_val(&attributes)
            );
            assert_eq!(attributes.bInheritHandle, 0);
            assert!(!attributes.lpSecurityDescriptor.is_null());
        }
    }
}

#[cfg(not(windows))]
mod platform {
    use std::io;

    use super::*;

    pub struct ScenePipeListener;

    impl ScenePipeListener {
        pub fn start(_config: ScenePipeConfig, _registry: ScenePeerRegistry) -> Result<Self> {
            Err(BrainregiondError::Config(
                "Runtime scene named pipes are only supported on Windows".to_owned(),
            ))
        }

        pub fn pipe_path(&self) -> &str {
            ""
        }

        pub fn failure_signal(
            &self,
        ) -> impl std::future::Future<Output = io::Result<()>> + Send + 'static {
            std::future::pending()
        }

        pub async fn shutdown(self) -> Result<()> {
            Ok(())
        }
    }
}

pub use platform::ScenePipeListener;
