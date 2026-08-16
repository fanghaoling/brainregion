#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
using System;
using System.ComponentModel;
using System.IO;
using System.IO.Pipes;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace BrainRegion.RuntimeBridge
{
    /// <summary>
    /// Opt-in Windows Player transport for brainregiond's current-user named pipe.
    /// All pipe I/O and authentication run off the Unity main thread. The main
    /// thread only builds registration snapshots and drains SceneRpcDispatcher.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class WindowsScenePipeTransport : MonoBehaviour
    {
        private const int MaximumFrameBytes = 1024 * 1024;
        private const int MinimumPairingSecretBytes = 32;
        private const int MaximumPairingSecretBytes = 4096;
        private const string PipeNameEnvironmentVariable = "BRAINREGIOND_SCENE_PIPE_NAME";
        private const string PairingSecretEnvironmentVariable =
            "BRAINREGIOND_SCENE_PAIRING_SECRET";
        private const int MaximumIoFramesPerDirectionPerPoll = 32;

        private static long processConnectionEpoch;

        [SerializeField] private SceneRpcDispatcher dispatcher;
        [SerializeField] private bool connectOnEnable;
        [SerializeField] private string pipeName;
        [SerializeField, Range(250, 30000)] private int connectTimeoutMs = 5000;
        [SerializeField, Range(250, 30000)] private int retryDelayMs = 2000;
        [SerializeField, Range(8, 512)] private int maximumQueuedFrames = 128;
        [SerializeField, Range(1, 16)] private int maximumQueuedMiB = 4;

        private readonly object stateGate = new object();
        private byte[] configuredPairingSecret;
        private CancellationTokenSource attemptCancellation;
        private Task connectionTask;
        private NamedPipeClientStream currentPipe;
        private PipeConnection activeConnection;
        private long retryAfterUtcTicks;
        private int wantsConnection;
        private int connected;
        private string lastError;

        public bool IsConnected => Volatile.Read(ref connected) != 0;
        public string LastError => Volatile.Read(ref lastError);

        private void Awake()
        {
            if (dispatcher == null) dispatcher = GetComponent<SceneRpcDispatcher>();
        }

        private void OnEnable()
        {
            if (dispatcher != null) dispatcher.OutboundNotification += OnOutboundNotification;
            if (connectOnEnable) Connect();
        }

        private void OnDisable()
        {
            if (dispatcher != null) dispatcher.OutboundNotification -= OnOutboundNotification;
            Disconnect();
        }

        private void OnDestroy()
        {
            lock (stateGate)
            {
                ClearBytes(configuredPairingSecret);
                configuredPairingSecret = null;
            }
        }

        private void Update()
        {
            ObserveCompletedAttempt();
            if (Volatile.Read(ref wantsConnection) == 0) return;
            if (dispatcher == null)
            {
                Volatile.Write(ref lastError, "SceneRpcDispatcher is missing");
                Volatile.Write(ref wantsConnection, 0);
                return;
            }
            if (DateTime.UtcNow.Ticks < Interlocked.Read(ref retryAfterUtcTicks)) return;

            lock (stateGate)
            {
                if (connectionTask != null || Volatile.Read(ref wantsConnection) == 0) return;
                if (!TryResolveSettings(out AttemptSettings settings, out string error))
                {
                    Volatile.Write(ref lastError, error);
                    Interlocked.Exchange(
                        ref retryAfterUtcTicks,
                        DateTime.UtcNow.AddMilliseconds(Math.Max(250, retryDelayMs)).Ticks);
                    return;
                }

                try
                {
                    settings.BaseRegistrationJson = dispatcher.BuildRegistrationNotification();
                }
                catch (Exception exception)
                {
                    ClearBytes(settings.PairingSecret);
                    Volatile.Write(ref lastError, $"Could not build Runtime registration: {exception.Message}");
                    Interlocked.Exchange(
                        ref retryAfterUtcTicks,
                        DateTime.UtcNow.AddMilliseconds(Math.Max(250, retryDelayMs)).Ticks);
                    return;
                }

                long connectionEpoch = Interlocked.Increment(ref processConnectionEpoch);
                if (connectionEpoch <= 0)
                {
                    ClearBytes(settings.PairingSecret);
                    Volatile.Write(ref lastError, "Scene pipe connection epoch is exhausted");
                    Volatile.Write(ref wantsConnection, 0);
                    return;
                }
                settings.ConnectionEpoch = connectionEpoch;
                attemptCancellation = new CancellationTokenSource();
                CancellationToken cancellationToken = attemptCancellation.Token;
                connectionTask = Task.Factory.StartNew(
                    () => RunAttempt(settings, cancellationToken),
                    CancellationToken.None,
                    TaskCreationOptions.LongRunning,
                    TaskScheduler.Default);
            }
        }

        public void Connect()
        {
            Volatile.Write(ref lastError, null);
            Interlocked.Exchange(ref retryAfterUtcTicks, 0);
            Volatile.Write(ref wantsConnection, 1);
        }

        public void Disconnect()
        {
            Volatile.Write(ref wantsConnection, 0);
            Volatile.Write(ref connected, 0);
            lock (stateGate)
            {
                attemptCancellation?.Cancel();
                activeConnection?.Abort();
                try { currentPipe?.Dispose(); }
                catch (ObjectDisposedException) { }
            }
        }

        /// <summary>
        /// Inject a high-entropy secret at runtime. The value is never serialized.
        /// Passing null clears the injected value and restores environment lookup.
        /// </summary>
        public void SetPairingSecret(string secret)
        {
            byte[] replacement = secret == null ? null : Encoding.UTF8.GetBytes(secret);
            if (replacement != null &&
                (replacement.Length < MinimumPairingSecretBytes ||
                 replacement.Length > MaximumPairingSecretBytes))
            {
                ClearBytes(replacement);
                throw new ArgumentException(
                    $"Pairing secret must contain {MinimumPairingSecretBytes}..{MaximumPairingSecretBytes} UTF-8 bytes",
                    nameof(secret));
            }
            lock (stateGate)
            {
                ClearBytes(configuredPairingSecret);
                configuredPairingSecret = replacement;
            }
        }

        private void ObserveCompletedAttempt()
        {
            lock (stateGate)
            {
                if (connectionTask == null || !connectionTask.IsCompleted) return;
                // Observe a theoretically unexpected wrapper fault. Normal transport
                // failures are caught inside RunAttemptAsync and surfaced in LastError.
                _ = connectionTask.Exception;
                connectionTask = null;
                attemptCancellation?.Dispose();
                attemptCancellation = null;
            }
        }

        private bool TryResolveSettings(out AttemptSettings settings, out string error)
        {
            settings = default;
            error = null;
            string resolvedPipeName;
            try
            {
                resolvedPipeName = Environment.GetEnvironmentVariable(
                    PipeNameEnvironmentVariable);
            }
            catch (Exception exception) when (
                exception is ArgumentException || exception is System.Security.SecurityException)
            {
                error = $"Could not read scene pipe environment variable: {exception.Message}";
                return false;
            }
            if (string.IsNullOrEmpty(resolvedPipeName)) resolvedPipeName = pipeName;
            if (!IsPipeName(resolvedPipeName))
            {
                error = "Scene pipe name is missing or contains unsupported characters";
                return false;
            }

            byte[] secret = configuredPairingSecret == null
                ? null
                : (byte[])configuredPairingSecret.Clone();
            if (secret == null)
            {
                string environmentSecret;
                try
                {
                    environmentSecret = Environment.GetEnvironmentVariable(
                        PairingSecretEnvironmentVariable);
                }
                catch (Exception exception) when (
                    exception is ArgumentException || exception is System.Security.SecurityException)
                {
                    error = $"Could not read pairing-secret environment variable: {exception.Message}";
                    return false;
                }
                if (!string.IsNullOrEmpty(environmentSecret))
                    secret = Encoding.UTF8.GetBytes(environmentSecret);
            }
            if (secret == null || secret.Length < MinimumPairingSecretBytes ||
                secret.Length > MaximumPairingSecretBytes)
            {
                ClearBytes(secret);
                error = "Scene pairing secret is missing or outside the 32..4096 byte range";
                return false;
            }

            settings = new AttemptSettings
            {
                PipeName = resolvedPipeName,
                PairingSecret = secret,
                ConnectTimeoutMs = Math.Min(30000, Math.Max(250, connectTimeoutMs)),
                RetryDelayMs = Math.Min(30000, Math.Max(250, retryDelayMs)),
                MaximumQueuedFrames = Math.Min(512, Math.Max(8, maximumQueuedFrames)),
                MaximumQueuedBytes = Math.Min(16, Math.Max(1, maximumQueuedMiB)) * 1024 * 1024,
            };
            return true;
        }

        private void RunAttempt(
            AttemptSettings settings,
            CancellationToken cancellationToken)
        {
            PipeConnection connection = null;
            NamedPipeClientStream pipe = null;
            try
            {
                pipe = new NamedPipeClientStream(
                    ".",
                    settings.PipeName,
                    PipeDirection.InOut,
                    PipeOptions.None);
                lock (stateGate)
                {
                    if (cancellationToken.IsCancellationRequested) return;
                    currentPipe = pipe;
                }
                pipe.Connect(settings.ConnectTimeoutMs);
                cancellationToken.ThrowIfCancellationRequested();

                var reader = new BoundedJsonLineReader(pipe, MaximumFrameBytes);
                string challengeJson = reader.ReadLine();
                cancellationToken.ThrowIfCancellationRequested();
                if (challengeJson == null)
                    throw new EndOfStreamException("Scene pipe closed before pairing challenge");
                if (!ScenePairingProof.TryParseChallenge(
                        challengeJson,
                        out ScenePairingChallenge challenge,
                        out string challengeError))
                    throw new InvalidDataException($"Pairing challenge rejected: {challengeError}");
                if (!ScenePairingProof.TryCreateRegistration(
                        settings.BaseRegistrationJson,
                        challenge,
                        settings.PairingSecret,
                        out string registrationJson,
                        out string registrationError))
                    throw new InvalidDataException($"Runtime registration rejected: {registrationError}");
                WriteFrame(pipe, registrationJson);
                cancellationToken.ThrowIfCancellationRequested();
                ClearBytes(settings.PairingSecret);
                settings.PairingSecret = null;

                var peer = new AuthenticatedPeerContext(
                    challenge.PrincipalId,
                    settings.ConnectionEpoch,
                    challenge.GrantedCapabilities);
                connection = new PipeConnection(
                    pipe,
                    settings.MaximumQueuedFrames,
                    settings.MaximumQueuedBytes,
                    cancellationToken);
                lock (stateGate)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    activeConnection = connection;
                }
                Volatile.Write(ref connected, 1);
                Volatile.Write(ref lastError, null);

                while (!connection.Token.IsCancellationRequested)
                {
                    int outboundProcessed = 0;
                    while (outboundProcessed < MaximumIoFramesPerDirectionPerPoll &&
                           connection.TryDequeue(out string outbound))
                    {
                        WriteFrame(pipe, outbound);
                        outboundProcessed++;
                    }

                    int inboundProcessed = 0;
                    while (inboundProcessed < MaximumIoFramesPerDirectionPerPoll)
                    {
                        if (!reader.TryReadLine(0, out string request))
                        {
                            uint availableBytes = GetAvailableBytes(pipe);
                            if (availableBytes == 0 || !reader.TryReadLine(
                                    checked((int)Math.Min(availableBytes, int.MaxValue)),
                                    out request))
                                break;
                        }
                        connection.Token.ThrowIfCancellationRequested();
                        bool accepted = dispatcher.TryEnqueue(
                            peer,
                            request,
                            response => QueueOutbound(connection, response),
                            out string immediateResponse);
                        if (!accepted && immediateResponse != null)
                            QueueOutbound(connection, immediateResponse);
                        inboundProcessed++;
                    }

                    if (outboundProcessed == 0 && inboundProcessed == 0 &&
                        connection.Token.WaitHandle.WaitOne(2))
                        connection.Token.ThrowIfCancellationRequested();
                }
            }
            catch (Exception exception) when (
                exception is OperationCanceledException || exception is IOException ||
                exception is InvalidDataException || exception is UnauthorizedAccessException ||
                exception is TimeoutException || exception is ObjectDisposedException ||
                exception is ArgumentException)
            {
                if (Volatile.Read(ref wantsConnection) != 0 && !cancellationToken.IsCancellationRequested)
                    Volatile.Write(ref lastError, exception.Message);
            }
            catch (Exception exception)
            {
                if (Volatile.Read(ref wantsConnection) != 0 && !cancellationToken.IsCancellationRequested)
                    Volatile.Write(ref lastError, $"Unexpected scene pipe failure: {exception.Message}");
            }
            finally
            {
                ClearBytes(settings.PairingSecret);
                connection?.Abort();
                lock (stateGate)
                {
                    if (ReferenceEquals(activeConnection, connection)) activeConnection = null;
                    if (ReferenceEquals(currentPipe, pipe)) currentPipe = null;
                }
                connection?.Dispose();
                pipe?.Dispose();
                Volatile.Write(ref connected, 0);
                if (Volatile.Read(ref wantsConnection) != 0)
                {
                    Interlocked.Exchange(
                        ref retryAfterUtcTicks,
                        DateTime.UtcNow.AddMilliseconds(settings.RetryDelayMs).Ticks);
                }
            }
        }

        private void OnOutboundNotification(string notification)
        {
            PipeConnection connection;
            lock (stateGate) connection = activeConnection;
            if (connection != null) QueueOutbound(connection, notification);
        }

        private static void QueueOutbound(PipeConnection connection, string json)
        {
            if (connection == null || string.IsNullOrEmpty(json) ||
                json.IndexOf('\n') >= 0 || json.IndexOf('\r') >= 0)
            {
                connection?.Abort();
                return;
            }
            if (Encoding.UTF8.GetByteCount(json) > MaximumFrameBytes)
                json = BuildOversizedResponse(json);
            if (json == null || !connection.TryEnqueue(json)) connection.Abort();
        }

        private static string BuildOversizedResponse(string original)
        {
            try
            {
                JObject value = JObject.Parse(original);
                JToken id = value["id"];
                if (id == null) return null;
                return new JObject
                {
                    ["jsonrpc"] = "2.0",
                    ["id"] = id.DeepClone(),
                    ["error"] = new JObject
                    {
                        ["code"] = -32603,
                        ["message"] = $"Scene RPC response exceeds {MaximumFrameBytes} bytes",
                        ["data"] = new JObject
                        {
                            ["reason"] = "response_too_large",
                            ["retryable"] = false,
                        },
                    },
                }.ToString(Formatting.None);
            }
            catch (JsonException)
            {
                return null;
            }
        }

        private static void WriteFrame(Stream stream, string json)
        {
            if (json == null || json.IndexOf('\n') >= 0 || json.IndexOf('\r') >= 0)
                throw new InvalidDataException("Scene pipe frame must be one JSON line");
            byte[] encoded = Encoding.UTF8.GetBytes(json);
            if (encoded.Length > MaximumFrameBytes)
                throw new InvalidDataException($"Scene pipe frame exceeds {MaximumFrameBytes} bytes");
            byte[] framed = new byte[encoded.Length + 1];
            Buffer.BlockCopy(encoded, 0, framed, 0, encoded.Length);
            framed[framed.Length - 1] = (byte)'\n';
            stream.Write(framed, 0, framed.Length);
            stream.Flush();
        }

        private static uint GetAvailableBytes(NamedPipeClientStream pipe)
        {
            if (PeekNamedPipe(
                    pipe.SafePipeHandle,
                    IntPtr.Zero,
                    0,
                    IntPtr.Zero,
                    out uint availableBytes,
                    IntPtr.Zero))
                return availableBytes;
            int error = Marshal.GetLastWin32Error();
            throw new IOException(
                $"Could not inspect Runtime scene pipe: {new Win32Exception(error).Message}",
                error);
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool PeekNamedPipe(
            Microsoft.Win32.SafeHandles.SafePipeHandle pipe,
            IntPtr buffer,
            uint bufferSize,
            IntPtr bytesRead,
            out uint totalBytesAvailable,
            IntPtr bytesLeftThisMessage);

        private static bool IsPipeName(string value)
        {
            if (string.IsNullOrEmpty(value) || value.Length > 128) return false;
            foreach (char character in value)
            {
                bool valid = character <= 127 &&
                    (char.IsLetterOrDigit(character) || character == '.' ||
                     character == '_' || character == '-');
                if (!valid) return false;
            }
            return true;
        }

        private static void ClearBytes(byte[] value)
        {
            if (value != null) Array.Clear(value, 0, value.Length);
        }

        private struct AttemptSettings
        {
            public string PipeName;
            public byte[] PairingSecret;
            public string BaseRegistrationJson;
            public int ConnectTimeoutMs;
            public int RetryDelayMs;
            public int MaximumQueuedFrames;
            public int MaximumQueuedBytes;
            public long ConnectionEpoch;
        }

        private sealed class PipeConnection : IDisposable
        {
            private readonly NamedPipeClientStream pipe;
            private readonly BoundedSceneWriterQueue writerQueue;
            private readonly CancellationTokenSource cancellation;
            private int accepting = 1;

            public CancellationToken Token => cancellation.Token;

            public PipeConnection(
                NamedPipeClientStream pipe,
                int maximumQueuedFrames,
                int maximumQueuedBytes,
                CancellationToken parentCancellation)
            {
                this.pipe = pipe;
                writerQueue = new BoundedSceneWriterQueue(
                    maximumQueuedFrames,
                    maximumQueuedBytes);
                cancellation = CancellationTokenSource.CreateLinkedTokenSource(parentCancellation);
            }

            public bool TryEnqueue(string json)
            {
                return Volatile.Read(ref accepting) != 0 &&
                    writerQueue.TryEnqueue(json, MaximumFrameBytes);
            }

            public bool TryDequeue(out string json)
            {
                json = null;
                return Volatile.Read(ref accepting) != 0 && writerQueue.TryDequeue(out json);
            }

            public void Abort()
            {
                if (Interlocked.Exchange(ref accepting, 0) == 0) return;
                cancellation.Cancel();
                try { pipe.Dispose(); }
                catch (ObjectDisposedException) { }
            }

            public void Dispose()
            {
                Abort();
                writerQueue.Dispose();
                cancellation.Dispose();
            }
        }
    }
}
#else
using System;
using System.Text;
using UnityEngine;

namespace BrainRegion.RuntimeBridge
{
    [DisallowMultipleComponent]
    public sealed class WindowsScenePipeTransport : MonoBehaviour
    {
        public bool IsConnected => false;
        public string LastError => "Windows named pipes are unavailable on this platform";
        public void Connect() { }
        public void Disconnect() { }
        public void SetPairingSecret(string secret)
        {
            if (secret == null) return;
            int bytes = Encoding.UTF8.GetByteCount(secret);
            if (bytes < 32 || bytes > 4096)
                throw new ArgumentException(
                    "Pairing secret must contain 32..4096 UTF-8 bytes",
                    nameof(secret));
        }
    }
}
#endif
