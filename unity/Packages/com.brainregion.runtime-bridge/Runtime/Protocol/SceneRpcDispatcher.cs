using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace BrainRegion.RuntimeBridge
{
    /// <summary>
    /// Transport-neutral JSON-RPC dispatcher. Network code may enqueue from any
    /// thread; JSON and typed DTO parsing happen before enqueue, while all access to
    /// Unity state and all command execution occur in Update under a frame budget.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class SceneRpcDispatcher : MonoBehaviour
    {
        public const int MaxFrameBytes = 1024 * 1024;
        private const int MaxAuthorizedPreviews = 512;
        private const int MaxAppliedAuthorizations = 256;
        private const int MaxDeferredPersistenceOperations = 8;

        [SerializeField] private RuntimeSceneController controller;
        [SerializeField] private RuntimeLogBuffer logBuffer;
        [SerializeField, Range(1, 64)] private int maxRequestsPerFrame = 8;
        [SerializeField, Range(0.05f, 5f)] private float maxMillisecondsPerFrame = 0.5f;
        [SerializeField, Range(8, 1024)] private int maxQueuedRequests = 128;

        private readonly ConcurrentQueue<QueuedRequest> queue = new ConcurrentQueue<QueuedRequest>();
        private readonly ConcurrentQueue<CompletedPersistenceRequest> completedPersistence =
            new ConcurrentQueue<CompletedPersistenceRequest>();
        private readonly Dictionary<long, DeferredPersistenceRequest> activePersistence =
            new Dictionary<long, DeferredPersistenceRequest>();
        private readonly object lifecycleGate = new object();
        private readonly object peerEpochGate = new object();
        private readonly Dictionary<string, PeerEpochState> latestPeerEpochs =
            new Dictionary<string, PeerEpochState>(StringComparer.Ordinal);
        private readonly Dictionary<string, PreviewAuthorization> previewAuthorizations =
            new Dictionary<string, PreviewAuthorization>(StringComparer.Ordinal);
        private readonly Dictionary<string, AppliedAuthorization> appliedAuthorizations =
            new Dictionary<string, AppliedAuthorization>(StringComparer.Ordinal);
        private readonly Queue<string> appliedAuthorizationOrder = new Queue<string>();
        private int queuedCount;
        private int mainThreadId;
        private bool acceptingRequests;
        private bool hasStarted;
        private string cachedRegistrationNotification;
        private long nextPersistenceOperationId;

        public event Action<string> OutboundNotification;

        private void Awake()
        {
            mainThreadId = Thread.CurrentThread.ManagedThreadId;
            if (controller == null) controller = GetComponent<RuntimeSceneController>();
            if (logBuffer == null) logBuffer = GetComponent<RuntimeLogBuffer>();
            if (controller == null)
                enabled = false;
        }

        private void OnEnable()
        {
            if (controller != null) controller.SceneChanged += OnSceneChanged;
            lock (lifecycleGate)
                acceptingRequests = controller != null;
            if (hasStarted && controller != null)
                RefreshRegistrationNotification();
        }

        private void Start()
        {
            hasStarted = true;
            if (controller != null)
                RefreshRegistrationNotification();
        }

        private void OnDisable()
        {
            List<QueuedRequest> abandoned = new List<QueuedRequest>();
            lock (lifecycleGate)
            {
                acceptingRequests = false;
                while (queue.TryDequeue(out QueuedRequest queued))
                {
                    Interlocked.Decrement(ref queuedCount);
                    abandoned.Add(queued);
                }
            }
            if (controller != null) controller.SceneChanged -= OnSceneChanged;

            lock (peerEpochGate)
                latestPeerEpochs.Clear();
            previewAuthorizations.Clear();
            appliedAuthorizations.Clear();
            appliedAuthorizationOrder.Clear();
            Volatile.Write(ref cachedRegistrationNotification, null);

            foreach (QueuedRequest queued in abandoned)
            {
                CompleteSafely(queued, SerializeError(
                    queued.Request.Id,
                    RpcFailure.Create(
                        SceneErrorCodes.Busy,
                        "Scene RPC dispatcher stopped before this request could execute",
                        "dispatcher_stopped",
                        true)));
            }
            foreach (DeferredPersistenceRequest deferred in activePersistence.Values)
            {
                RpcFailure stopped = deferred.Work.Kind ==
                    RuntimeSceneController.WorldPersistenceWorkKind.Save
                    ? PersistenceOutcomeUnknownFailure()
                    : RpcFailure.Create(
                        SceneErrorCodes.Busy,
                        "Scene RPC dispatcher stopped during background persistence I/O",
                        "dispatcher_stopped",
                        true);
                deferred.TryComplete(SerializeError(deferred.Request.Request.Id, stopped));
            }
        }

        /// <summary>
        /// Enqueue one UTF-8 JSON request. On rejection, immediateResponse is a
        /// complete JSON-RPC error that the transport can send without touching Unity.
        /// The completion callback runs on the Unity main thread and should only enqueue
        /// the response into the transport's writer queue.
        /// </summary>
        public bool TryEnqueue(
            AuthenticatedPeerContext peer,
            string requestJson,
            Action<string> completion,
            out string immediateResponse)
        {
            immediateResponse = null;
            if (requestJson == null || requestJson.Length > MaxFrameBytes ||
                Encoding.UTF8.GetByteCount(requestJson) > MaxFrameBytes)
            {
                immediateResponse = SerializeError(
                    JValue.CreateNull(),
                    RpcFailure.Create(
                        SceneErrorCodes.InvalidRequest,
                        $"Scene RPC frame exceeds {MaxFrameBytes} bytes",
                        "frame_too_large"));
                return false;
            }

            if (peer == null)
            {
                immediateResponse = SerializeError(
                    JValue.CreateNull(),
                    RpcFailure.Create(
                        SceneErrorCodes.Unauthenticated,
                        "Scene RPC requires an authenticated peer context",
                        "peer_context_required"));
                return false;
            }

            if (!TryPrepareRequest(peer, requestJson, out PreparedRequest request, out immediateResponse))
                return false;

            lock (lifecycleGate)
            {
                if (!acceptingRequests)
                {
                    immediateResponse = SerializeError(
                        request.Id,
                        RpcFailure.Create(
                            SceneErrorCodes.Busy,
                            "Scene RPC dispatcher is not accepting requests",
                            "dispatcher_unavailable",
                            true));
                    return false;
                }
                if (!TryAcceptPeerEpoch(peer, out RpcFailure epochFailure))
                {
                    immediateResponse = SerializeError(request.Id, epochFailure);
                    return false;
                }
                if (Volatile.Read(ref queuedCount) >= Math.Max(8, maxQueuedRequests))
                {
                    immediateResponse = SerializeError(
                        request.Id,
                        RpcFailure.Create(
                            SceneErrorCodes.Busy,
                            "Scene RPC main-thread queue is full",
                            "queue_full",
                            true));
                    return false;
                }

                Interlocked.Increment(ref queuedCount);
                queue.Enqueue(new QueuedRequest(request, peer, completion));
            }
            return true;
        }

        /// <summary>
        /// Build a fresh registration on the Unity main thread. Network threads must
        /// use TryGetCachedRegistrationNotification instead of calling Unity APIs.
        /// </summary>
        public string BuildRegistrationNotification()
        {
            AssertMainThread();
            return RefreshRegistrationNotification();
        }

        public bool TryGetCachedRegistrationNotification(out string notification)
        {
            notification = Volatile.Read(ref cachedRegistrationNotification);
            return notification != null;
        }

        private string RefreshRegistrationNotification()
        {
            AssertMainThread();
            JObject info = controller.GetRuntimeInfo();
            if (logBuffer != null && info["capabilities"] is JArray capabilities &&
                !capabilities.Any(item => (string)item == SceneCapabilities.LogsRead))
            {
                capabilities.Add(SceneCapabilities.LogsRead);
            }
            string notification = new JObject
            {
                ["jsonrpc"] = "2.0",
                ["method"] = "runtime/register",
                ["params"] = info,
            }.ToString(Formatting.None);
            Volatile.Write(ref cachedRegistrationNotification, notification);
            return notification;
        }

        private void Update()
        {
            var stopwatch = Stopwatch.StartNew();
            int processed = 0;
            while (processed < Mathf.Max(1, maxRequestsPerFrame) &&
                   stopwatch.Elapsed.TotalMilliseconds <= Mathf.Max(0.05f, maxMillisecondsPerFrame) &&
                   completedPersistence.TryDequeue(out CompletedPersistenceRequest completed))
            {
                CompleteDeferredPersistence(completed);
                processed++;
            }
            while (processed < Mathf.Max(1, maxRequestsPerFrame) &&
                   stopwatch.Elapsed.TotalMilliseconds <= Mathf.Max(0.05f, maxMillisecondsPerFrame) &&
                   queue.TryDequeue(out QueuedRequest queued))
            {
                Interlocked.Decrement(ref queuedCount);
                if (TryStartDeferredPersistence(queued, out string deferredResponse))
                {
                    if (deferredResponse != null)
                        CompleteSafely(queued, deferredResponse);
                    processed++;
                    continue;
                }
                string response = Dispatch(queued.Request, queued.Peer);
                CompleteSafely(queued, response);
                processed++;
            }
        }

        private bool TryStartDeferredPersistence(
            QueuedRequest queued,
            out string immediateResponse)
        {
            immediateResponse = null;
            string method = queued.Request.Method;
            if (method != "persistence/list" &&
                method != "persistence/save" &&
                method != "persistence/loadPreview")
                return false;

            if (!IsCurrentPeerEpoch(queued.Peer))
            {
                immediateResponse = SerializeError(
                    queued.Request.Id,
                    StaleConnectionFailure());
                return true;
            }
            if (queued.Request.DeadlineUnixMs < DateTimeOffset.UtcNow.ToUnixTimeMilliseconds())
            {
                immediateResponse = SerializeError(
                    queued.Request.Id,
                    RpcFailure.Create(
                        SceneErrorCodes.ValidationFailed,
                        "Scene RPC request deadline elapsed before main-thread execution",
                        "deadline_elapsed",
                        true));
                return true;
            }
            if (activePersistence.Count >= MaxDeferredPersistenceOperations)
            {
                immediateResponse = SerializeError(
                    queued.Request.Id,
                    RpcFailure.Create(
                        SceneErrorCodes.Busy,
                        "Too many WorldDocument I/O operations are active",
                        "persistence_queue_full",
                        true));
                return true;
            }

            RuntimeSceneController.WorldPersistenceWork work;
            JObject immediateResult = null;
            RpcFailure failure = null;
            switch (method)
            {
                case "persistence/list":
                    work = controller.BeginWorldSlotList();
                    break;
                case "persistence/save":
                    if (!controller.TryBeginWorldSave(
                            queued.Peer.PrincipalId,
                            (WorldSaveRequest)queued.Request.Parameters,
                            out work,
                            out immediateResult,
                            out failure))
                    {
                        immediateResponse = SerializeError(queued.Request.Id, failure);
                        return true;
                    }
                    if (work == null)
                    {
                        immediateResponse = SerializeSuccess(queued.Request.Id, immediateResult);
                        return true;
                    }
                    break;
                case "persistence/loadPreview":
                    if (!controller.TryBeginWorldLoadPreview(
                            queued.Peer.PrincipalId,
                            queued.Peer.ConnectionEpoch,
                            (WorldLoadPreviewRequest)queued.Request.Parameters,
                            out work,
                            out failure))
                    {
                        immediateResponse = SerializeError(queued.Request.Id, failure);
                        return true;
                    }
                    break;
                default:
                    throw new InvalidOperationException("Unsupported deferred persistence method");
            }

            work.DeadlineUnixMs = queued.Request.DeadlineUnixMs;
            long operationId = Interlocked.Increment(ref nextPersistenceOperationId);
            var deferred = new DeferredPersistenceRequest(operationId, queued, work);
            activePersistence.Add(operationId, deferred);
            try
            {
                _ = Task.Run(work.Execute).ContinueWith(
                    task =>
                    {
                        RuntimeSceneController.WorldPersistenceWorkerResult worker;
                        if (task.Status == TaskStatus.RanToCompletion)
                        {
                            worker = task.Result;
                        }
                        else
                        {
                            string message = task.Exception?.GetBaseException().Message ??
                                "WorldDocument worker did not complete";
                            worker = new RuntimeSceneController.WorldPersistenceWorkerResult(
                                null,
                                RpcFailure.Create(
                                    SceneErrorCodes.PersistenceError,
                                    "WorldDocument background operation failed: " + message,
                                    "persistence_worker_failed"));
                        }
                        completedPersistence.Enqueue(
                            new CompletedPersistenceRequest(operationId, worker));
                    },
                    CancellationToken.None,
                    TaskContinuationOptions.ExecuteSynchronously,
                    TaskScheduler.Default);
            }
            catch (Exception exception)
            {
                activePersistence.Remove(operationId);
                controller.DiscardWorldPersistenceWork(work);
                immediateResponse = SerializeError(
                    queued.Request.Id,
                    RpcFailure.Create(
                        SceneErrorCodes.Busy,
                        "Could not schedule WorldDocument I/O: " + exception.Message,
                        "persistence_worker_unavailable",
                        true));
            }
            return true;
        }

        private void CompleteDeferredPersistence(CompletedPersistenceRequest completed)
        {
            if (!activePersistence.TryGetValue(
                    completed.OperationId,
                    out DeferredPersistenceRequest deferred))
                return;
            activePersistence.Remove(completed.OperationId);

            RuntimeSceneController.WorldPersistenceWork work = deferred.Work;
            JObject result;
            RpcFailure failure;
            bool succeeded;
            if (work.Kind == RuntimeSceneController.WorldPersistenceWorkKind.Save)
            {
                // A save may already be atomically committed even when the peer or
                // deadline disappeared. Always record its receipt on the main thread.
                succeeded = controller.TryCompleteWorldPersistence(
                    work,
                    completed.Worker,
                    out result,
                    out failure);
            }
            else if (!IsCurrentPeerEpoch(deferred.Request.Peer))
            {
                controller.DiscardWorldPersistenceWork(work);
                succeeded = false;
                result = null;
                failure = StaleConnectionFailure();
            }
            else if (deferred.Request.Request.DeadlineUnixMs <
                     DateTimeOffset.UtcNow.ToUnixTimeMilliseconds())
            {
                controller.DiscardWorldPersistenceWork(work);
                succeeded = false;
                result = null;
                failure = RpcFailure.Create(
                    SceneErrorCodes.ValidationFailed,
                    "Scene RPC deadline elapsed during WorldDocument I/O",
                    "deadline_elapsed",
                    true);
            }
            else
            {
                succeeded = controller.TryCompleteWorldPersistence(
                    work,
                    completed.Worker,
                    out result,
                    out failure);
            }

            deferred.TryComplete(succeeded
                ? SerializeSuccess(deferred.Request.Request.Id, result)
                : SerializeError(deferred.Request.Request.Id, failure));
        }

        private static RpcFailure PersistenceOutcomeUnknownFailure()
        {
            RpcFailure failure = RpcFailure.Create(
                SceneErrorCodes.PersistenceError,
                "WorldDocument save may still complete after the dispatcher stopped",
                "operation_outcome_unknown");
            failure.Data["outcome"] = "unknown";
            failure.Data["retryable"] = false;
            return failure;
        }

        private string Dispatch(PreparedRequest request, AuthenticatedPeerContext peer)
        {
            JToken id = request.Id;
            try
            {
                if (!IsCurrentPeerEpoch(peer))
                {
                    return SerializeError(id, RpcFailure.Create(
                        SceneErrorCodes.Unauthenticated,
                        "Request belongs to a superseded authenticated connection",
                        "stale_connection_epoch"));
                }
                if (request.DeadlineUnixMs < DateTimeOffset.UtcNow.ToUnixTimeMilliseconds())
                {
                    return SerializeError(id, RpcFailure.Create(
                        SceneErrorCodes.ValidationFailed,
                        "Scene RPC request deadline elapsed before main-thread execution",
                        "deadline_elapsed",
                        true));
                }

                switch (request.Method)
                {
                    case "runtime/info":
                        return SerializeSuccess(id, controller.GetRuntimeInfo());

                    case "scene/hierarchy":
                        var hierarchy = (SceneHierarchyRequest)request.Parameters;
                        return controller.TryGetHierarchy(hierarchy, out SceneHierarchyResult hierarchyResult, out RpcFailure hierarchyError)
                            ? SerializeSuccess(id, hierarchyResult)
                            : SerializeError(id, hierarchyError);

                    case "object/inspect":
                        var inspect = (ObjectInspectRequest)request.Parameters;
                        if (!controller.TryInspect(inspect, out RpcObjectSnapshot objectResult, out RpcFailure inspectError))
                            return SerializeError(id, inspectError);
                        return SerializeSuccess(id, new JObject
                        {
                            ["sceneRevision"] = controller.SceneRevision,
                            ["observedFrame"] = Time.frameCount,
                            ["object"] = JToken.FromObject(objectResult),
                        });

                    case "prefab/list":
                        var prefabs = (PrefabListRequest)request.Parameters;
                        return controller.TryListPrefabs(prefabs, out JObject prefabResult, out RpcFailure prefabError)
                            ? SerializeSuccess(id, prefabResult)
                            : SerializeError(id, prefabError);

                    case "scene/preview":
                        CleanupAuthorizationState();
                        if (previewAuthorizations.Count >= MaxAuthorizedPreviews)
                        {
                            return SerializeError(id, RpcFailure.Create(
                                SceneErrorCodes.Busy,
                                "Too many Scene RPC previews are awaiting apply",
                                "preview_authorization_full",
                                true));
                        }
                        var preview = (ScenePreviewRequest)request.Parameters;
                        if (!controller.TryPreview(
                            peer.PrincipalId,
                            preview,
                            out ScenePreviewResult previewResult,
                            out RpcFailure previewError))
                            return SerializeError(id, previewError);
                        previewAuthorizations[previewResult.PreviewToken] = new PreviewAuthorization(
                            peer,
                            previewResult.ClientMutationId,
                            previewResult.ExpiresAtUnixMs,
                            preview.Commands.Any(command => command != null && command.Kind == "spawn"));
                        return SerializeSuccess(id, previewResult);

                    case "scene/apply":
                        CleanupAuthorizationState();
                        var apply = (SceneApplyRequest)request.Parameters;
                        if (!TryAuthorizeApply(peer, apply, out bool requiredSpawn, out RpcFailure applyAuthError))
                            return SerializeError(id, applyAuthError);
                        if (!controller.TryApply(
                            peer.PrincipalId,
                            apply,
                            out SceneMutationResult applyResult,
                            out RpcFailure applyError))
                            return SerializeError(id, applyError);
                        previewAuthorizations.Remove(apply.PreviewToken);
                        RememberAppliedAuthorization(peer, apply, requiredSpawn);
                        return SerializeSuccess(id, applyResult);

                    case "history/undo":
                        var undo = (SceneUndoRequest)request.Parameters;
                        return controller.TryUndo(
                            peer.PrincipalId,
                            undo,
                            out JObject undoResult,
                            out RpcFailure undoError)
                            ? SerializeSuccess(id, undoResult)
                            : SerializeError(id, undoError);

                    case "persistence/list":
                        return controller.TryListWorldSlots(out JObject slotsResult, out RpcFailure slotsError)
                            ? SerializeSuccess(id, slotsResult)
                            : SerializeError(id, slotsError);

                    case "persistence/save":
                        var save = (WorldSaveRequest)request.Parameters;
                        return controller.TrySaveWorld(
                            peer.PrincipalId,
                            save,
                            out JObject saveResult,
                            out RpcFailure saveError)
                            ? SerializeSuccess(id, saveResult)
                            : SerializeError(id, saveError);

                    case "persistence/loadPreview":
                        var loadPreview = (WorldLoadPreviewRequest)request.Parameters;
                        return controller.TryPreviewWorldLoad(
                            peer.PrincipalId,
                            peer.ConnectionEpoch,
                            loadPreview,
                            out JObject loadPreviewResult,
                            out RpcFailure loadPreviewError)
                            ? SerializeSuccess(id, loadPreviewResult)
                            : SerializeError(id, loadPreviewError);

                    case "persistence/load":
                        var load = (WorldLoadRequest)request.Parameters;
                        return controller.TryLoadWorld(
                            peer.PrincipalId,
                            peer.ConnectionEpoch,
                            load,
                            out JObject loadResult,
                            out RpcFailure loadError)
                            ? SerializeSuccess(id, loadResult)
                            : SerializeError(id, loadError);

                    case "logs/poll":
                        if (logBuffer == null)
                            return SerializeError(id, RpcFailure.Create(
                                SceneErrorCodes.MethodNotFound,
                                "Runtime log capture is not configured",
                                "logs_unavailable"));
                        var logs = (LogsPollRequest)request.Parameters;
                        return SerializeSuccess(id, logBuffer.Poll(logs));

                    default:
                        return SerializeError(id, RpcFailure.Create(
                            SceneErrorCodes.MethodNotFound,
                            $"Scene RPC method '{request.Method}' was not found",
                            "method_not_found"));
                }
            }
            catch (Exception exception)
            {
                UnityEngine.Debug.LogError("[BrainRegion] Scene RPC dispatch failed: " + exception);
                return SerializeError(id, RpcFailure.Create(
                    SceneErrorCodes.Internal,
                    "Scene RPC internal error",
                    "internal_error"));
            }
        }

        private static bool TryPrepareRequest(
            AuthenticatedPeerContext peer,
            string json,
            out PreparedRequest prepared,
            out string immediateResponse)
        {
            prepared = null;
            immediateResponse = null;
            JToken id = JValue.CreateNull();
            try
            {
                JObject request = JObject.Parse(json);
                id = request["id"]?.DeepClone() ?? JValue.CreateNull();
                if (!TryValidateEnvelope(
                    request,
                    out string method,
                    out JObject parameters,
                    out long deadline,
                    out RpcFailure envelopeError))
                {
                    immediateResponse = SerializeError(id, envelopeError);
                    return false;
                }

                object typedParameters;
                RpcFailure parseFailure;
                switch (method)
                {
                    case "runtime/info":
                        if (!TryRequireCapability(peer, SceneCapabilities.SceneRead, out RpcFailure infoAuth))
                        {
                            immediateResponse = SerializeError(id, infoAuth);
                            return false;
                        }
                        if (parameters.HasValues)
                        {
                            immediateResponse = InvalidParams(id, "runtime/info params must be empty");
                            return false;
                        }
                        typedParameters = null;
                        break;

                    case "scene/hierarchy":
                        if (!TryRequireCapability(peer, SceneCapabilities.SceneRead, out RpcFailure hierarchyAuth))
                        {
                            immediateResponse = SerializeError(id, hierarchyAuth);
                            return false;
                        }
                        if (!TryDeserialize(parameters, out SceneHierarchyRequest hierarchy, out parseFailure))
                        {
                            immediateResponse = SerializeError(id, parseFailure);
                            return false;
                        }
                        typedParameters = hierarchy;
                        break;

                    case "object/inspect":
                        if (!TryRequireCapability(peer, SceneCapabilities.SceneRead, out RpcFailure inspectAuth))
                        {
                            immediateResponse = SerializeError(id, inspectAuth);
                            return false;
                        }
                        if (!TryDeserialize(parameters, out ObjectInspectRequest inspect, out parseFailure))
                        {
                            immediateResponse = SerializeError(id, parseFailure);
                            return false;
                        }
                        typedParameters = inspect;
                        break;

                    case "prefab/list":
                        if (!TryRequireCapability(peer, SceneCapabilities.SceneRead, out RpcFailure prefabAuth))
                        {
                            immediateResponse = SerializeError(id, prefabAuth);
                            return false;
                        }
                        if (!TryDeserialize(parameters, out PrefabListRequest prefabs, out parseFailure))
                        {
                            immediateResponse = SerializeError(id, parseFailure);
                            return false;
                        }
                        typedParameters = prefabs;
                        break;

                    case "scene/preview":
                        if (!TryRequireCapability(peer, SceneCapabilities.SceneWrite, out RpcFailure previewAuth))
                        {
                            immediateResponse = SerializeError(id, previewAuth);
                            return false;
                        }
                        if (!TryDeserialize(parameters, out ScenePreviewRequest preview, out parseFailure))
                        {
                            immediateResponse = SerializeError(id, parseFailure);
                            return false;
                        }
                        if (preview.Commands != null &&
                            preview.Commands.Any(command => command != null && command.Kind == "spawn") &&
                            !TryRequireCapability(peer, SceneCapabilities.SceneSpawn, out RpcFailure spawnAuth))
                        {
                            immediateResponse = SerializeError(id, spawnAuth);
                            return false;
                        }
                        typedParameters = preview;
                        break;

                    case "scene/apply":
                        if (!TryRequireCapability(peer, SceneCapabilities.SceneWrite, out RpcFailure applyAuth))
                        {
                            immediateResponse = SerializeError(id, applyAuth);
                            return false;
                        }
                        if (!TryDeserialize(parameters, out SceneApplyRequest apply, out parseFailure))
                        {
                            immediateResponse = SerializeError(id, parseFailure);
                            return false;
                        }
                        typedParameters = apply;
                        break;

                    case "history/undo":
                        if (!TryRequireCapability(peer, SceneCapabilities.SceneUndo, out RpcFailure undoAuth))
                        {
                            immediateResponse = SerializeError(id, undoAuth);
                            return false;
                        }
                        if (!TryDeserialize(parameters, out SceneUndoRequest undo, out parseFailure))
                        {
                            immediateResponse = SerializeError(id, parseFailure);
                            return false;
                        }
                        typedParameters = undo;
                        break;

                    case "persistence/list":
                        if (!TryRequireCapability(peer, SceneCapabilities.PersistenceRead, out RpcFailure listAuth))
                        {
                            immediateResponse = SerializeError(id, listAuth);
                            return false;
                        }
                        if (parameters.HasValues)
                        {
                            immediateResponse = InvalidParams(id, "persistence/list params must be empty");
                            return false;
                        }
                        typedParameters = null;
                        break;

                    case "persistence/save":
                        if (!TryRequireCapability(peer, SceneCapabilities.PersistenceWrite, out RpcFailure saveAuth) ||
                            !TryRequireCapability(peer, SceneCapabilities.SceneRead, out saveAuth))
                        {
                            immediateResponse = SerializeError(id, saveAuth);
                            return false;
                        }
                        if (!TryDeserialize(parameters, out WorldSaveRequest save, out parseFailure))
                        {
                            immediateResponse = SerializeError(id, parseFailure);
                            return false;
                        }
                        typedParameters = save;
                        break;

                    case "persistence/loadPreview":
                        if (!TryRequireWorldLoadCapabilities(peer, out RpcFailure loadPreviewAuth))
                        {
                            immediateResponse = SerializeError(id, loadPreviewAuth);
                            return false;
                        }
                        if (!TryDeserialize(parameters, out WorldLoadPreviewRequest loadPreview, out parseFailure))
                        {
                            immediateResponse = SerializeError(id, parseFailure);
                            return false;
                        }
                        typedParameters = loadPreview;
                        break;

                    case "persistence/load":
                        if (!TryRequireCapability(peer, SceneCapabilities.PersistenceWrite, out RpcFailure loadWriteAuth) ||
                            !TryRequireWorldLoadCapabilities(peer, out loadWriteAuth))
                        {
                            immediateResponse = SerializeError(id, loadWriteAuth);
                            return false;
                        }
                        if (!TryDeserialize(parameters, out WorldLoadRequest load, out parseFailure))
                        {
                            immediateResponse = SerializeError(id, parseFailure);
                            return false;
                        }
                        typedParameters = load;
                        break;

                    case "logs/poll":
                        if (!TryRequireCapability(peer, SceneCapabilities.LogsRead, out RpcFailure logsAuth))
                        {
                            immediateResponse = SerializeError(id, logsAuth);
                            return false;
                        }
                        if (!TryDeserialize(parameters, out LogsPollRequest logs, out parseFailure))
                        {
                            immediateResponse = SerializeError(id, parseFailure);
                            return false;
                        }
                        if (logs.AfterSeq < 0 || logs.Limit < 1 || logs.Limit > SceneProtocol.MaxPageSize ||
                            (logs.Levels != null &&
                             (logs.Levels.Count > 5 || logs.Levels.Any(level =>
                                 level != "info" && level != "warning" && level != "error" &&
                                 level != "exception" && level != "assert"))))
                        {
                            immediateResponse = InvalidParams(id, "logs/poll params are outside the Scene RPC bounds");
                            return false;
                        }
                        typedParameters = logs;
                        break;

                    default:
                        immediateResponse = SerializeError(id, RpcFailure.Create(
                            SceneErrorCodes.MethodNotFound,
                            $"Scene RPC method '{method}' was not found",
                            "method_not_found"));
                        return false;
                }

                prepared = new PreparedRequest(id, method, deadline, typedParameters);
                return true;
            }
            catch (JsonReaderException exception)
            {
                immediateResponse = SerializeError(JValue.CreateNull(), RpcFailure.Create(
                    -32700,
                    "Invalid JSON: " + exception.Message,
                    "parse_error"));
                return false;
            }
        }

        private static bool TryValidateEnvelope(
            JObject request,
            out string method,
            out JObject parameters,
            out long deadline,
            out RpcFailure failure)
        {
            method = null;
            parameters = null;
            deadline = 0;
            failure = null;
            string[] allowed = { "jsonrpc", "id", "method", "deadlineUnixMs", "params" };
            JProperty unexpected = request.Properties().FirstOrDefault(property => !allowed.Contains(property.Name));
            if (unexpected != null || (string)request["jsonrpc"] != "2.0" ||
                request["id"]?.Type != JTokenType.String ||
                request["method"]?.Type != JTokenType.String ||
                request["params"]?.Type != JTokenType.Object ||
                request["deadlineUnixMs"]?.Type != JTokenType.Integer)
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidRequest,
                    "Request must contain only jsonrpc, string id, method, integer deadlineUnixMs, and object params",
                    "invalid_envelope");
                return false;
            }

            method = (string)request["method"];
            parameters = (JObject)request["params"];
            try
            {
                deadline = request["deadlineUnixMs"].Value<long>();
            }
            catch (Exception)
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidRequest,
                    "deadlineUnixMs must fit in a signed 64-bit integer",
                    "invalid_deadline");
                return false;
            }
            if (deadline < 0 || deadline > 9007199254740991L)
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidRequest,
                    "deadlineUnixMs must be within the JSON safe-integer range",
                    "invalid_deadline");
                return false;
            }
            if (string.IsNullOrWhiteSpace(method) ||
                !SceneProtocol.IsIdentifier((string)request["id"], 128))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidRequest,
                    "id and method must not be empty",
                    "empty_id_or_method");
                return false;
            }
            return true;
        }

        private static bool TryDeserialize<T>(JObject parameters, out T value, out RpcFailure failure)
        {
            try
            {
                var serializer = JsonSerializer.Create(new JsonSerializerSettings
                {
                    MissingMemberHandling = MissingMemberHandling.Error,
                });
                value = parameters.ToObject<T>(serializer);
                failure = null;
                return true;
            }
            catch (JsonException exception)
            {
                value = default;
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    "Invalid params: " + exception.Message,
                    "invalid_params");
                return false;
            }
        }

        private static bool TryRequireCapability(
            AuthenticatedPeerContext peer,
            string capability,
            out RpcFailure failure)
        {
            if (peer != null && peer.HasCapability(capability))
            {
                failure = null;
                return true;
            }

            failure = RpcFailure.Create(
                SceneErrorCodes.Forbidden,
                $"Authenticated peer is not granted '{capability}'",
                "capability_required");
            failure.Data["requiredCapability"] = capability;
            return false;
        }

        private static bool TryRequireWorldLoadCapabilities(
            AuthenticatedPeerContext peer,
            out RpcFailure failure)
        {
            return TryRequireCapability(peer, SceneCapabilities.PersistenceRead, out failure) &&
                   TryRequireCapability(peer, SceneCapabilities.SceneWrite, out failure) &&
                   TryRequireCapability(peer, SceneCapabilities.SceneSpawn, out failure);
        }

        private bool TryAcceptPeerEpoch(AuthenticatedPeerContext peer, out RpcFailure failure)
        {
            string capabilityFingerprint = CapabilityFingerprint(peer);
            lock (peerEpochGate)
            {
                if (latestPeerEpochs.TryGetValue(peer.PrincipalId, out PeerEpochState current))
                {
                    if (peer.ConnectionEpoch < current.ConnectionEpoch)
                    {
                        failure = StaleConnectionFailure();
                        return false;
                    }
                    if (peer.ConnectionEpoch == current.ConnectionEpoch &&
                        !string.Equals(capabilityFingerprint, current.CapabilityFingerprint, StringComparison.Ordinal))
                    {
                        failure = RpcFailure.Create(
                            SceneErrorCodes.Unauthenticated,
                            "Authenticated peer context changed within one connection epoch",
                            "peer_context_changed");
                        return false;
                    }
                }

                if (!latestPeerEpochs.TryGetValue(peer.PrincipalId, out current) ||
                    peer.ConnectionEpoch > current.ConnectionEpoch)
                {
                    latestPeerEpochs[peer.PrincipalId] =
                        new PeerEpochState(peer.ConnectionEpoch, capabilityFingerprint);
                }
            }

            failure = null;
            return true;
        }

        private bool IsCurrentPeerEpoch(AuthenticatedPeerContext peer)
        {
            string capabilityFingerprint = CapabilityFingerprint(peer);
            lock (peerEpochGate)
            {
                return latestPeerEpochs.TryGetValue(peer.PrincipalId, out PeerEpochState current) &&
                       current.ConnectionEpoch == peer.ConnectionEpoch &&
                       string.Equals(
                           current.CapabilityFingerprint,
                           capabilityFingerprint,
                           StringComparison.Ordinal);
            }
        }

        private static string CapabilityFingerprint(AuthenticatedPeerContext peer)
        {
            return string.Join("\n", peer.GrantedCapabilities);
        }

        private static RpcFailure StaleConnectionFailure()
        {
            return RpcFailure.Create(
                SceneErrorCodes.Unauthenticated,
                "Request belongs to a superseded authenticated connection",
                "stale_connection_epoch");
        }

        private bool TryAuthorizeApply(
            AuthenticatedPeerContext peer,
            SceneApplyRequest request,
            out bool requiresSpawn,
            out RpcFailure failure)
        {
            requiresSpawn = false;
            if (request != null &&
                previewAuthorizations.TryGetValue(request.PreviewToken, out PreviewAuthorization preview))
            {
                if (!preview.Matches(peer, request.ClientMutationId))
                {
                    failure = RpcFailure.Create(
                        SceneErrorCodes.Forbidden,
                        "Preview token is not authorized for this peer connection",
                        "preview_owner_mismatch");
                    return false;
                }
                requiresSpawn = preview.RequiresSpawn;
                if (requiresSpawn && !peer.HasCapability(SceneCapabilities.SceneSpawn))
                {
                    failure = RpcFailure.Create(
                        SceneErrorCodes.Forbidden,
                        "Applying this preview requires 'scene.spawn'",
                        "capability_required");
                    failure.Data["requiredCapability"] = SceneCapabilities.SceneSpawn;
                    return false;
                }
                failure = null;
                return true;
            }

            string mutationKey = request == null
                ? null
                : AuthorizationMutationKey(peer.PrincipalId, request.ClientMutationId);
            if (request != null &&
                appliedAuthorizations.TryGetValue(mutationKey, out AppliedAuthorization applied) &&
                applied.Matches(peer, request.PreviewToken))
            {
                requiresSpawn = applied.RequiresSpawn;
                if (requiresSpawn && !peer.HasCapability(SceneCapabilities.SceneSpawn))
                {
                    failure = RpcFailure.Create(
                        SceneErrorCodes.Forbidden,
                        "Replaying this mutation receipt requires 'scene.spawn'",
                        "capability_required");
                    failure.Data["requiredCapability"] = SceneCapabilities.SceneSpawn;
                    return false;
                }
                failure = null;
                return true;
            }

            failure = RpcFailure.Create(
                SceneErrorCodes.Forbidden,
                "Apply requires a preview issued to this authenticated connection",
                "preview_authorization_missing");
            return false;
        }

        private void CleanupAuthorizationState()
        {
            long now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            List<string> expired = previewAuthorizations
                .Where(pair => pair.Value.ExpiresAtUnixMs < now)
                .Select(pair => pair.Key)
                .ToList();
            foreach (string token in expired)
                previewAuthorizations.Remove(token);
        }

        private void RememberAppliedAuthorization(
            AuthenticatedPeerContext peer,
            SceneApplyRequest request,
            bool requiresSpawn)
        {
            string mutationKey = AuthorizationMutationKey(peer.PrincipalId, request.ClientMutationId);
            bool isNew = !appliedAuthorizations.ContainsKey(mutationKey);
            appliedAuthorizations[mutationKey] = new AppliedAuthorization(
                peer.PrincipalId,
                request.PreviewToken,
                requiresSpawn);
            if (isNew)
                appliedAuthorizationOrder.Enqueue(mutationKey);

            while (appliedAuthorizationOrder.Count > MaxAppliedAuthorizations)
            {
                string oldest = appliedAuthorizationOrder.Dequeue();
                appliedAuthorizations.Remove(oldest);
            }
        }

        private static string AuthorizationMutationKey(string principalId, string clientMutationId)
        {
            return principalId + "\u001f" + clientMutationId;
        }

        private static void CompleteSafely(QueuedRequest queued, string response)
        {
            try
            {
                queued.Completion?.Invoke(response);
            }
            catch (Exception exception)
            {
                UnityEngine.Debug.LogError(
                    "[BrainRegion] Scene RPC response callback failed: " + exception.Message);
            }
        }

        private void AssertMainThread()
        {
            if (Thread.CurrentThread.ManagedThreadId != mainThreadId)
                throw new InvalidOperationException("SceneRpcDispatcher Unity API must run on the main thread");
        }

        private static string InvalidParams(JToken id, string message)
        {
            return SerializeError(id, RpcFailure.Create(
                SceneErrorCodes.InvalidParams,
                message,
                "invalid_params"));
        }

        private static string SerializeSuccess(JToken id, object result)
        {
            JToken resultToken = result as JToken ?? JToken.FromObject(result);
            return BoundResponse(new JObject
            {
                ["jsonrpc"] = "2.0",
                ["id"] = id?.DeepClone() ?? JValue.CreateNull(),
                ["result"] = resultToken,
            }, id);
        }

        private static string SerializeError(JToken id, RpcFailure failure)
        {
            var error = new JObject
            {
                ["code"] = failure.Code,
                ["message"] = failure.Message,
            };
            if (failure.Data != null) error["data"] = failure.Data;
            return BoundResponse(new JObject
            {
                ["jsonrpc"] = "2.0",
                ["id"] = id?.DeepClone() ?? JValue.CreateNull(),
                ["error"] = error,
            }, id);
        }

        private static string BoundResponse(JObject response, JToken id)
        {
            string encoded = response.ToString(Formatting.None);
            if (Encoding.UTF8.GetByteCount(encoded) <= MaxFrameBytes) return encoded;
            return new JObject
            {
                ["jsonrpc"] = "2.0",
                ["id"] = id?.DeepClone() ?? JValue.CreateNull(),
                ["error"] = new JObject
                {
                    ["code"] = SceneErrorCodes.Internal,
                    ["message"] = $"Scene RPC response exceeds {MaxFrameBytes} bytes",
                    ["data"] = new JObject { ["reason"] = "response_too_large" },
                },
            }.ToString(Formatting.None);
        }

        private void OnSceneChanged(long revision, string clientMutationId, string summary)
        {
            RefreshRegistrationNotification();
            string notification = new JObject
            {
                ["jsonrpc"] = "2.0",
                ["method"] = "scene/changed",
                ["params"] = new JObject
                {
                    ["sceneRevision"] = revision,
                    ["clientMutationId"] = clientMutationId == null
                        ? JValue.CreateNull()
                        : new JValue(clientMutationId),
                    ["summary"] = summary,
                },
            }.ToString(Formatting.None);
            Delegate[] handlers = OutboundNotification?.GetInvocationList();
            if (handlers == null) return;
            foreach (Delegate handler in handlers)
            {
                try
                {
                    ((Action<string>)handler)(notification);
                }
                catch (Exception exception)
                {
                    UnityEngine.Debug.LogError(
                        "[BrainRegion] Scene RPC notification callback failed: " + exception.Message);
                }
            }
        }

        private sealed class QueuedRequest
        {
            public readonly PreparedRequest Request;
            public readonly AuthenticatedPeerContext Peer;
            public readonly Action<string> Completion;

            public QueuedRequest(
                PreparedRequest request,
                AuthenticatedPeerContext peer,
                Action<string> completion)
            {
                Request = request;
                Peer = peer;
                Completion = completion;
            }
        }

        private sealed class DeferredPersistenceRequest
        {
            private int completed;

            public readonly long OperationId;
            public readonly QueuedRequest Request;
            public readonly RuntimeSceneController.WorldPersistenceWork Work;

            public DeferredPersistenceRequest(
                long operationId,
                QueuedRequest request,
                RuntimeSceneController.WorldPersistenceWork work)
            {
                OperationId = operationId;
                Request = request;
                Work = work;
            }

            public void TryComplete(string response)
            {
                if (Interlocked.Exchange(ref completed, 1) == 0)
                    CompleteSafely(Request, response);
            }
        }

        private sealed class CompletedPersistenceRequest
        {
            public readonly long OperationId;
            public readonly RuntimeSceneController.WorldPersistenceWorkerResult Worker;

            public CompletedPersistenceRequest(
                long operationId,
                RuntimeSceneController.WorldPersistenceWorkerResult worker)
            {
                OperationId = operationId;
                Worker = worker;
            }
        }

        private sealed class PreparedRequest
        {
            public readonly JToken Id;
            public readonly string Method;
            public readonly long DeadlineUnixMs;
            public readonly object Parameters;

            public PreparedRequest(JToken id, string method, long deadlineUnixMs, object parameters)
            {
                Id = id;
                Method = method;
                DeadlineUnixMs = deadlineUnixMs;
                Parameters = parameters;
            }
        }

        private sealed class PeerEpochState
        {
            public readonly long ConnectionEpoch;
            public readonly string CapabilityFingerprint;

            public PeerEpochState(long connectionEpoch, string capabilityFingerprint)
            {
                ConnectionEpoch = connectionEpoch;
                CapabilityFingerprint = capabilityFingerprint;
            }
        }

        private sealed class PreviewAuthorization
        {
            public readonly string PrincipalId;
            public readonly long ConnectionEpoch;
            public readonly string ClientMutationId;
            public readonly long ExpiresAtUnixMs;
            public readonly bool RequiresSpawn;

            public PreviewAuthorization(
                AuthenticatedPeerContext peer,
                string clientMutationId,
                long expiresAtUnixMs,
                bool requiresSpawn)
            {
                PrincipalId = peer.PrincipalId;
                ConnectionEpoch = peer.ConnectionEpoch;
                ClientMutationId = clientMutationId;
                ExpiresAtUnixMs = expiresAtUnixMs;
                RequiresSpawn = requiresSpawn;
            }

            public bool Matches(AuthenticatedPeerContext peer, string clientMutationId)
            {
                return peer != null &&
                       string.Equals(PrincipalId, peer.PrincipalId, StringComparison.Ordinal) &&
                       ConnectionEpoch == peer.ConnectionEpoch &&
                       string.Equals(ClientMutationId, clientMutationId, StringComparison.Ordinal);
            }
        }

        private sealed class AppliedAuthorization
        {
            public readonly string PrincipalId;
            public readonly string PreviewToken;
            public readonly bool RequiresSpawn;

            public AppliedAuthorization(string principalId, string previewToken, bool requiresSpawn)
            {
                PrincipalId = principalId;
                PreviewToken = previewToken;
                RequiresSpawn = requiresSpawn;
            }

            public bool Matches(AuthenticatedPeerContext peer, string previewToken)
            {
                return peer != null &&
                       string.Equals(PrincipalId, peer.PrincipalId, StringComparison.Ordinal) &&
                       string.Equals(PreviewToken, previewToken, StringComparison.Ordinal);
            }
        }
    }
}
