using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace BrainRegion.RuntimeBridge
{
    /// <summary>
    /// Owns the editable runtime world. All public methods must be called from the
    /// Unity main thread; SceneRpcDispatcher provides the bounded cross-thread queue.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed partial class RuntimeSceneController : MonoBehaviour
    {
        [SerializeField] private Transform sandboxRoot;
        [SerializeField] private RuntimePrefabCatalog prefabCatalog;
        [SerializeField] private bool includeLoadedScenes;
        [SerializeField, Min(1)] private int maxUndoEntries = 128;
        [SerializeField, Min(1f)] private float previewLifetimeSeconds = 30f;

        private readonly Dictionary<string, RpcObjectIdentity> objects =
            new Dictionary<string, RpcObjectIdentity>(StringComparer.Ordinal);
        private int mainThreadId;
        private string initializationError;
        private string instanceId;
        private string sessionId;
        private long sceneRevision;
        private bool lifecycleInitialized;
        private bool lifecycleReconcileRequested;
        private bool lifecycleChangePending;
        private string lastLifecycleWarning;

        public long SceneRevision => sceneRevision;
        public string InstanceId => instanceId;
        public string SessionId => sessionId;
        public string InitializationError => initializationError;
        public bool IsReady => string.IsNullOrEmpty(initializationError);
        public Transform SandboxRoot => sandboxRoot != null ? sandboxRoot : transform;
        public RuntimePrefabCatalog PrefabCatalog => prefabCatalog;

        public event Action<long, string, string> SceneChanged;

        private void Awake()
        {
            mainThreadId = System.Threading.Thread.CurrentThread.ManagedThreadId;
            instanceId = Guid.NewGuid().ToString("N");
            sessionId = Guid.NewGuid().ToString("N");

            if (sandboxRoot == null)
                sandboxRoot = transform;

            if (prefabCatalog != null && !prefabCatalog.TryValidate(out string catalogError))
                initializationError = catalogError;

            if (!TryIndexExistingObjects(out string registryError))
                initializationError = AppendError(initializationError, registryError);

            lifecycleInitialized = true;
            lifecycleReconcileRequested = true;
        }

        private void OnEnable()
        {
            RuntimeIdentityLifecycle.Changed += HandleIdentityLifecycleChanged;
            SceneManager.sceneLoaded += HandleSceneLoaded;
            SceneManager.sceneUnloaded += HandleSceneUnloaded;
            lifecycleReconcileRequested = true;
        }

        private void OnDisable()
        {
            RuntimeIdentityLifecycle.Changed -= HandleIdentityLifecycleChanged;
            SceneManager.sceneLoaded -= HandleSceneLoaded;
            SceneManager.sceneUnloaded -= HandleSceneUnloaded;
        }

        private void OnDestroy()
        {
            foreach (RpcObjectIdentity identity in objects.Values.ToArray())
            {
                if (identity != null)
                    identity.Release(this);
            }
            objects.Clear();
        }

        private void LateUpdate()
        {
            if (!lifecycleInitialized || !lifecycleReconcileRequested) return;
            lifecycleReconcileRequested = false;
            ReconcileRuntimeIdentities();
        }

        public JObject GetRuntimeInfo()
        {
            AssertMainThread();
            Scene activeScene = SceneManager.GetActiveScene();
            return new JObject
            {
                ["protocolVersion"] = SceneProtocol.Version,
                ["instanceId"] = instanceId,
                ["sessionId"] = sessionId,
                ["buildId"] = string.IsNullOrEmpty(Application.buildGUID) ? Application.version : Application.buildGUID,
                ["unityVersion"] = Application.unityVersion,
                ["platform"] = Application.platform.ToString(),
                ["product"] = Application.productName,
                ["sceneId"] = string.IsNullOrEmpty(activeScene.path) ? activeScene.name : activeScene.path,
                ["sceneRevision"] = sceneRevision,
                ["status"] = IsReady ? "ready" : "degraded",
                ["error"] = initializationError,
                ["capabilities"] = new JArray("scene.read", "scene.write", "scene.spawn", "scene.undo"),
            };
        }

        public bool TryGetHierarchy(
            SceneHierarchyRequest request,
            out SceneHierarchyResult result,
            out RpcFailure failure)
        {
            AssertMainThread();
            result = null;
            failure = null;
            if (!TryRequireReady(out failure)) return false;

            request = request ?? new SceneHierarchyRequest();
            if (request.Depth < 0 || request.Depth > 8 ||
                request.Limit < 1 || request.Limit > SceneProtocol.MaxPageSize ||
                (request.IfRevision.HasValue && request.IfRevision.Value < 0) ||
                (!string.IsNullOrEmpty(request.RootId) && !SceneProtocol.IsIdentifier(request.RootId, 160)) ||
                (!string.IsNullOrEmpty(request.Cursor) && request.Cursor.Length > 256))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    "Hierarchy depth, limit, rootId, or cursor is outside the Scene RPC bounds",
                    "invalid_hierarchy_params");
                return false;
            }
            int depth = request.Depth;
            int limit = request.Limit;

            if (request.IfRevision.HasValue && request.IfRevision.Value == sceneRevision)
            {
                result = new SceneHierarchyResult
                {
                    SceneRevision = sceneRevision,
                    NotModified = true,
                };
                return true;
            }

            RpcObjectIdentity requestedRoot = null;
            if (!string.IsNullOrWhiteSpace(request.RootId) &&
                !TryResolveObject(request.RootId, out requestedRoot, out failure))
            {
                return false;
            }

            int offset = 0;
            string hierarchyFingerprint = ComputeStableDigest(
                sessionId + "\nhierarchy\n" + (request.RootId ?? string.Empty) + "\n" +
                depth.ToString(CultureInfo.InvariantCulture) + "\n" +
                (request.IncludeInactive ? "1" : "0")).Substring(0, 32);
            if (!string.IsNullOrWhiteSpace(request.Cursor) &&
                !TryParseCursor(request.Cursor, sceneRevision, hierarchyFingerprint, out offset))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.RevisionConflict,
                    "Hierarchy cursor belongs to another revision or query",
                    "stale_cursor",
                    true);
                failure.Data["actualRevision"] = sceneRevision;
                return false;
            }

            List<RpcObjectIdentity> candidates = objects.Values
                .Where(IsAlive)
                .Where(identity => request.IncludeInactive || identity.gameObject.activeInHierarchy)
                .Where(identity => requestedRoot == null
                    ? GetRegisteredDepth(identity) <= depth
                    : IsWithinDepth(identity.transform, requestedRoot.transform, depth))
                .OrderBy(identity => BuildHierarchySortKey(identity.transform), StringComparer.Ordinal)
                .ThenBy(identity => identity.StableId, StringComparer.Ordinal)
                .ToList();

            var parentIds = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (RpcObjectIdentity identity in objects.Values.Where(IsAlive))
                parentIds[identity.StableId] = FindRegisteredParentId(identity.transform);

            var directChildCounts = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (string parentId in parentIds.Values)
            {
                if (string.IsNullOrEmpty(parentId)) continue;
                directChildCounts.TryGetValue(parentId, out int count);
                directChildCounts[parentId] = count + 1;
            }

            List<RpcObjectIdentity> page = candidates.Skip(offset).Take(limit).ToList();
            result = new SceneHierarchyResult
            {
                SceneRevision = sceneRevision,
                NextCursor = offset + page.Count < candidates.Count
                    ? FormatCursor(sceneRevision, offset + page.Count, hierarchyFingerprint)
                    : null,
            };
            foreach (RpcObjectIdentity identity in page)
            {
                directChildCounts.TryGetValue(identity.StableId, out int childCount);
                result.Nodes.Add(BuildObjectSnapshot(identity, false, null, childCount));
            }
            return true;
        }

        public bool TryInspect(
            ObjectInspectRequest request,
            out RpcObjectSnapshot result,
            out RpcFailure failure)
        {
            AssertMainThread();
            result = null;
            failure = null;
            if (!TryRequireReady(out failure)) return false;
            if (request == null || string.IsNullOrWhiteSpace(request.ObjectId))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    "object/inspect requires objectId",
                    "missing_object_id");
                return false;
            }
            if (!SceneProtocol.IsIdentifier(request.ObjectId, 160) ||
                (request.ComponentIds != null &&
                 (request.ComponentIds.Count > 64 ||
                  request.ComponentIds.Any(componentId => !SceneProtocol.IsIdentifier(componentId, 200)))))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    "objectId or componentIds are outside the Scene RPC bounds",
                    "invalid_inspect_params");
                return false;
            }
            if (!TryResolveObject(request.ObjectId, out RpcObjectIdentity identity, out failure))
                return false;

            HashSet<string> filter = request.ComponentIds == null
                ? null
                : new HashSet<string>(request.ComponentIds, StringComparer.Ordinal);
            result = BuildObjectSnapshot(identity, true, filter, CountDirectRegisteredChildren(identity));
            return true;
        }

        public bool TryListPrefabs(PrefabListRequest request, out JObject result, out RpcFailure failure)
        {
            AssertMainThread();
            result = null;
            failure = null;
            if (!TryRequireReady(out failure)) return false;
            if (prefabCatalog == null)
            {
                result = new JObject
                {
                    ["schemaVersion"] = "none",
                    ["entries"] = new JArray(),
                    ["nextCursor"] = null,
                };
                return true;
            }

            request = request ?? new PrefabListRequest();
            if (request.Limit < 1 || request.Limit > SceneProtocol.MaxPageSize ||
                (request.Query != null && request.Query.Length > 256) ||
                (request.Cursor != null && request.Cursor.Length > 256))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    "Prefab query, cursor, or limit is outside the Scene RPC bounds",
                    "invalid_prefab_params");
                return false;
            }
            int limit = request.Limit;
            int offset = 0;
            string query = request.Query?.Trim();
            string prefabFingerprint = ComputeStableDigest(
                sessionId + "\nprefabs\n" + (prefabCatalog.SchemaVersion ?? string.Empty) + "\n" +
                (query ?? string.Empty)).Substring(0, 32);
            if (!string.IsNullOrWhiteSpace(request.Cursor) &&
                !TryParsePrefabCursor(request.Cursor, prefabFingerprint, out offset))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    "Prefab cursor belongs to another catalog or query",
                    "stale_cursor",
                    true);
                return false;
            }

            List<RuntimePrefabEntry> entries = prefabCatalog.Entries
                .Where(entry => entry != null && entry.Prefab != null)
                .Where(entry => string.IsNullOrEmpty(query) || MatchesPrefabQuery(entry, query))
                .OrderBy(entry => entry.PrefabId, StringComparer.Ordinal)
                .ToList();
            List<RuntimePrefabEntry> page = entries.Skip(Mathf.Max(0, offset)).Take(limit).ToList();

            var serialized = new JArray();
            foreach (RuntimePrefabEntry entry in page)
            {
                serialized.Add(new JObject
                {
                    ["prefabId"] = entry.PrefabId,
                    ["displayName"] = entry.DisplayName,
                    ["tags"] = new JArray(entry.Tags),
                });
            }
            result = new JObject
            {
                ["schemaVersion"] = prefabCatalog.SchemaVersion,
                ["entries"] = serialized,
                ["nextCursor"] = offset + page.Count < entries.Count
                    ? FormatPrefabCursor(offset + page.Count, prefabFingerprint)
                    : null,
            };
            return true;
        }

        /// <summary>
        /// Project systems call this when they intentionally change persistent,
        /// RPC-visible state outside Scene RPC. Do not call it for transient physics
        /// or animation samples.
        /// </summary>
        public void NotifyExternalPersistentMutation(string summary)
        {
            AssertMainThread();
            // An application-owned mutation is outside the reversible Scene RPC
            // command log. It invalidates every outstanding preview and forms a
            // linear-history barrier so a later RPC undo cannot overwrite newer game
            // state with a stale inverse snapshot.
            EstablishExternalMutationBarrier();
            AdvanceRevision(null, string.IsNullOrWhiteSpace(summary) ? "external mutation" : summary);
        }

        /// <summary>
        /// Register an application-created editable object. Scene RPC spawns call the
        /// same invariant checks internally. Registration itself does not advance the
        /// revision; the creating system should call NotifyExternalPersistentMutation.
        /// </summary>
        public bool TryRegister(RpcObjectIdentity identity, out string error)
        {
            AssertMainThread();
            if (!IsIdentityInScope(identity))
            {
                error = includeLoadedScenes
                    ? "Editable object must belong to a loaded scene"
                    : "Editable object must be inside the configured sandbox root";
                return false;
            }
            identity.EnsureIdentity();
            return RegisterSpawned(identity, out error);
        }

        internal bool TryResolveObject(
            string objectId,
            out RpcObjectIdentity identity,
            out RpcFailure failure)
        {
            if (!string.IsNullOrWhiteSpace(objectId) &&
                SceneProtocol.IsIdentifier(objectId, 160) &&
                objects.TryGetValue(objectId, out identity) && IsAlive(identity))
            {
                failure = null;
                return true;
            }

            identity = null;
            failure = RpcFailure.Create(
                SceneErrorCodes.ObjectNotFound,
                $"Runtime object '{objectId}' was not found",
                "object_not_found");
            failure.Data["objectId"] = objectId;
            return false;
        }

        internal bool TryResolveAdapter(
            RpcObjectIdentity identity,
            string componentId,
            out IRpcPropertyAdapter adapter,
            out RpcFailure failure)
        {
            adapter = null;
            failure = null;
            if (identity == null || string.IsNullOrWhiteSpace(componentId))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    "componentId is required",
                    "missing_component_id");
                return false;
            }
            if (!SceneProtocol.IsIdentifier(componentId, 200))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    "componentId is outside the Scene RPC bounds",
                    "invalid_component_id");
                return false;
            }

            foreach (MonoBehaviour behaviour in identity.GetComponents<MonoBehaviour>())
            {
                if (!(behaviour is IRpcPropertyAdapter candidate)) continue;
                string candidateId = MakeComponentId(identity.StableId, candidate.ComponentKey);
                if (string.Equals(candidateId, componentId, StringComparison.Ordinal))
                {
                    adapter = candidate;
                    return true;
                }
            }

            failure = RpcFailure.Create(
                SceneErrorCodes.PropertyNotExposed,
                $"Component '{componentId}' is not exposed on '{identity.StableId}'",
                "component_not_exposed");
            failure.Data["objectId"] = identity.StableId;
            failure.Data["componentId"] = componentId;
            return false;
        }

        internal bool RegisterSpawned(RpcObjectIdentity identity, out string error)
        {
            if (identity == null)
            {
                error = "Spawned prefab is missing RpcObjectIdentity";
                return false;
            }
            if (objects.ContainsKey(identity.StableId))
            {
                error = $"Duplicate runtime object id '{identity.StableId}'";
                return false;
            }
            if (!SceneProtocol.IsIdentifier(identity.StableId, 160))
            {
                error = $"Runtime object id '{identity.StableId}' is not a valid wire identifier";
                return false;
            }
            if (!TryValidateAdapters(identity, out error)) return false;
            if (!identity.TryClaim(this, out error)) return false;
            objects.Add(identity.StableId, identity);
            error = null;
            return true;
        }

        internal void Unregister(RpcObjectIdentity identity)
        {
            if (identity == null || string.IsNullOrEmpty(identity.StableId)) return;
            if (objects.TryGetValue(identity.StableId, out RpcObjectIdentity current) && current == identity)
            {
                objects.Remove(identity.StableId);
                identity.Release(this);
            }
        }

        private bool TryIndexExistingObjects(out string error)
        {
            foreach (RpcObjectIdentity existing in objects.Values.ToArray())
            {
                if (existing != null)
                    existing.Release(this);
            }
            objects.Clear();
            foreach (RpcObjectIdentity identity in EnumerateScopedIdentities())
            {
                identity.EnsureIdentity();
                if (!RegisterSpawned(identity, out error)) return false;
            }
            error = null;
            return true;
        }

        internal void NotifyIdentityDestroyed(RpcObjectIdentity identity, string objectId)
        {
            if (string.IsNullOrEmpty(objectId)) return;
            if (objects.TryGetValue(objectId, out RpcObjectIdentity current) &&
                ReferenceEquals(current, identity))
            {
                objects.Remove(objectId);
                identity.Release(this);
                lifecycleChangePending = true;
                lifecycleReconcileRequested = true;
            }
        }

        private void HandleIdentityLifecycleChanged(RpcObjectIdentity identity)
        {
            lifecycleReconcileRequested = true;
        }

        private void HandleSceneLoaded(Scene scene, LoadSceneMode mode)
        {
            lifecycleReconcileRequested = true;
        }

        private void HandleSceneUnloaded(Scene scene)
        {
            lifecycleReconcileRequested = true;
        }

        private void ReconcileRuntimeIdentities()
        {
            bool changed = lifecycleChangePending;
            lifecycleChangePending = false;
            List<RpcObjectIdentity> scoped = EnumerateScopedIdentities();
            var scopedSet = new HashSet<RpcObjectIdentity>(scoped);

            foreach (KeyValuePair<string, RpcObjectIdentity> pair in objects.ToArray())
            {
                RpcObjectIdentity identity = pair.Value;
                if (IsAlive(identity) && scopedSet.Contains(identity)) continue;
                objects.Remove(pair.Key);
                if (identity != null)
                    identity.Release(this);
                changed = true;
            }

            foreach (RpcObjectIdentity identity in scoped)
            {
                identity.EnsureIdentity();
                if (objects.TryGetValue(identity.StableId, out RpcObjectIdentity registered))
                {
                    if (registered == identity) continue;
                    WarnLifecycleOnce(
                        $"Duplicate runtime object id '{identity.StableId}' was ignored during lifecycle reconciliation");
                    continue;
                }
                if (!RegisterSpawned(identity, out string error))
                {
                    WarnLifecycleOnce(error);
                    continue;
                }
                changed = true;
            }

            if (!changed) return;
            lastLifecycleWarning = null;
            NotifyExternalPersistentMutation("runtime object lifecycle changed");
        }

        private List<RpcObjectIdentity> EnumerateScopedIdentities()
        {
            var result = new List<RpcObjectIdentity>();
            if (!includeLoadedScenes)
            {
                result.AddRange(SandboxRoot.GetComponentsInChildren<RpcObjectIdentity>(true));
            }
            else
            {
                // Awake can run before Unity reports the startup scene as loaded.
                // The configured sandbox is nevertheless the authoritative baseline
                // and must be indexed without manufacturing an external revision.
                result.AddRange(SandboxRoot.GetComponentsInChildren<RpcObjectIdentity>(true));
                for (int sceneIndex = 0; sceneIndex < SceneManager.sceneCount; sceneIndex++)
                {
                    Scene scene = SceneManager.GetSceneAt(sceneIndex);
                    if (!scene.IsValid() || !scene.isLoaded) continue;
                    foreach (GameObject root in scene.GetRootGameObjects())
                        result.AddRange(root.GetComponentsInChildren<RpcObjectIdentity>(true));
                }
            }
            return result
                .Where(IsAlive)
                .Distinct()
                .OrderBy(identity => identity.gameObject.scene.handle)
                .ThenBy(identity => BuildHierarchySortKey(identity.transform), StringComparer.Ordinal)
                .ToList();
        }

        private bool IsIdentityInScope(RpcObjectIdentity identity)
        {
            if (!IsAlive(identity)) return false;
            if (includeLoadedScenes)
            {
                Scene scene = identity.gameObject.scene;
                return scene.IsValid() && scene.isLoaded;
            }
            return identity.transform == SandboxRoot || identity.transform.IsChildOf(SandboxRoot);
        }

        private void WarnLifecycleOnce(string warning)
        {
            if (string.IsNullOrEmpty(warning) ||
                string.Equals(lastLifecycleWarning, warning, StringComparison.Ordinal)) return;
            lastLifecycleWarning = warning;
            Debug.LogWarning("[BrainRegion] " + warning);
        }

        private RpcObjectSnapshot BuildObjectSnapshot(
            RpcObjectIdentity identity,
            bool includeProperties,
            HashSet<string> componentFilter,
            int childCount)
        {
            var snapshot = new RpcObjectSnapshot
            {
                ObjectId = identity.StableId,
                ParentId = FindRegisteredParentId(identity.transform),
                Name = identity.gameObject.name,
                Active = identity.gameObject.activeSelf,
                Layer = identity.gameObject.layer,
                Tag = identity.gameObject.tag,
                PrefabId = string.IsNullOrWhiteSpace(identity.PrefabId) ? null : identity.PrefabId,
                ChildCount = childCount,
            };

            string transformId = MakeComponentId(identity.StableId, "transform");
            if (componentFilter == null || componentFilter.Contains(transformId))
                snapshot.Components.Add(BuildTransformSnapshot(identity.transform, transformId, includeProperties));

            var componentKeys = new HashSet<string>(StringComparer.Ordinal);
            foreach (MonoBehaviour behaviour in identity.GetComponents<MonoBehaviour>())
            {
                if (!(behaviour is IRpcPropertyAdapter adapter)) continue;
                string key = adapter.ComponentKey?.Trim();
                if (string.IsNullOrEmpty(key) || !componentKeys.Add(key)) continue;
                string componentId = MakeComponentId(identity.StableId, key);
                if (componentFilter != null && !componentFilter.Contains(componentId)) continue;

                var component = new RpcComponentSnapshot
                {
                    ComponentId = componentId,
                    TypeId = adapter.TypeId,
                };
                if (includeProperties)
                {
                    IReadOnlyList<RpcPropertyDescriptor> descriptors = adapter.DescribeProperties();
                    if (descriptors != null)
                    {
                        foreach (RpcPropertyDescriptor descriptor in descriptors)
                        {
                            if (descriptor == null || string.IsNullOrWhiteSpace(descriptor.PropertyId)) continue;
                            adapter.TryRead(descriptor.PropertyId, out JToken value, out string readError);
                            component.Properties.Add(new RpcPropertySnapshot
                            {
                                Descriptor = descriptor,
                                Value = readError == null ? value : JValue.CreateNull(),
                            });
                        }
                    }
                }
                snapshot.Components.Add(component);
            }
            return snapshot;
        }

        private static RpcComponentSnapshot BuildTransformSnapshot(
            Transform target,
            string componentId,
            bool includeProperties)
        {
            var snapshot = new RpcComponentSnapshot
            {
                ComponentId = componentId,
                TypeId = "unity.transform",
            };
            if (!includeProperties) return snapshot;

            snapshot.Properties.Add(VectorProperty("position", "Local Position", target.localPosition));
            snapshot.Properties.Add(VectorProperty("rotationEuler", "Local Rotation", target.localEulerAngles));
            snapshot.Properties.Add(VectorProperty("scale", "Local Scale", target.localScale));
            return snapshot;
        }

        private static RpcPropertySnapshot VectorProperty(string id, string displayName, Vector3 value)
        {
            return new RpcPropertySnapshot
            {
                Descriptor = new RpcPropertyDescriptor
                {
                    PropertyId = id,
                    DisplayName = displayName,
                    ValueType = "vec3",
                    Persistent = true,
                },
                Value = JObject.FromObject(new RpcVector3 { X = value.x, Y = value.y, Z = value.z }),
            };
        }

        private int CountDirectRegisteredChildren(RpcObjectIdentity parent)
        {
            int count = 0;
            foreach (RpcObjectIdentity identity in objects.Values)
                if (IsAlive(identity) && FindRegisteredParentId(identity.transform) == parent.StableId)
                    count++;
            return count;
        }

        private string FindRegisteredParentId(Transform target)
        {
            Transform parent = target.parent;
            while (parent != null)
            {
                RpcObjectIdentity identity = parent.GetComponent<RpcObjectIdentity>();
                if (identity != null && objects.ContainsKey(identity.StableId))
                    return identity.StableId;
                if (parent == SandboxRoot) break;
                parent = parent.parent;
            }
            return null;
        }

        private bool IsWithinDepth(Transform candidate, Transform root, int maxDepth)
        {
            if (candidate == root) return true;
            int depth = 1;
            Transform current = candidate.parent;
            while (current != null)
            {
                if (current == root) return depth <= maxDepth;
                RpcObjectIdentity identity = current.GetComponent<RpcObjectIdentity>();
                if (identity != null && objects.ContainsKey(identity.StableId)) depth++;
                current = current.parent;
            }
            return false;
        }

        private int GetRegisteredDepth(RpcObjectIdentity identity)
        {
            if (identity.transform == SandboxRoot) return 0;
            int depth = 0;
            Transform parent = identity.transform.parent;
            while (parent != null)
            {
                RpcObjectIdentity parentIdentity = parent.GetComponent<RpcObjectIdentity>();
                if (parentIdentity != null && objects.ContainsKey(parentIdentity.StableId)) depth++;
                if (parent == SandboxRoot) break;
                parent = parent.parent;
            }
            return depth;
        }

        private static string BuildHierarchySortKey(Transform target)
        {
            var segments = new Stack<string>();
            Transform current = target;
            while (current != null)
            {
                segments.Push(current.GetSiblingIndex().ToString("D6", CultureInfo.InvariantCulture));
                current = current.parent;
            }
            return string.Join("/", segments);
        }

        private static bool MatchesPrefabQuery(RuntimePrefabEntry entry, string query)
        {
            if (entry.PrefabId.IndexOf(query, StringComparison.OrdinalIgnoreCase) >= 0) return true;
            if (!string.IsNullOrEmpty(entry.DisplayName) &&
                entry.DisplayName.IndexOf(query, StringComparison.OrdinalIgnoreCase) >= 0) return true;
            return entry.Tags.Any(tag => tag != null && tag.IndexOf(query, StringComparison.OrdinalIgnoreCase) >= 0);
        }

        private static bool TryParseCursor(
            string cursor,
            long revision,
            string queryFingerprint,
            out int offset)
        {
            offset = 0;
            string[] pieces = cursor.Split(':');
            return pieces.Length == 3 &&
                   long.TryParse(pieces[0], NumberStyles.None, CultureInfo.InvariantCulture, out long parsedRevision) &&
                   parsedRevision == revision &&
                   int.TryParse(pieces[1], NumberStyles.None, CultureInfo.InvariantCulture, out offset) &&
                   offset >= 0 &&
                   string.Equals(pieces[2], queryFingerprint, StringComparison.Ordinal);
        }

        private static string FormatCursor(long revision, int offset, string queryFingerprint)
        {
            return revision.ToString(CultureInfo.InvariantCulture) + ":" +
                   offset.ToString(CultureInfo.InvariantCulture) + ":" + queryFingerprint;
        }

        private static bool TryParsePrefabCursor(
            string cursor,
            string queryFingerprint,
            out int offset)
        {
            offset = 0;
            string[] pieces = cursor.Split(':');
            return pieces.Length == 2 &&
                   int.TryParse(pieces[0], NumberStyles.None, CultureInfo.InvariantCulture, out offset) &&
                   offset >= 0 &&
                   string.Equals(pieces[1], queryFingerprint, StringComparison.Ordinal);
        }

        private static string FormatPrefabCursor(int offset, string queryFingerprint)
        {
            return offset.ToString(CultureInfo.InvariantCulture) + ":" + queryFingerprint;
        }

        private static string MakeComponentId(string objectId, string componentKey)
        {
            return objectId + "/" + componentKey;
        }

        private static bool IsAlive(RpcObjectIdentity identity)
        {
            return identity != null && identity.gameObject != null;
        }

        private bool TryRequireReady(out RpcFailure failure)
        {
            if (IsReady)
            {
                failure = null;
                return true;
            }
            failure = RpcFailure.Create(
                SceneErrorCodes.Internal,
                "Runtime scene controller is degraded",
                initializationError ?? "initialization_failed");
            return false;
        }

        private void AssertMainThread()
        {
            if (System.Threading.Thread.CurrentThread.ManagedThreadId != mainThreadId)
                throw new InvalidOperationException("RuntimeSceneController must be called on the Unity main thread");
        }

        private void AdvanceRevision(string clientMutationId, string summary)
        {
            if (sceneRevision == 9007199254740991L)
                throw new InvalidOperationException("Scene revision exhausted the JSON safe-integer range");
            sceneRevision++;
            Delegate[] handlers = SceneChanged?.GetInvocationList();
            if (handlers == null) return;
            foreach (Delegate handler in handlers)
            {
                try
                {
                    ((Action<long, string, string>)handler)(sceneRevision, clientMutationId, summary);
                }
                catch (Exception exception)
                {
                    UnityEngine.Debug.LogError(
                        "[BrainRegion] Scene change notification subscriber failed: " + exception.Message);
                }
            }
        }

        private static bool TryValidateAdapters(RpcObjectIdentity identity, out string error)
        {
            var keys = new HashSet<string>(StringComparer.Ordinal);
            try
            {
                foreach (MonoBehaviour behaviour in identity.GetComponents<MonoBehaviour>())
                {
                    if (!(behaviour is IRpcPropertyAdapter adapter)) continue;
                    if (!SceneProtocol.IsIdentifierSegment(adapter.ComponentKey, 128))
                    {
                        error = $"Object '{identity.StableId}' has an adapter with an empty component key";
                        return false;
                    }
                    if (!keys.Add(adapter.ComponentKey))
                    {
                        error = $"Object '{identity.StableId}' has duplicate adapter key '{adapter.ComponentKey}'";
                        return false;
                    }
                    if (!SceneProtocol.IsIdentifier(adapter.TypeId, 128))
                    {
                        error = $"Adapter '{adapter.ComponentKey}' on '{identity.StableId}' has an empty type id";
                        return false;
                    }

                    IReadOnlyList<RpcPropertyDescriptor> descriptors = adapter.DescribeProperties();
                    if (descriptors == null) continue;
                    var propertyIds = new HashSet<string>(StringComparer.Ordinal);
                    foreach (RpcPropertyDescriptor descriptor in descriptors)
                    {
                        if (descriptor == null || !SceneProtocol.IsIdentifierSegment(descriptor.PropertyId, 128) ||
                            !propertyIds.Add(descriptor.PropertyId))
                        {
                            error = $"Adapter '{adapter.ComponentKey}' on '{identity.StableId}' has an empty or duplicate property id";
                            return false;
                        }
                    }
                }
            }
            catch (Exception exception)
            {
                error = $"Adapter validation failed on '{identity.StableId}': {exception.Message}";
                return false;
            }
            error = null;
            return true;
        }

        private static string AppendError(string existing, string next)
        {
            if (string.IsNullOrEmpty(existing)) return next;
            if (string.IsNullOrEmpty(next)) return existing;
            return existing + "; " + next;
        }
    }
}
