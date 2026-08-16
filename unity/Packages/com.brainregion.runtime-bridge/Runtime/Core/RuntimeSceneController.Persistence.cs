using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace BrainRegion.RuntimeBridge
{
    public sealed partial class RuntimeSceneController
    {
        private const string WorldFormatVersion = "brainregion.world.v1";
        private const int MaxWorldSlots = 32;
        private const int MaxWorldEntities = 256;
        private const int MaxWorldProperties = 2048;
        private const int MaxWorldFileBytes = 1024 * 1024;
        private const long MaxWorldStorageBytes = 16L * 1024L * 1024L;
        private const int MaxWorldLoadPreviews = 16;
        private const int MaxWorldReceipts = 128;

        private readonly Dictionary<string, WorldLoadProposal> worldLoadProposals =
            new Dictionary<string, WorldLoadProposal>(StringComparer.Ordinal);
        private readonly Dictionary<string, WorldSaveReceipt> worldSaveReceipts =
            new Dictionary<string, WorldSaveReceipt>(StringComparer.Ordinal);
        private readonly Dictionary<string, WorldLoadReceipt> worldLoadReceipts =
            new Dictionary<string, WorldLoadReceipt>(StringComparer.Ordinal);

        public bool TryListWorldSlots(out JObject result, out RpcFailure failure)
        {
            AssertMainThread();
            result = null;
            failure = null;
            try
            {
                string root = GetWorldStorageRoot();
                var slots = new JArray();
                var corruptSlots = new JArray();
                if (Directory.Exists(root))
                {
                    foreach (string path in Directory.GetFiles(root, "*.brworld.json")
                                 .OrderBy(candidate => candidate, StringComparer.Ordinal))
                    {
                        string slot = Path.GetFileName(path).Substring(
                            0,
                            Path.GetFileName(path).Length - ".brworld.json".Length);
                        if (!TryReadWorldEnvelope(slot, out JObject envelope, out RpcFailure readFailure))
                        {
                            corruptSlots.Add(new JObject
                            {
                                ["slot"] = slot,
                                ["reason"] = (string)readFailure.Data?["reason"] ?? "invalid_world_document",
                            });
                            continue;
                        }
                        JObject document = (JObject)envelope["document"];
                        slots.Add(new JObject
                        {
                            ["slot"] = slot,
                            ["digest"] = envelope["digest"].DeepClone(),
                            ["savedRevision"] = document["savedRevision"].DeepClone(),
                            ["savedUnixMs"] = document["savedUnixMs"].DeepClone(),
                            ["label"] = document["metadata"]?["label"]?.DeepClone() ?? JValue.CreateNull(),
                            ["bytes"] = new FileInfo(path).Length,
                        });
                    }
                }
                result = new JObject
                {
                    ["slots"] = slots,
                    ["corruptSlots"] = corruptSlots,
                    ["maximumSlots"] = MaxWorldSlots,
                    ["maximumFileBytes"] = MaxWorldFileBytes,
                    ["maximumStorageBytes"] = MaxWorldStorageBytes,
                };
                return true;
            }
            catch (Exception exception)
            {
                failure = PersistenceFailure(
                    "Could not enumerate WorldDocument slots: " + exception.Message,
                    "slot_list_failed",
                    true);
                return false;
            }
        }

        public bool TrySaveWorld(
            string principalId,
            WorldSaveRequest request,
            out JObject result,
            out RpcFailure failure)
        {
            AssertMainThread();
            result = null;
            failure = null;
            if (!TryValidateWorldMutationHeader(
                    principalId,
                    request?.Slot,
                    request?.ClientMutationId,
                    request?.ExpectedRevision ?? -1,
                    out failure) ||
                !TryValidateSaveMetadata(request.Metadata, out failure) ||
                (!string.IsNullOrEmpty(request.ExpectedSlotDigest) &&
                 !IsWorldDigest(request.ExpectedSlotDigest)))
            {
                if (failure == null)
                    failure = PersistenceFailure(
                        "expectedSlotDigest must contain a lowercase SHA-256 digest",
                        "invalid_slot_digest");
                return false;
            }

            string receiptKey = WorldMutationKey(principalId, request.ClientMutationId);
            if (worldSaveReceipts.TryGetValue(receiptKey, out WorldSaveReceipt prior))
            {
                if (!prior.Matches(request))
                {
                    failure = PersistenceFailure(
                        "clientMutationId is already bound to another save request",
                        "save_mutation_reused");
                    return false;
                }
                result = (JObject)prior.Result.DeepClone();
                result["idempotentReplay"] = true;
                return true;
            }
            if (worldSaveReceipts.Count >= MaxWorldReceipts)
            {
                failure = PersistenceFailure(
                    "World save receipt capacity is exhausted for this Player session",
                    "save_receipt_capacity",
                    true);
                return false;
            }
            if (request.ExpectedRevision != sceneRevision)
            {
                failure = RevisionFailure(request.ExpectedRevision);
                return false;
            }
            JObject document;
            try
            {
                if (!TryCaptureWorldDocument(request.Metadata, out document, out failure))
                    return false;
            }
            catch (Exception exception)
            {
                failure = PersistenceFailure(
                    "Could not capture persistent state: " + exception.Message,
                    "persistent_capture_failed");
                return false;
            }
            string digest = ComputeWorldDigest(document);
            var envelope = new JObject
            {
                ["digest"] = digest,
                ["document"] = document,
            };
            string serialized = envelope.ToString(Formatting.None);
            int byteCount = Encoding.UTF8.GetByteCount(serialized);
            if (byteCount > MaxWorldFileBytes)
            {
                failure = PersistenceFailure(
                    $"WorldDocument requires {byteCount} bytes; limit is {MaxWorldFileBytes}",
                    "world_document_too_large");
                return false;
            }

            try
            {
                string root = GetWorldStorageRoot();
                Directory.CreateDirectory(root);
                string target = GetWorldSlotPath(request.Slot);
                string[] existing = Directory.GetFiles(root, "*.brworld.json");
                if (!File.Exists(target) && existing.Length >= MaxWorldSlots)
                {
                    failure = PersistenceFailure(
                        $"WorldDocument slot quota of {MaxWorldSlots} is full",
                        "slot_quota_exceeded");
                    return false;
                }

                string currentDigest = null;
                long currentBytes = 0;
                if (File.Exists(target))
                {
                    currentBytes = new FileInfo(target).Length;
                    if (!TryReadWorldEnvelope(request.Slot, out JObject current, out RpcFailure readFailure))
                    {
                        failure = readFailure;
                        return false;
                    }
                    currentDigest = (string)current["digest"];
                }
                if (!string.IsNullOrEmpty(request.ExpectedSlotDigest) &&
                    !string.Equals(currentDigest, request.ExpectedSlotDigest, StringComparison.Ordinal))
                {
                    failure = PersistenceFailure(
                        "WorldDocument slot digest changed before save",
                        "slot_digest_conflict",
                        true);
                    failure.Data["expectedSlotDigest"] = request.ExpectedSlotDigest;
                    failure.Data["actualSlotDigest"] = currentDigest == null
                        ? JValue.CreateNull()
                        : new JValue(currentDigest);
                    return false;
                }
                long totalBytes = existing.Sum(path => new FileInfo(path).Length) - currentBytes + byteCount;
                if (totalBytes > MaxWorldStorageBytes)
                {
                    failure = PersistenceFailure(
                        $"WorldDocument storage would exceed {MaxWorldStorageBytes} bytes",
                        "storage_quota_exceeded");
                    return false;
                }

                AtomicWriteWorld(target, serialized);
                result = new JObject
                {
                    ["slot"] = request.Slot,
                    ["digest"] = digest,
                    ["savedRevision"] = sceneRevision,
                    ["bytes"] = byteCount,
                    ["idempotentReplay"] = false,
                };
                worldSaveReceipts.Add(receiptKey, new WorldSaveReceipt(request, result));
                return true;
            }
            catch (Exception exception)
            {
                failure = PersistenceFailure(
                    "WorldDocument save failed: " + exception.Message,
                    "save_failed",
                    true);
                return false;
            }
        }

        public bool TryPreviewWorldLoad(
            string principalId,
            long connectionEpoch,
            WorldLoadPreviewRequest request,
            out JObject result,
            out RpcFailure failure)
        {
            AssertMainThread();
            result = null;
            failure = null;
            CleanupWorldLoadProposals();
            if (!TryValidateWorldMutationHeader(
                    principalId,
                    request?.Slot,
                    request?.ClientMutationId,
                    request?.ExpectedRevision ?? -1,
                    out failure) || connectionEpoch <= 0)
                return false;
            string mutationKey = WorldMutationKey(principalId, request.ClientMutationId);
            if (worldLoadReceipts.ContainsKey(mutationKey) ||
                worldLoadProposals.Values.Any(candidate => candidate.MutationKey == mutationKey))
            {
                failure = PersistenceFailure(
                    "World load clientMutationId is already reserved in this Player session",
                    "load_mutation_reused");
                return false;
            }
            if (request.ExpectedRevision != sceneRevision)
            {
                failure = RevisionFailure(request.ExpectedRevision);
                return false;
            }
            if (sceneRevision >= 9007199254740991L)
            {
                failure = PersistenceFailure(
                    "WorldDocument load cannot advance an exhausted scene revision",
                    "scene_revision_exhausted");
                return false;
            }
            if (worldLoadProposals.Count >= MaxWorldLoadPreviews)
            {
                failure = PersistenceFailure(
                    "Too many WorldDocument load previews are pending",
                    "load_preview_capacity",
                    true);
                return false;
            }
            if (!TryReadWorldEnvelope(request.Slot, out JObject envelope, out failure) ||
                !TryBuildWorldLoadPlan((JObject)envelope["document"], out WorldLoadPlan plan, out failure))
                return false;

            string token = Guid.NewGuid().ToString("N");
            long expiresAt = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() +
                checked((long)(Mathf.Max(1f, previewLifetimeSeconds) * 1000f));
            var proposal = new WorldLoadProposal
            {
                Token = token,
                MutationKey = mutationKey,
                PrincipalId = principalId,
                ConnectionEpoch = connectionEpoch,
                Slot = request.Slot,
                ClientMutationId = request.ClientMutationId,
                BaseRevision = sceneRevision,
                ExpiresAtUnixMs = expiresAt,
                Digest = (string)envelope["digest"],
                Document = (JObject)envelope["document"].DeepClone(),
            };
            worldLoadProposals.Add(token, proposal);
            result = new JObject
            {
                ["previewToken"] = token,
                ["slot"] = request.Slot,
                ["digest"] = proposal.Digest,
                ["baseRevision"] = sceneRevision,
                ["clientMutationId"] = request.ClientMutationId,
                ["expiresAtUnixMs"] = expiresAt,
                ["summary"] = new JObject
                {
                    ["entities"] = plan.DocumentEntities.Count,
                    ["create"] = plan.CreateCount,
                    ["reuse"] = plan.ReuseCount,
                    ["remove"] = plan.Remove.Count,
                },
            };
            return true;
        }

        public bool TryLoadWorld(
            string principalId,
            long connectionEpoch,
            WorldLoadRequest request,
            out JObject result,
            out RpcFailure failure)
        {
            AssertMainThread();
            result = null;
            failure = null;
            if (request == null || !SceneProtocol.IsIdentifier(principalId, 128) ||
                connectionEpoch <= 0 || !SceneProtocol.IsIdentifier(request.PreviewToken, 128) ||
                !SceneProtocol.IsIdentifier(request.ClientMutationId, 128) || request.ExpectedRevision < 0)
            {
                failure = PersistenceFailure(
                    "persistence/load requires valid previewToken, expectedRevision, and clientMutationId",
                    "invalid_load_params");
                return false;
            }

            string mutationKey = WorldMutationKey(principalId, request.ClientMutationId);
            if (worldLoadReceipts.TryGetValue(mutationKey, out WorldLoadReceipt prior))
            {
                if (!prior.Matches(request))
                {
                    failure = PersistenceFailure(
                        "clientMutationId is already bound to another WorldDocument load",
                        "load_mutation_reused");
                    return false;
                }
                if (prior.Result == null)
                {
                    failure = prior.Failure == null
                        ? PersistenceFailure(
                            "WorldDocument load was previously consumed and did not commit",
                            "load_previously_failed")
                        : CloneFailure(prior.Failure);
                    return false;
                }
                result = (JObject)prior.Result.DeepClone();
                result["idempotentReplay"] = true;
                return true;
            }
            if (worldLoadReceipts.Count >= MaxWorldReceipts)
            {
                failure = PersistenceFailure(
                    "World load receipt capacity is exhausted for this Player session",
                    "load_receipt_capacity",
                    true);
                return false;
            }
            if (!worldLoadProposals.TryGetValue(request.PreviewToken, out WorldLoadProposal proposal) ||
                proposal.ExpiresAtUnixMs < DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() ||
                proposal.BaseRevision != request.ExpectedRevision ||
                !string.Equals(proposal.PrincipalId, principalId, StringComparison.Ordinal) ||
                proposal.ConnectionEpoch != connectionEpoch ||
                !string.Equals(proposal.ClientMutationId, request.ClientMutationId, StringComparison.Ordinal))
            {
                worldLoadProposals.Remove(request.PreviewToken);
                failure = PersistenceFailure(
                    "WorldDocument load preview is expired or does not match this request",
                    "load_preview_mismatch");
                return false;
            }
            worldLoadProposals.Remove(request.PreviewToken);
            var receipt = new WorldLoadReceipt(request);
            worldLoadReceipts.Add(mutationKey, receipt);
            if (sceneRevision != request.ExpectedRevision)
            {
                failure = RevisionFailure(request.ExpectedRevision);
                receipt.Failure = CloneFailure(failure);
                return false;
            }
            if (sceneRevision >= 9007199254740991L)
            {
                failure = PersistenceFailure(
                    "WorldDocument load cannot advance an exhausted scene revision",
                    "scene_revision_exhausted");
                receipt.Failure = CloneFailure(failure);
                return false;
            }
            if (!TryBuildWorldLoadPlan(proposal.Document, out WorldLoadPlan plan, out failure))
            {
                receipt.Failure = CloneFailure(failure);
                return false;
            }
            if (!TryApplyWorldLoad(plan, request.ClientMutationId, out failure))
            {
                receipt.Failure = CloneFailure(failure);
                return false;
            }

            result = new JObject
            {
                ["slot"] = proposal.Slot,
                ["digest"] = proposal.Digest,
                ["sceneRevision"] = sceneRevision,
                ["clientMutationId"] = request.ClientMutationId,
                ["idempotentReplay"] = false,
            };
            receipt.Result = (JObject)result.DeepClone();
            return true;
        }

        private bool TryCaptureWorldDocument(
            WorldSaveMetadata metadata,
            out JObject document,
            out RpcFailure failure)
        {
            document = null;
            failure = null;
            List<RpcObjectIdentity> identities = objects.Values
                .Where(IsAlive)
                .Where(identity => identity.transform == SandboxRoot || identity.transform.IsChildOf(SandboxRoot))
                .OrderBy(identity => identity.StableId, StringComparer.Ordinal)
                .ToList();
            if (identities.Count == 0 || identities.Count > MaxWorldEntities)
            {
                failure = PersistenceFailure(
                    $"WorldDocument requires 1..{MaxWorldEntities} sandbox entities",
                    "world_entity_count");
                return false;
            }

            var registered = new HashSet<RpcObjectIdentity>(identities);
            var entities = new JArray();
            int propertyCount = 0;
            foreach (RpcObjectIdentity identity in identities)
            {
                if (!TryGetPersistentParentId(identity, registered, out string parentId, out failure))
                    return false;
                var components = new JArray();
                foreach (IRpcPropertyAdapter adapter in identity.GetComponents<MonoBehaviour>()
                             .OfType<IRpcPropertyAdapter>()
                             .OrderBy(candidate => candidate.ComponentKey, StringComparer.Ordinal))
                {
                    var properties = new JObject();
                    foreach (RpcPropertyDescriptor descriptor in (adapter.DescribeProperties() ??
                                 Array.Empty<RpcPropertyDescriptor>())
                             .Where(candidate => candidate != null && candidate.Persistent && !candidate.ReadOnly)
                             .OrderBy(candidate => candidate.PropertyId, StringComparer.Ordinal))
                    {
                        if (!adapter.TryRead(descriptor.PropertyId, out JToken value, out string readError) ||
                            !SceneProtocol.IsPropertyValue(value))
                        {
                            failure = PersistenceFailure(
                                $"Could not persist '{identity.StableId}/{adapter.ComponentKey}/{descriptor.PropertyId}': {readError}",
                                "persistent_property_read_failed");
                            return false;
                        }
                        properties[descriptor.PropertyId] = value?.DeepClone() ?? JValue.CreateNull();
                        propertyCount++;
                        if (propertyCount > MaxWorldProperties)
                        {
                            failure = PersistenceFailure(
                                $"WorldDocument exceeds {MaxWorldProperties} persistent properties",
                                "world_property_count");
                            return false;
                        }
                    }
                    if (properties.Count == 0) continue;
                    components.Add(new JObject
                    {
                        ["componentKey"] = adapter.ComponentKey,
                        ["typeId"] = adapter.TypeId,
                        ["properties"] = properties,
                    });
                }

                entities.Add(new JObject
                {
                    ["objectId"] = identity.StableId,
                    ["prefabId"] = string.IsNullOrWhiteSpace(identity.PrefabId)
                        ? JValue.CreateNull()
                        : new JValue(identity.PrefabId),
                    ["parentId"] = parentId == null ? JValue.CreateNull() : new JValue(parentId),
                    ["active"] = identity.gameObject.activeSelf,
                    ["localTransform"] = SerializeTransform(identity.transform),
                    ["components"] = components,
                });
            }

            Scene activeScene = SceneManager.GetActiveScene();
            document = new JObject
            {
                ["formatVersion"] = WorldFormatVersion,
                ["protocolVersion"] = SceneProtocol.Version,
                ["product"] = Application.productName,
                ["buildId"] = string.IsNullOrEmpty(Application.buildGUID) ? Application.version : Application.buildGUID,
                ["sceneId"] = string.IsNullOrEmpty(activeScene.path) ? activeScene.name : activeScene.path,
                ["catalogSchemaVersion"] = prefabCatalog == null ? "none" : prefabCatalog.SchemaVersion,
                ["savedRevision"] = sceneRevision,
                ["savedUnixMs"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                ["metadata"] = metadata == null || string.IsNullOrEmpty(metadata.Label)
                    ? new JObject()
                    : new JObject { ["label"] = metadata.Label },
                ["entities"] = entities,
            };
            return true;
        }

        private bool TryBuildWorldLoadPlan(
            JObject document,
            out WorldLoadPlan plan,
            out RpcFailure failure)
        {
            plan = null;
            failure = null;
            if (!TryValidateWorldDocument(document, out List<JObject> documentEntities, out failure))
                return false;

            List<RpcObjectIdentity> current = objects.Values
                .Where(IsAlive)
                .Where(identity => identity.transform == SandboxRoot || identity.transform.IsChildOf(SandboxRoot))
                .ToList();
            var currentById = current.ToDictionary(identity => identity.StableId, StringComparer.Ordinal);
            var documentById = documentEntities.ToDictionary(entity => (string)entity["objectId"], StringComparer.Ordinal);
            HashSet<string> currentBase = current
                .Where(identity => string.IsNullOrWhiteSpace(identity.PrefabId))
                .Select(identity => identity.StableId)
                .ToHashSet(StringComparer.Ordinal);
            HashSet<string> documentBase = documentEntities
                .Where(entity => entity["prefabId"].Type == JTokenType.Null)
                .Select(entity => (string)entity["objectId"])
                .ToHashSet(StringComparer.Ordinal);
            if (!currentBase.SetEquals(documentBase))
            {
                failure = PersistenceFailure(
                    "WorldDocument base-scene identities do not match the running build",
                    "base_scene_mismatch");
                return false;
            }
            foreach (string baseId in documentBase)
            {
                if (!currentById[baseId].AllowRemoteChanges)
                {
                    failure = PersistenceFailure(
                        $"Base object '{baseId}' does not allow remote restoration",
                        "base_object_not_writable");
                    return false;
                }
            }

            int createCount = 0;
            int reuseCount = 0;
            foreach (JObject entity in documentEntities)
            {
                string objectId = (string)entity["objectId"];
                string prefabId = entity["prefabId"].Type == JTokenType.Null
                    ? null
                    : (string)entity["prefabId"];
                if (currentById.TryGetValue(objectId, out RpcObjectIdentity existing))
                {
                    string existingPrefab = string.IsNullOrWhiteSpace(existing.PrefabId) ? null : existing.PrefabId;
                    if (!string.Equals(existingPrefab, prefabId, StringComparison.Ordinal))
                    {
                        failure = PersistenceFailure(
                            $"Runtime object '{objectId}' has a different prefab identity",
                            "prefab_identity_mismatch");
                        return false;
                    }
                    if (!TryValidatePersistentComponents(existing.gameObject, entity, out failure))
                        return false;
                    reuseCount++;
                }
                else
                {
                    if (prefabId == null || prefabCatalog == null ||
                        !prefabCatalog.TryGet(prefabId, out RuntimePrefabEntry entry) ||
                        entry.Prefab.activeSelf)
                    {
                        failure = PersistenceFailure(
                            $"WorldDocument prefab '{prefabId}' cannot recreate '{objectId}'",
                            "world_prefab_unavailable");
                        return false;
                    }
                    if (!TryValidatePersistentComponents(entry.Prefab, entity, out failure))
                        return false;
                    createCount++;
                }
            }

            List<RpcObjectIdentity> remove = current
                .Where(identity => !string.IsNullOrWhiteSpace(identity.PrefabId) &&
                                   !documentById.ContainsKey(identity.StableId))
                .ToList();
            plan = new WorldLoadPlan
            {
                DocumentEntities = documentEntities,
                CreateCount = createCount,
                ReuseCount = reuseCount,
                Remove = remove,
            };
            return true;
        }

        private bool TryApplyWorldLoad(
            WorldLoadPlan plan,
            string clientMutationId,
            out RpcFailure failure)
        {
            failure = null;
            var inverseActions = new List<IUndoAction>();
            var targets = objects.Values.Where(IsAlive)
                .ToDictionary(identity => identity.StableId, StringComparer.Ordinal);
            var stagedActivations = new List<RpcObjectIdentity>();
            try
            {
                foreach (JObject entity in plan.DocumentEntities)
                {
                    string objectId = (string)entity["objectId"];
                    if (targets.ContainsKey(objectId)) continue;
                    string prefabId = (string)entity["prefabId"];
                    if (!prefabCatalog.TryGet(prefabId, out RuntimePrefabEntry entry))
                        throw new InvalidOperationException($"Prefab '{prefabId}' disappeared after preview");
                    GameObject instance = Instantiate(entry.Prefab, SandboxRoot, false);
                    RpcObjectIdentity identity = instance.GetComponent<RpcObjectIdentity>();
                    if (identity == null)
                    {
                        instance.SetActive(false);
                        Destroy(instance);
                        throw new InvalidOperationException(
                            $"Prefab '{prefabId}' lost its root RpcObjectIdentity after preview");
                    }
                    identity.AssignRuntimeIdentity(objectId, prefabId);
                    if (!RegisterSpawned(identity, out string registrationError))
                    {
                        instance.SetActive(false);
                        Destroy(instance);
                        throw new InvalidOperationException(registrationError);
                    }
                    inverseActions.Add(new SpawnUndoAction(this, identity));
                    targets.Add(objectId, identity);
                }

                foreach (JObject entity in plan.DocumentEntities)
                {
                    string objectId = (string)entity["objectId"];
                    RpcObjectIdentity identity = targets[objectId];
                    string parentId = entity["parentId"].Type == JTokenType.Null
                        ? null
                        : (string)entity["parentId"];
                    Transform desiredParent = parentId == null ? SandboxRoot : targets[parentId].transform;
                    if (identity.transform == SandboxRoot)
                        desiredParent = SandboxRoot.parent;
                    if (identity.transform.parent != desiredParent)
                    {
                        inverseActions.Add(new PersistenceParentUndoAction(identity.transform));
                        identity.transform.SetParent(desiredParent, false);
                    }
                    inverseActions.Add(new TransformUndoAction(identity.transform));
                    ApplyWorldTransform(identity.transform, (JObject)entity["localTransform"]);

                    foreach (JObject component in (JArray)entity["components"])
                    {
                        IRpcPropertyAdapter adapter = FindAdapter(identity.gameObject, (string)component["componentKey"]);
                        var previous = new List<RpcPropertyChange>();
                        foreach (JProperty property in ((JObject)component["properties"]).Properties())
                        {
                            if (!adapter.TryRead(property.Name, out JToken oldValue, out string readError))
                                throw new InvalidOperationException(readError);
                            previous.Add(new RpcPropertyChange
                            {
                                PropertyId = property.Name,
                                Value = oldValue?.DeepClone() ?? JValue.CreateNull(),
                            });
                        }
                        inverseActions.Add(new PropertyUndoAction(adapter, previous));
                        foreach (JProperty property in ((JObject)component["properties"]).Properties())
                        {
                            string validationError = null;
                            string writeError = null;
                            if (!adapter.TryValidate(
                                    property.Name,
                                    property.Value,
                                    out JToken canonical,
                                    out validationError) ||
                                !JToken.DeepEquals(canonical ?? JValue.CreateNull(), property.Value) ||
                                !adapter.TryWrite(property.Name, property.Value, out writeError))
                            {
                                throw new InvalidOperationException(validationError ?? writeError ??
                                    $"Persistent property '{property.Name}' changed validity");
                            }
                        }
                    }

                    bool desiredActive = entity["active"].Value<bool>();
                    if (!string.IsNullOrWhiteSpace(identity.PrefabId) && !identity.gameObject.activeSelf && desiredActive)
                    {
                        inverseActions.Add(new ActiveUndoAction(identity.gameObject));
                        stagedActivations.Add(identity);
                    }
                    else if (identity.gameObject.activeSelf != desiredActive)
                    {
                        inverseActions.Add(new ActiveUndoAction(identity.gameObject));
                        identity.gameObject.SetActive(desiredActive);
                    }
                }

                if (!TryActivateStagedSpawns(stagedActivations, out RpcFailure activationFailure))
                {
                    failure = activationFailure;
                    throw new WorldLoadApplyException();
                }
            }
            catch (Exception exception)
            {
                if (exception is not WorldLoadApplyException)
                {
                    failure = PersistenceFailure(
                        "WorldDocument apply failed: " + exception.Message,
                        "load_apply_failed");
                }
                if (TryRollback(inverseActions, out string rollbackError))
                    return false;

                initializationError = AppendError(
                    initializationError,
                    "WorldDocument rollback failed: " + rollbackError);
                try
                {
                    AdvanceRevision(clientMutationId, "indeterminate WorldDocument rollback");
                }
                catch (Exception revisionException)
                {
                    initializationError = AppendError(
                        initializationError,
                        "could not advance dirty revision: " + revisionException.Message);
                }
                failure = PersistenceFailure(
                    "WorldDocument load failed and could not be completely rolled back",
                    "load_rollback_failed");
                failure.Data["rollbackError"] = rollbackError;
                return false;
            }

            try
            {
                foreach (RpcObjectIdentity extra in plan.Remove)
                {
                    if (!IsAlive(extra)) continue;
                    Unregister(extra);
                    extra.gameObject.SetActive(false);
                    Destroy(extra.gameObject);
                }
            }
            catch (Exception exception)
            {
                initializationError = AppendError(
                    initializationError,
                    "WorldDocument commit cleanup failed: " + exception.Message);
                try
                {
                    AdvanceRevision(clientMutationId, "indeterminate WorldDocument cleanup");
                }
                catch (Exception revisionException)
                {
                    initializationError = AppendError(
                        initializationError,
                        "could not advance dirty revision: " + revisionException.Message);
                }
                failure = PersistenceFailure(
                    "WorldDocument applied but stale entity cleanup did not complete",
                    "load_cleanup_failed");
                return false;
            }
            InvalidateWorldLoadPreviews();
            EstablishExternalMutationBarrier();
            AdvanceRevision(clientMutationId, "loaded WorldDocument");
            return true;
        }

        private bool TryValidateWorldDocument(
            JObject document,
            out List<JObject> entities,
            out RpcFailure failure)
        {
            entities = null;
            failure = null;
            string[] allowed =
            {
                "formatVersion", "protocolVersion", "product", "buildId", "sceneId",
                "catalogSchemaVersion", "savedRevision", "savedUnixMs", "metadata", "entities",
            };
            if (document == null || document.Count != allowed.Length ||
                document.Properties().Any(property => !allowed.Contains(property.Name)) ||
                (string)document["formatVersion"] != WorldFormatVersion ||
                (string)document["protocolVersion"] != SceneProtocol.Version ||
                document["savedRevision"]?.Type != JTokenType.Integer ||
                document["savedUnixMs"]?.Type != JTokenType.Integer ||
                document["metadata"]?.Type != JTokenType.Object ||
                document["entities"] is not JArray entityArray ||
                entityArray.Count < 1 || entityArray.Count > MaxWorldEntities)
            {
                failure = PersistenceFailure(
                    "WorldDocument envelope is invalid or unsupported",
                    "invalid_world_document");
                return false;
            }
            long savedRevision = document["savedRevision"].Value<long>();
            long savedUnixMs = document["savedUnixMs"].Value<long>();
            if (savedRevision < 0 || savedRevision > 9007199254740991L ||
                savedUnixMs < 0 || savedUnixMs > 9007199254740991L ||
                document["metadata"] is not JObject metadataObject ||
                metadataObject.Properties().Any(property => property.Name != "label") ||
                (metadataObject["label"] != null && metadataObject["label"].Type != JTokenType.String))
            {
                failure = PersistenceFailure(
                    "WorldDocument revision, timestamp, or metadata is invalid",
                    "invalid_world_document");
                return false;
            }
            Scene activeScene = SceneManager.GetActiveScene();
            string buildId = string.IsNullOrEmpty(Application.buildGUID) ? Application.version : Application.buildGUID;
            string sceneId = string.IsNullOrEmpty(activeScene.path) ? activeScene.name : activeScene.path;
            string catalogVersion = prefabCatalog == null ? "none" : prefabCatalog.SchemaVersion;
            if ((string)document["product"] != Application.productName ||
                (string)document["buildId"] != buildId ||
                (string)document["sceneId"] != sceneId ||
                (string)document["catalogSchemaVersion"] != catalogVersion)
            {
                failure = PersistenceFailure(
                    "WorldDocument product, build, scene, or Catalog schema is incompatible",
                    "world_document_incompatible");
                return false;
            }
            if (!TryValidateSaveMetadata(metadataObject.ToObject<WorldSaveMetadata>(), out failure))
                return false;

            var ids = new HashSet<string>(StringComparer.Ordinal);
            entities = new List<JObject>(entityArray.Count);
            int propertyCount = 0;
            foreach (JToken token in entityArray)
            {
                if (token is not JObject entity ||
                    !HasExactProperties(entity, "objectId", "prefabId", "parentId", "active", "localTransform", "components") ||
                    !SceneProtocol.IsIdentifier((string)entity["objectId"], 160) ||
                    !ids.Add((string)entity["objectId"]) ||
                    (entity["prefabId"].Type != JTokenType.Null &&
                     !SceneProtocol.IsIdentifier((string)entity["prefabId"], 128)) ||
                    (entity["parentId"].Type != JTokenType.Null &&
                     !SceneProtocol.IsIdentifier((string)entity["parentId"], 160)) ||
                    entity["active"].Type != JTokenType.Boolean ||
                    !TryValidateWorldTransform(entity["localTransform"] as JObject) ||
                    entity["components"] is not JArray components || components.Count > 128)
                {
                    failure = PersistenceFailure(
                        "WorldDocument contains an invalid entity",
                        "invalid_world_entity");
                    return false;
                }
                var componentKeys = new HashSet<string>(StringComparer.Ordinal);
                foreach (JToken componentToken in components)
                {
                    if (componentToken is not JObject component ||
                        !HasExactProperties(component, "componentKey", "typeId", "properties") ||
                        !SceneProtocol.IsIdentifierSegment((string)component["componentKey"], 128) ||
                        !componentKeys.Add((string)component["componentKey"]) ||
                        !SceneProtocol.IsIdentifier((string)component["typeId"], 128) ||
                        component["properties"] is not JObject properties || properties.Count > 128)
                    {
                        failure = PersistenceFailure(
                            "WorldDocument contains an invalid component",
                            "invalid_world_component");
                        return false;
                    }
                    foreach (JProperty property in properties.Properties())
                    {
                        propertyCount++;
                        if (propertyCount > MaxWorldProperties ||
                            !SceneProtocol.IsIdentifierSegment(property.Name, 128) ||
                            !SceneProtocol.IsPropertyValue(property.Value))
                        {
                            failure = PersistenceFailure(
                                "WorldDocument contains an invalid persistent property",
                                "invalid_world_property");
                            return false;
                        }
                    }
                }
                entities.Add(entity);
            }
            foreach (JObject entity in entities)
            {
                string parentId = entity["parentId"].Type == JTokenType.Null
                    ? null
                    : (string)entity["parentId"];
                if (parentId != null && (!ids.Contains(parentId) || parentId == (string)entity["objectId"]))
                {
                    failure = PersistenceFailure(
                        "WorldDocument contains a missing or self-referential parent",
                        "invalid_world_parent");
                    return false;
                }
                var visited = new HashSet<string>(StringComparer.Ordinal);
                JObject current = entity;
                while (current["parentId"].Type != JTokenType.Null)
                {
                    string next = (string)current["parentId"];
                    if (!visited.Add(next))
                    {
                        failure = PersistenceFailure(
                            "WorldDocument parent graph contains a cycle",
                            "world_parent_cycle");
                        return false;
                    }
                    current = entities.First(candidate => (string)candidate["objectId"] == next);
                }
            }
            return true;
        }

        private bool TryValidatePersistentComponents(
            GameObject target,
            JObject entity,
            out RpcFailure failure)
        {
            failure = null;
            foreach (JObject component in (JArray)entity["components"])
            {
                IRpcPropertyAdapter adapter = FindAdapter(target, (string)component["componentKey"]);
                if (adapter == null || !string.Equals(adapter.TypeId, (string)component["typeId"], StringComparison.Ordinal))
                {
                    failure = PersistenceFailure(
                        $"Persistent component '{component["componentKey"]}' is unavailable on '{entity["objectId"]}'",
                        "persistent_component_unavailable");
                    return false;
                }
                var descriptors = new Dictionary<string, RpcPropertyDescriptor>(StringComparer.Ordinal);
                try
                {
                    foreach (RpcPropertyDescriptor descriptor in
                             adapter.DescribeProperties() ?? Array.Empty<RpcPropertyDescriptor>())
                    {
                        if (descriptor == null ||
                            !SceneProtocol.IsIdentifierSegment(descriptor.PropertyId, 128) ||
                            !descriptors.TryAdd(descriptor.PropertyId, descriptor))
                        {
                            failure = PersistenceFailure(
                                $"Persistent component '{component["componentKey"]}' exposes an invalid or duplicate property descriptor",
                                "persistent_component_contract");
                            return false;
                        }
                    }
                }
                catch (Exception exception)
                {
                    failure = PersistenceFailure(
                        $"Persistent component '{component["componentKey"]}' could not describe its properties: {exception.Message}",
                        "persistent_component_contract");
                    return false;
                }
                foreach (JProperty property in ((JObject)component["properties"]).Properties())
                {
                    string validationError = null;
                    RpcPropertyDescriptor descriptor;
                    JToken canonical = null;
                    bool valid;
                    try
                    {
                        valid = descriptors.TryGetValue(property.Name, out descriptor) &&
                                descriptor.Persistent &&
                                !descriptor.ReadOnly &&
                                adapter.TryValidate(
                                    property.Name,
                                    property.Value,
                                    out canonical,
                                    out validationError) &&
                                JToken.DeepEquals(canonical ?? JValue.CreateNull(), property.Value);
                    }
                    catch (Exception exception)
                    {
                        validationError = exception.Message;
                        valid = false;
                    }
                    if (!valid)
                    {
                        failure = PersistenceFailure(
                            $"Persistent property '{property.Name}' is unavailable or invalid: {validationError}",
                            "persistent_property_invalid");
                        return false;
                    }
                }
            }
            return true;
        }

        private bool TryReadWorldEnvelope(
            string slot,
            out JObject envelope,
            out RpcFailure failure)
        {
            envelope = null;
            failure = null;
            if (!IsWorldSlot(slot))
            {
                failure = PersistenceFailure("WorldDocument slot is invalid", "invalid_slot");
                return false;
            }
            try
            {
                string path = GetWorldSlotPath(slot);
                if (!File.Exists(path))
                {
                    failure = PersistenceFailure(
                        $"WorldDocument slot '{slot}' does not exist",
                        "slot_not_found");
                    return false;
                }
                var info = new FileInfo(path);
                if (info.Length < 1 || info.Length > MaxWorldFileBytes)
                {
                    failure = PersistenceFailure(
                        "WorldDocument file exceeds the configured bounds",
                        "world_file_size");
                    return false;
                }
                string text = File.ReadAllText(path, new UTF8Encoding(false, true));
                envelope = JObject.Parse(text, new JsonLoadSettings
                {
                    DuplicatePropertyNameHandling = DuplicatePropertyNameHandling.Error,
                });
                if (!HasExactProperties(envelope, "digest", "document") ||
                    envelope["digest"]?.Type != JTokenType.String ||
                    envelope["document"] is not JObject document ||
                    !string.Equals((string)envelope["digest"], ComputeWorldDigest(document), StringComparison.Ordinal))
                {
                    failure = PersistenceFailure(
                        "WorldDocument digest or envelope is invalid",
                        "world_digest_mismatch");
                    envelope = null;
                    return false;
                }
                return true;
            }
            catch (Exception exception)
            {
                failure = PersistenceFailure(
                    "WorldDocument could not be read: " + exception.Message,
                    "world_read_failed");
                envelope = null;
                return false;
            }
        }

        private static JObject SerializeTransform(Transform target)
        {
            return new JObject
            {
                ["position"] = SerializeVector(target.localPosition),
                ["rotationEuler"] = SerializeVector(target.localEulerAngles),
                ["scale"] = SerializeVector(target.localScale),
            };
        }

        private static JObject SerializeVector(Vector3 value)
        {
            return new JObject
            {
                ["x"] = value.x,
                ["y"] = value.y,
                ["z"] = value.z,
            };
        }

        private static bool TryValidateWorldTransform(JObject transform)
        {
            if (transform == null ||
                !HasExactProperties(transform, "position", "rotationEuler", "scale")) return false;
            foreach (string key in new[] { "position", "rotationEuler", "scale" })
            {
                if (transform[key] is not JObject vector ||
                    !HasExactProperties(vector, "x", "y", "z")) return false;
                foreach (string axis in new[] { "x", "y", "z" })
                {
                    if (vector[axis].Type != JTokenType.Integer && vector[axis].Type != JTokenType.Float)
                        return false;
                    double value = vector[axis].Value<double>();
                    if (double.IsNaN(value) || double.IsInfinity(value) || Math.Abs(value) > float.MaxValue)
                        return false;
                }
            }
            return true;
        }

        private static void ApplyWorldTransform(Transform target, JObject transform)
        {
            target.localPosition = ReadWorldVector((JObject)transform["position"]);
            target.localEulerAngles = ReadWorldVector((JObject)transform["rotationEuler"]);
            target.localScale = ReadWorldVector((JObject)transform["scale"]);
        }

        private static Vector3 ReadWorldVector(JObject value)
        {
            return new Vector3(
                value["x"].Value<float>(),
                value["y"].Value<float>(),
                value["z"].Value<float>());
        }

        private bool TryGetPersistentParentId(
            RpcObjectIdentity identity,
            HashSet<RpcObjectIdentity> registered,
            out string parentId,
            out RpcFailure failure)
        {
            parentId = null;
            failure = null;
            if (identity.transform == SandboxRoot) return true;
            Transform parent = identity.transform.parent;
            if (parent == null)
            {
                failure = PersistenceFailure(
                    $"Sandbox entity '{identity.StableId}' has no parent",
                    "unsupported_world_topology");
                return false;
            }
            RpcObjectIdentity parentIdentity = parent.GetComponent<RpcObjectIdentity>();
            if (parentIdentity != null && registered.Contains(parentIdentity))
            {
                parentId = parentIdentity.StableId;
                return true;
            }
            if (parent == SandboxRoot) return true;
            failure = PersistenceFailure(
                $"Sandbox entity '{identity.StableId}' has an unregistered intermediate parent",
                "unsupported_world_topology");
            return false;
        }

        private static IRpcPropertyAdapter FindAdapter(GameObject target, string componentKey)
        {
            return target == null
                ? null
                : target.GetComponents<MonoBehaviour>()
                    .OfType<IRpcPropertyAdapter>()
                    .FirstOrDefault(candidate =>
                        string.Equals(candidate.ComponentKey, componentKey, StringComparison.Ordinal));
        }

        private static bool HasExactProperties(JObject value, params string[] names)
        {
            return value != null && value.Count == names.Length &&
                   names.All(name => value.Property(name) != null);
        }

        private bool TryValidateWorldMutationHeader(
            string principalId,
            string slot,
            string clientMutationId,
            long expectedRevision,
            out RpcFailure failure)
        {
            if (!SceneProtocol.IsIdentifier(principalId, 128) || !IsWorldSlot(slot) ||
                !SceneProtocol.IsIdentifier(clientMutationId, 128) || expectedRevision < 0 ||
                expectedRevision > 9007199254740991L)
            {
                failure = PersistenceFailure(
                    "WorldDocument request has an invalid slot, revision, principal, or clientMutationId",
                    "invalid_persistence_params");
                return false;
            }
            failure = null;
            return true;
        }

        private static bool TryValidateSaveMetadata(WorldSaveMetadata metadata, out RpcFailure failure)
        {
            if (metadata?.Label != null && metadata.Label.Length > 256)
            {
                failure = PersistenceFailure(
                    "WorldDocument label must not exceed 256 characters",
                    "invalid_world_metadata");
                return false;
            }
            failure = null;
            return true;
        }

        private static bool IsWorldSlot(string slot)
        {
            if (string.IsNullOrEmpty(slot) || slot.Length > 64 ||
                !char.IsLetterOrDigit(slot[0])) return false;
            return slot.All(character => character <= 127 &&
                (char.IsLetterOrDigit(character) || character == '_' || character == '-'));
        }

        private static bool IsWorldDigest(string digest)
        {
            if (digest == null || !digest.StartsWith("sha256:", StringComparison.Ordinal) ||
                digest.Length != "sha256:".Length + 64) return false;
            return digest.Skip("sha256:".Length)
                .All(character => character >= '0' && character <= '9' ||
                                  character >= 'a' && character <= 'f');
        }

        private string GetWorldStorageRoot()
        {
            return Path.GetFullPath(Path.Combine(
                Application.persistentDataPath,
                "BrainRegion",
                "Worlds"));
        }

        private string GetWorldSlotPath(string slot)
        {
            if (!IsWorldSlot(slot)) throw new ArgumentException("Invalid WorldDocument slot", nameof(slot));
            string root = GetWorldStorageRoot();
            string path = Path.GetFullPath(Path.Combine(root, slot + ".brworld.json"));
            string prefix = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) +
                Path.DirectorySeparatorChar;
            if (!path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("WorldDocument slot escaped its storage root");
            return path;
        }

        private static void AtomicWriteWorld(string target, string text)
        {
            string temporary = target + ".tmp." + Guid.NewGuid().ToString("N");
            string backup = target + ".bak";
            try
            {
                byte[] bytes = new UTF8Encoding(false, true).GetBytes(text);
                using (var stream = new FileStream(
                           temporary,
                           FileMode.CreateNew,
                           FileAccess.Write,
                           FileShare.None,
                           4096,
                           FileOptions.WriteThrough))
                {
                    stream.Write(bytes, 0, bytes.Length);
                    stream.Flush(true);
                }
                if (File.Exists(target))
                {
                    if (File.Exists(backup)) File.Delete(backup);
                    File.Replace(temporary, target, backup);
                    try
                    {
                        if (File.Exists(backup)) File.Delete(backup);
                    }
                    catch
                    {
                        // The target is already committed. A stale backup is safer
                        // than reporting failure and inviting a duplicate retry.
                    }
                }
                else
                {
                    File.Move(temporary, target);
                }
            }
            finally
            {
                if (File.Exists(temporary)) File.Delete(temporary);
            }
        }

        private static string ComputeWorldDigest(JObject document)
        {
            using (SHA256 sha256 = SHA256.Create())
            {
                byte[] bytes = Encoding.UTF8.GetBytes(document.ToString(Formatting.None));
                byte[] digest = sha256.ComputeHash(bytes);
                return "sha256:" + BitConverter.ToString(digest).Replace("-", string.Empty).ToLowerInvariant();
            }
        }

        private static string WorldMutationKey(string principalId, string clientMutationId)
        {
            return principalId + "\n" + clientMutationId;
        }

        private void CleanupWorldLoadProposals()
        {
            long now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            foreach (string token in worldLoadProposals
                         .Where(pair => pair.Value.ExpiresAtUnixMs < now)
                         .Select(pair => pair.Key)
                         .ToArray())
                worldLoadProposals.Remove(token);
        }

        private void InvalidateWorldLoadPreviews()
        {
            worldLoadProposals.Clear();
        }

        private RpcFailure RevisionFailure(long expectedRevision)
        {
            RpcFailure failure = RpcFailure.Create(
                SceneErrorCodes.RevisionConflict,
                "WorldDocument request revision does not match the runtime scene",
                "revision_conflict",
                true);
            failure.Data["expectedRevision"] = expectedRevision;
            failure.Data["actualRevision"] = sceneRevision;
            return failure;
        }

        private static RpcFailure PersistenceFailure(
            string message,
            string reason,
            bool retryable = false)
        {
            return RpcFailure.Create(SceneErrorCodes.PersistenceError, message, reason, retryable);
        }

        private static RpcFailure CloneFailure(RpcFailure source)
        {
            return source == null
                ? null
                : new RpcFailure
                {
                    Code = source.Code,
                    Message = source.Message,
                    Data = source.Data == null ? new JObject() : (JObject)source.Data.DeepClone(),
                };
        }

        private sealed class WorldLoadPlan
        {
            public List<JObject> DocumentEntities;
            public int CreateCount;
            public int ReuseCount;
            public List<RpcObjectIdentity> Remove;
        }

        private sealed class WorldLoadProposal
        {
            public string Token;
            public string MutationKey;
            public string PrincipalId;
            public long ConnectionEpoch;
            public string Slot;
            public string ClientMutationId;
            public long BaseRevision;
            public long ExpiresAtUnixMs;
            public string Digest;
            public JObject Document;
        }

        private sealed class WorldSaveReceipt
        {
            private readonly string slot;
            private readonly long expectedRevision;
            private readonly string expectedSlotDigest;
            private readonly string label;
            public readonly JObject Result;

            public WorldSaveReceipt(WorldSaveRequest request, JObject result)
            {
                slot = request.Slot;
                expectedRevision = request.ExpectedRevision;
                expectedSlotDigest = request.ExpectedSlotDigest;
                label = request.Metadata?.Label;
                Result = (JObject)result.DeepClone();
            }

            public bool Matches(WorldSaveRequest request)
            {
                return request != null && slot == request.Slot &&
                       expectedRevision == request.ExpectedRevision &&
                       expectedSlotDigest == request.ExpectedSlotDigest &&
                       label == request.Metadata?.Label;
            }
        }

        private sealed class WorldLoadReceipt
        {
            private readonly string previewToken;
            private readonly long expectedRevision;
            public JObject Result;
            public RpcFailure Failure;

            public WorldLoadReceipt(WorldLoadRequest request)
            {
                previewToken = request.PreviewToken;
                expectedRevision = request.ExpectedRevision;
            }

            public bool Matches(WorldLoadRequest request)
            {
                return request != null && previewToken == request.PreviewToken &&
                       expectedRevision == request.ExpectedRevision;
            }
        }

        private sealed class PersistenceParentUndoAction : IUndoAction
        {
            private readonly Transform target;
            private readonly Transform previousParent;
            private readonly int previousSiblingIndex;

            public PersistenceParentUndoAction(Transform target)
            {
                this.target = target;
                previousParent = target.parent;
                previousSiblingIndex = target.GetSiblingIndex();
            }

            public bool CanRevert(out string error)
            {
                error = target == null ? "Parent target no longer exists" : null;
                return target != null;
            }

            public bool Revert(out string error)
            {
                if (!CanRevert(out error)) return false;
                target.SetParent(previousParent, false);
                target.SetSiblingIndex(previousSiblingIndex);
                return true;
            }
        }

        private sealed class WorldLoadApplyException : Exception
        {
        }
    }
}
