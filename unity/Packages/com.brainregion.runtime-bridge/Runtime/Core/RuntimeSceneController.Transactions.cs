using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace BrainRegion.RuntimeBridge
{
    public sealed partial class RuntimeSceneController
    {
        private const int MaxReceiptEntries = 256;
        private const string LegacyLocalPrincipalId = "legacy-local";

        private readonly Dictionary<string, PendingProposal> proposals =
            new Dictionary<string, PendingProposal>(StringComparer.Ordinal);
        private readonly Dictionary<string, string> proposalByMutation =
            new Dictionary<string, string>(StringComparer.Ordinal);
        private readonly Dictionary<string, SceneMutationResult> mutationReceipts =
            new Dictionary<string, SceneMutationResult>(StringComparer.Ordinal);
        private readonly Dictionary<string, MutationTombstone> mutationTombstones =
            new Dictionary<string, MutationTombstone>(StringComparer.Ordinal);
        private readonly Queue<string> receiptOrder = new Queue<string>();
        private readonly LinkedList<UndoRecord> undoHistory = new LinkedList<UndoRecord>();

        public bool TryPreview(
            ScenePreviewRequest request,
            out ScenePreviewResult result,
            out RpcFailure failure)
        {
            return TryPreview(LegacyLocalPrincipalId, request, out result, out failure);
        }

        public bool TryPreview(
            string principalId,
            ScenePreviewRequest request,
            out ScenePreviewResult result,
            out RpcFailure failure)
        {
            AssertMainThread();
            result = null;
            failure = null;
            if (!TryRequireReady(out failure)) return false;
            CleanupExpiredProposals();

            if (!TryValidatePrincipal(principalId, out failure)) return false;
            if (!TryValidateMutationHeader(request, out failure)) return false;
            if (!TryCheckRevision(request.ExpectedRevision, out failure)) return false;

            string mutationKey = MakeMutationKey(principalId, request.ClientMutationId);
            if (mutationTombstones.TryGetValue(mutationKey, out MutationTombstone tombstone))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.ValidationFailed,
                    $"Mutation '{request.ClientMutationId}' was already consumed in this runtime session",
                    "mutation_id_already_consumed");
                AddMutationStatus(failure, tombstone);
                return false;
            }

            var canonicalCommands = new List<SceneOperation>(request.Commands.Count);
            var summary = new List<string>(request.Commands.Count);
            var tempIds = new HashSet<string>(StringComparer.Ordinal);
            foreach (SceneOperation input in request.Commands)
            {
                if (!TryValidateOperation(input, tempIds, out SceneOperation canonical, out string description, out failure))
                    return false;
                canonicalCommands.Add(canonical);
                summary.Add(description);
            }

            string canonicalJson = JsonConvert.SerializeObject(canonicalCommands, Formatting.None);
            string digest = ComputeProposalDigest(
                sessionId,
                principalId,
                request.ExpectedRevision,
                canonicalJson);
            if (proposalByMutation.TryGetValue(mutationKey, out string existingToken) &&
                proposals.TryGetValue(existingToken, out PendingProposal existing))
            {
                if (existing.SessionId == sessionId &&
                    existing.PrincipalId == principalId &&
                    existing.BaseRevision == request.ExpectedRevision &&
                    existing.Digest == digest)
                {
                    result = existing.ToResult();
                    return true;
                }

                failure = RpcFailure.Create(
                    SceneErrorCodes.ValidationFailed,
                    "clientMutationId is already bound to a different preview",
                    "mutation_id_reused");
                return false;
            }

            long expiresAtUnixMs = DateTimeOffset.UtcNow
                .AddSeconds(Mathf.Max(1f, previewLifetimeSeconds))
                .ToUnixTimeMilliseconds();
            var proposal = new PendingProposal
            {
                Token = Guid.NewGuid().ToString("N"),
                SessionId = sessionId,
                PrincipalId = principalId,
                MutationKey = mutationKey,
                ClientMutationId = request.ClientMutationId,
                BaseRevision = request.ExpectedRevision,
                ExpiresAtUnixMs = expiresAtUnixMs,
                Commands = canonicalCommands,
                CanonicalJson = canonicalJson,
                Digest = digest,
                Summary = summary,
            };
            proposals.Add(proposal.Token, proposal);
            proposalByMutation.Add(proposal.MutationKey, proposal.Token);
            result = proposal.ToResult();
            return true;
        }

        public bool TryApply(
            SceneApplyRequest request,
            out SceneMutationResult result,
            out RpcFailure failure)
        {
            return TryApply(LegacyLocalPrincipalId, request, out result, out failure);
        }

        public bool TryApply(
            string principalId,
            SceneApplyRequest request,
            out SceneMutationResult result,
            out RpcFailure failure)
        {
            AssertMainThread();
            result = null;
            failure = null;
            if (!TryRequireReady(out failure)) return false;
            CleanupExpiredProposals();

            if (!TryValidatePrincipal(principalId, out failure)) return false;
            if (request == null || !SceneProtocol.IsIdentifier(request.ClientMutationId, 128) ||
                !SceneProtocol.IsIdentifier(request.PreviewToken, 128))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    "scene/apply requires previewToken and clientMutationId",
                    "missing_apply_fields");
                return false;
            }

            string mutationKey = MakeMutationKey(principalId, request.ClientMutationId);
            if (mutationTombstones.TryGetValue(mutationKey, out MutationTombstone existingTombstone))
            {
                return TryReplayMutation(
                    existingTombstone,
                    principalId,
                    request,
                    mutationKey,
                    out result,
                    out failure);
            }
            if (!TryCheckRevision(request.ExpectedRevision, out failure)) return false;

            if (!proposals.TryGetValue(request.PreviewToken, out PendingProposal proposal))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.PreviewExpired,
                    "Preview token is unknown or expired",
                    "preview_expired_or_unknown");
                return false;
            }
            if (!string.Equals(proposal.SessionId, sessionId, StringComparison.Ordinal) ||
                !string.Equals(proposal.PrincipalId, principalId, StringComparison.Ordinal) ||
                !string.Equals(proposal.ClientMutationId, request.ClientMutationId, StringComparison.Ordinal) ||
                proposal.BaseRevision != request.ExpectedRevision ||
                proposal.ExpiresAtUnixMs < DateTimeOffset.UtcNow.ToUnixTimeMilliseconds())
            {
                if (proposal.ExpiresAtUnixMs < DateTimeOffset.UtcNow.ToUnixTimeMilliseconds())
                    RemoveProposal(proposal);
                failure = RpcFailure.Create(
                    SceneErrorCodes.PreviewExpired,
                    "Preview token does not match this mutation or revision",
                    "preview_mismatch");
                return false;
            }

            // Consume the token and reserve the mutation id before revalidation or
            // execution. From this point onward every retry observes the tombstone and
            // can never execute this mutation a second time.
            RemoveProposal(proposal);
            var tombstone = new MutationTombstone
            {
                SessionId = sessionId,
                PrincipalId = principalId,
                ClientMutationId = request.ClientMutationId,
                BaseRevision = proposal.BaseRevision,
                Digest = proposal.Digest,
                PreviewToken = proposal.Token,
                Status = MutationStatus.Applying,
                LastKnownRevision = sceneRevision,
            };
            mutationTombstones.Add(mutationKey, tombstone);

            if (!TryPreflightProposal(proposal, out failure))
            {
                tombstone.Status = MutationStatus.Rejected;
                tombstone.LastKnownRevision = sceneRevision;
                AddMutationStatus(failure, tombstone);
                return false;
            }

            var inverseActions = new List<IUndoAction>();
            var tempIdMap = new Dictionary<string, string>(StringComparer.Ordinal);
            RpcFailure executionFailure = null;
            try
            {
                foreach (SceneOperation command in proposal.Commands)
                {
                    if (!TryExecuteOperation(command, inverseActions, tempIdMap, out failure))
                    {
                        executionFailure = failure;
                        break;
                    }
                }
            }
            catch (Exception exception)
            {
                Debug.LogError("[BrainRegion] Scene RPC transaction threw during apply: " + exception);
                executionFailure = RpcFailure.Create(
                    SceneErrorCodes.Internal,
                    "Scene RPC transaction failed during execution",
                    "transaction_exception");
            }
            if (executionFailure != null)
            {
                return FinishFailedApply(
                    request.ClientMutationId,
                    tombstone,
                    inverseActions,
                    executionFailure,
                    out failure);
            }

            string undoId = Guid.NewGuid().ToString("N");
            var undoRecord = new UndoRecord
            {
                UndoId = undoId,
                MutationKey = mutationKey,
                OwnerPrincipalId = principalId,
                ClientMutationId = request.ClientMutationId,
                Actions = inverseActions,
            };
            undoHistory.AddLast(undoRecord);
            while (undoHistory.Count > Mathf.Max(1, maxUndoEntries))
                undoHistory.RemoveFirst();

            AdvanceRevision(request.ClientMutationId, string.Join("; ", proposal.Summary));
            result = new SceneMutationResult
            {
                SceneRevision = sceneRevision,
                ClientMutationId = request.ClientMutationId,
                UndoId = undoId,
                TempIdMap = tempIdMap,
            };
            tombstone.Status = MutationStatus.Applied;
            tombstone.LastKnownRevision = sceneRevision;
            RememberReceipt(mutationKey, result);
            return true;
        }

        public bool TryUndo(SceneUndoRequest request, out JObject result, out RpcFailure failure)
        {
            return TryUndo(LegacyLocalPrincipalId, request, out result, out failure);
        }

        public bool TryUndo(
            string principalId,
            SceneUndoRequest request,
            out JObject result,
            out RpcFailure failure)
        {
            AssertMainThread();
            result = null;
            failure = null;
            if (!TryRequireReady(out failure)) return false;
            if (!TryValidatePrincipal(principalId, out failure)) return false;
            if (request == null)
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    "history/undo requires params",
                    "missing_undo_params");
                return false;
            }
            if (!TryCheckRevision(request.ExpectedRevision, out failure)) return false;
            if (undoHistory.Last == null)
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.NotReversible,
                    "There is no Scene RPC transaction to undo",
                    "undo_history_empty");
                return false;
            }

            UndoRecord record = undoHistory.Last.Value;
            if (!string.Equals(record.OwnerPrincipalId, principalId, StringComparison.Ordinal))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.Forbidden,
                    "The latest Scene RPC transaction belongs to another principal",
                    "undo_owner_mismatch");
                return false;
            }
            if (!string.IsNullOrWhiteSpace(request.UndoId) &&
                !string.Equals(request.UndoId, record.UndoId, StringComparison.Ordinal))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.NotReversible,
                    "Runtime undo is linear; only the most recent transaction can be reverted",
                    "undo_not_top_of_stack");
                failure.Data["latestUndoId"] = record.UndoId;
                return false;
            }

            for (int index = record.Actions.Count - 1; index >= 0; index--)
            {
                if (!TryCanRevertSafely(record.Actions[index], out string error))
                {
                    failure = RpcFailure.Create(
                        SceneErrorCodes.NotReversible,
                        "Undo preflight failed: " + error,
                        "undo_preflight_failed");
                    return false;
                }
            }
            for (int index = record.Actions.Count - 1; index >= 0; index--)
            {
                if (!TryRevertSafely(record.Actions[index], out string error))
                {
                    initializationError = AppendError(initializationError, "undo failed: " + error);
                    mutationTombstones.TryGetValue(record.MutationKey, out MutationTombstone dirty);
                    if (dirty != null)
                        dirty.Status = MutationStatus.Indeterminate;
                    try
                    {
                        AdvanceRevision(record.ClientMutationId, "indeterminate undo " + record.UndoId);
                    }
                    catch (Exception exception)
                    {
                        initializationError = AppendError(
                            initializationError,
                            "could not advance dirty undo revision: " + exception.Message);
                    }
                    if (dirty != null)
                        dirty.LastKnownRevision = sceneRevision;
                    failure = RpcFailure.Create(
                        SceneErrorCodes.Internal,
                        "Undo failed and the runtime bridge was degraded: " + error,
                        "undo_failed");
                    return false;
                }
            }

            undoHistory.RemoveLast();
            AdvanceRevision(record.ClientMutationId, "undo " + record.UndoId);
            if (mutationTombstones.TryGetValue(record.MutationKey, out MutationTombstone undone))
            {
                undone.Status = MutationStatus.Undone;
                undone.LastKnownRevision = sceneRevision;
            }
            mutationReceipts.Remove(record.MutationKey);
            result = new JObject
            {
                ["sceneRevision"] = sceneRevision,
                ["undoId"] = record.UndoId,
                ["undoneClientMutationId"] = record.ClientMutationId,
            };
            return true;
        }

        private bool TryPreflightProposal(PendingProposal proposal, out RpcFailure failure)
        {
            failure = null;
            if (proposal == null ||
                !string.Equals(proposal.SessionId, sessionId, StringComparison.Ordinal))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.PreviewExpired,
                    "Preview belongs to another runtime session",
                    "preview_session_mismatch");
                return false;
            }

            try
            {
                var canonicalCommands = new List<SceneOperation>(proposal.Commands.Count);
                var summary = new List<string>(proposal.Commands.Count);
                var tempIds = new HashSet<string>(StringComparer.Ordinal);
                foreach (SceneOperation input in proposal.Commands)
                {
                    if (!TryValidateOperation(
                            input,
                            tempIds,
                            out SceneOperation canonical,
                            out string description,
                            out failure))
                    {
                        return false;
                    }
                    canonicalCommands.Add(canonical);
                    summary.Add(description);
                }

                string canonicalJson = JsonConvert.SerializeObject(canonicalCommands, Formatting.None);
                string digest = ComputeProposalDigest(
                    proposal.SessionId,
                    proposal.PrincipalId,
                    proposal.BaseRevision,
                    canonicalJson);
                if (!string.Equals(proposal.CanonicalJson, canonicalJson, StringComparison.Ordinal) ||
                    !string.Equals(proposal.Digest, digest, StringComparison.Ordinal))
                {
                    failure = RpcFailure.Create(
                        SceneErrorCodes.PreviewExpired,
                        "Preview no longer canonicalizes to the approved command batch",
                        "preview_digest_mismatch");
                    return false;
                }
                if (sceneRevision >= 9007199254740991L)
                {
                    failure = RpcFailure.Create(
                        SceneErrorCodes.Internal,
                        "Scene revision exhausted the JSON safe-integer range",
                        "revision_exhausted");
                    return false;
                }

                // Execute fresh clones produced by the second validation pass. This
                // prevents adapters from retaining mutable JToken instances supplied
                // by the original request.
                proposal.Commands = canonicalCommands;
                proposal.Summary = summary;
                return true;
            }
            catch (Exception exception)
            {
                Debug.LogError("[BrainRegion] Scene RPC apply preflight threw: " + exception);
                failure = RpcFailure.Create(
                    SceneErrorCodes.ValidationFailed,
                    "Scene RPC apply preflight failed",
                    "preflight_exception");
                return false;
            }
        }

        private bool TryReplayMutation(
            MutationTombstone tombstone,
            string principalId,
            SceneApplyRequest request,
            string mutationKey,
            out SceneMutationResult result,
            out RpcFailure failure)
        {
            result = null;
            failure = null;
            bool exactBinding = tombstone != null &&
                string.Equals(tombstone.SessionId, sessionId, StringComparison.Ordinal) &&
                string.Equals(tombstone.PrincipalId, principalId, StringComparison.Ordinal) &&
                string.Equals(tombstone.ClientMutationId, request.ClientMutationId, StringComparison.Ordinal) &&
                string.Equals(tombstone.PreviewToken, request.PreviewToken, StringComparison.Ordinal) &&
                tombstone.BaseRevision == request.ExpectedRevision;
            if (!exactBinding)
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.ValidationFailed,
                    "clientMutationId is already bound to another Scene RPC transaction",
                    "mutation_id_reused");
                AddMutationStatus(failure, tombstone);
                return false;
            }

            switch (tombstone.Status)
            {
                case MutationStatus.Applied:
                    if (mutationReceipts.TryGetValue(mutationKey, out SceneMutationResult receipt))
                    {
                        result = CloneReceipt(receipt);
                        result.IdempotentReplay = true;
                        return true;
                    }
                    failure = RpcFailure.Create(
                        SceneErrorCodes.ValidationFailed,
                        "Mutation was applied, but its replay receipt has expired",
                        "mutation_receipt_expired");
                    break;

                case MutationStatus.Undone:
                    failure = RpcFailure.Create(
                        SceneErrorCodes.NotReversible,
                        "Mutation was applied and subsequently undone; it cannot replay as success",
                        "mutation_was_undone");
                    break;

                case MutationStatus.Applying:
                    failure = RpcFailure.Create(
                        SceneErrorCodes.Busy,
                        "Mutation is already being applied",
                        "mutation_in_progress",
                        true);
                    break;

                case MutationStatus.Indeterminate:
                    failure = RpcFailure.Create(
                        SceneErrorCodes.Internal,
                        "Mutation outcome is indeterminate because rollback or undo failed",
                        "mutation_indeterminate");
                    break;

                case MutationStatus.Rejected:
                    failure = RpcFailure.Create(
                        SceneErrorCodes.ValidationFailed,
                        "Mutation was consumed and rejected during apply preflight",
                        "mutation_previously_rejected");
                    break;

                default:
                    failure = RpcFailure.Create(
                        SceneErrorCodes.ValidationFailed,
                        "Mutation failed and was rolled back; submit a new clientMutationId",
                        "mutation_previously_rolled_back");
                    break;
            }
            AddMutationStatus(failure, tombstone);
            return false;
        }

        private bool FinishFailedApply(
            string clientMutationId,
            MutationTombstone tombstone,
            List<IUndoAction> inverseActions,
            RpcFailure cause,
            out RpcFailure failure)
        {
            cause = cause ?? RpcFailure.Create(
                SceneErrorCodes.Internal,
                "Scene RPC transaction failed",
                "transaction_failed");
            if (TryRollback(inverseActions, out string rollbackError))
            {
                tombstone.Status = MutationStatus.RolledBack;
                tombstone.LastKnownRevision = sceneRevision;
                failure = cause;
                AddMutationStatus(failure, tombstone);
                return false;
            }

            tombstone.Status = MutationStatus.Indeterminate;
            initializationError = AppendError(
                initializationError,
                "transaction rollback failed: " + rollbackError);
            try
            {
                AdvanceRevision(clientMutationId, "indeterminate transaction rollback");
            }
            catch (Exception exception)
            {
                initializationError = AppendError(
                    initializationError,
                    "could not advance dirty revision: " + exception.Message);
            }
            tombstone.LastKnownRevision = sceneRevision;

            failure = RpcFailure.Create(
                SceneErrorCodes.Internal,
                "Scene RPC transaction failed and could not be completely rolled back",
                "transaction_rollback_failed");
            failure.Data["rollbackError"] = rollbackError;
            failure.Data["causeReason"] = cause.Data?["reason"]?.DeepClone();
            AddMutationStatus(failure, tombstone);
            return false;
        }

        private static bool TryValidatePrincipal(string principalId, out RpcFailure failure)
        {
            if (SceneProtocol.IsIdentifier(principalId, 128))
            {
                failure = null;
                return true;
            }
            failure = RpcFailure.Create(
                SceneErrorCodes.InvalidParams,
                "Authenticated Scene RPC principalId is invalid",
                "invalid_principal_id");
            return false;
        }

        private static string MakeMutationKey(string principalId, string clientMutationId)
        {
            // Both inputs are restricted wire identifiers, so the separator cannot
            // collide with a valid value.
            return principalId + "|" + clientMutationId;
        }

        private static string ComputeProposalDigest(
            string proposalSessionId,
            string principalId,
            long baseRevision,
            string canonicalJson)
        {
            string input = proposalSessionId + "\n" + principalId + "\n" +
                baseRevision.ToString(System.Globalization.CultureInfo.InvariantCulture) + "\n" +
                canonicalJson;
            return ComputeStableDigest(input);
        }

        private static string ComputeStableDigest(string input)
        {
            using (SHA256 sha256 = SHA256.Create())
            {
                byte[] hash = sha256.ComputeHash(Encoding.UTF8.GetBytes(input ?? string.Empty));
                return BitConverter.ToString(hash).Replace("-", string.Empty).ToLowerInvariant();
            }
        }

        private static void AddMutationStatus(RpcFailure failure, MutationTombstone tombstone)
        {
            if (failure?.Data == null || tombstone == null) return;
            failure.Data["mutationStatus"] = tombstone.Status.ToString().ToLowerInvariant();
            failure.Data["baseRevision"] = tombstone.BaseRevision;
            failure.Data["lastKnownRevision"] = tombstone.LastKnownRevision;
            failure.Data["digest"] = tombstone.Digest;
        }

        private void EstablishExternalMutationBarrier()
        {
            proposals.Clear();
            proposalByMutation.Clear();
            undoHistory.Clear();
        }

        private bool TryValidateMutationHeader(ScenePreviewRequest request, out RpcFailure failure)
        {
            if (request == null || !SceneProtocol.IsIdentifier(request.ClientMutationId, 128))
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    "scene/preview requires clientMutationId",
                    "missing_mutation_id");
                return false;
            }
            if (request.ExpectedRevision < 0)
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    "expectedRevision must be non-negative",
                    "invalid_revision");
                return false;
            }
            if (request.Commands == null || request.Commands.Count == 0 ||
                request.Commands.Count > SceneProtocol.MaxCommandsPerTransaction)
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    $"commands must contain 1..{SceneProtocol.MaxCommandsPerTransaction} operations",
                    "invalid_command_count");
                return false;
            }
            failure = null;
            return true;
        }

        private bool TryValidateOperation(
            SceneOperation input,
            HashSet<string> tempIds,
            out SceneOperation canonical,
            out string summary,
            out RpcFailure failure)
        {
            canonical = null;
            summary = null;
            failure = null;
            if (input == null || string.IsNullOrWhiteSpace(input.Kind))
            {
                failure = ValidationFailure("Operation kind is required", "missing_operation_kind");
                return false;
            }

            canonical = input.Clone();
            switch (canonical.Kind)
            {
                case "spawn":
                    if (HasAny(canonical.ObjectId, canonical.ComponentId) || canonical.Active.HasValue ||
                        canonical.Changes != null)
                    {
                        failure = ValidationFailure("spawn contains fields owned by another operation", "unexpected_operation_field");
                        return false;
                    }
                    if (prefabCatalog == null || string.IsNullOrWhiteSpace(canonical.PrefabId) ||
                        !prefabCatalog.TryGet(canonical.PrefabId, out RuntimePrefabEntry entry))
                    {
                        failure = ValidationFailure(
                            $"Prefab '{canonical.PrefabId}' is not in the runtime catalog",
                            "prefab_not_found");
                        return false;
                    }
                    if (!SceneProtocol.IsIdentifier(canonical.TempId, 124) ||
                        !canonical.TempId.StartsWith("tmp:", StringComparison.Ordinal) ||
                        canonical.TempId.Length == "tmp:".Length ||
                        !tempIds.Add(canonical.TempId))
                    {
                        failure = ValidationFailure(
                            "spawn requires a unique tempId beginning with 'tmp:'",
                            "invalid_temp_id");
                        return false;
                    }
                    if (canonical.ParentId != null &&
                        !SceneProtocol.IsIdentifier(canonical.ParentId, 160))
                    {
                        failure = ValidationFailure("spawn parentId is invalid", "invalid_parent_id");
                        return false;
                    }
                    if (!string.IsNullOrWhiteSpace(canonical.ParentId) &&
                        !TryResolveWritableObject(canonical.ParentId, out _, out failure))
                        return false;
                    if (canonical.LocalTransform != null &&
                        !TryValidateTransformPatch(canonical.LocalTransform, false, out failure))
                        return false;
                    summary = $"spawn {entry.DisplayName} ({canonical.TempId})";
                    return true;

                case "set_transform":
                    if (HasAny(canonical.TempId, canonical.PrefabId, canonical.ParentId, canonical.ComponentId) ||
                        canonical.Active.HasValue || canonical.Changes != null)
                    {
                        failure = ValidationFailure("set_transform contains fields owned by another operation", "unexpected_operation_field");
                        return false;
                    }
                    if (!TryResolveWritableObject(canonical.ObjectId, out _, out failure)) return false;
                    if (!TryValidateTransformPatch(canonical.LocalTransform, false, out failure)) return false;
                    summary = $"set transform on {canonical.ObjectId}";
                    return true;

                case "set_active":
                    if (HasAny(canonical.TempId, canonical.PrefabId, canonical.ParentId, canonical.ComponentId) ||
                        canonical.LocalTransform != null || canonical.Changes != null)
                    {
                        failure = ValidationFailure("set_active contains fields owned by another operation", "unexpected_operation_field");
                        return false;
                    }
                    if (!TryResolveWritableObject(canonical.ObjectId, out _, out failure)) return false;
                    if (!canonical.Active.HasValue)
                    {
                        failure = ValidationFailure("set_active requires active", "missing_active");
                        return false;
                    }
                    summary = $"set {canonical.ObjectId} active={canonical.Active.Value}";
                    return true;

                case "set_properties":
                    if (HasAny(canonical.TempId, canonical.PrefabId, canonical.ParentId) ||
                        canonical.LocalTransform != null || canonical.Active.HasValue)
                    {
                        failure = ValidationFailure("set_properties contains fields owned by another operation", "unexpected_operation_field");
                        return false;
                    }
                    if (!TryResolveWritableObject(canonical.ObjectId, out RpcObjectIdentity identity, out failure))
                        return false;
                    if (!TryResolveAdapter(identity, canonical.ComponentId, out IRpcPropertyAdapter adapter, out failure))
                        return false;
                    if (canonical.Changes == null || canonical.Changes.Count == 0 || canonical.Changes.Count > 64)
                    {
                        failure = ValidationFailure("set_properties requires 1..64 changes", "invalid_property_count");
                        return false;
                    }

                    IReadOnlyList<RpcPropertyDescriptor> descriptors = adapter.DescribeProperties();
                    var descriptorMap = descriptors == null
                        ? new Dictionary<string, RpcPropertyDescriptor>(StringComparer.Ordinal)
                        : descriptors.Where(item => item != null && !string.IsNullOrWhiteSpace(item.PropertyId))
                            .GroupBy(item => item.PropertyId, StringComparer.Ordinal)
                            .ToDictionary(group => group.Key, group => group.First(), StringComparer.Ordinal);
                    var propertyIds = new HashSet<string>(StringComparer.Ordinal);
                    foreach (RpcPropertyChange change in canonical.Changes)
                    {
                        if (change == null || !SceneProtocol.IsIdentifierSegment(change.PropertyId, 128) ||
                            !propertyIds.Add(change.PropertyId) ||
                            !descriptorMap.TryGetValue(change.PropertyId, out RpcPropertyDescriptor descriptor) ||
                            descriptor.ReadOnly)
                        {
                            failure = ValidationFailure(
                                $"Property '{change?.PropertyId}' is missing, duplicated, or read-only",
                                "property_not_writable");
                            return false;
                        }
                        if (!SceneProtocol.IsPropertyValue(change.Value))
                        {
                            failure = ValidationFailure(
                                $"Property '{change.PropertyId}' uses a value shape outside Scene RPC v1",
                                "property_value_not_supported");
                            return false;
                        }
                        if (!adapter.TryValidate(change.PropertyId, change.Value, out JToken normalized, out string error))
                        {
                            failure = ValidationFailure(
                                $"Property '{change.PropertyId}' is invalid: {error}",
                                "property_validation_failed");
                            failure.Data["propertyId"] = change.PropertyId;
                            return false;
                        }
                        if (!SceneProtocol.IsPropertyValue(normalized))
                        {
                            failure = ValidationFailure(
                                $"Property adapter returned an unsupported canonical value for '{change.PropertyId}'",
                                "adapter_contract_violation");
                            return false;
                        }
                        change.Value = normalized?.DeepClone() ?? JValue.CreateNull();
                    }
                    summary = $"set {canonical.Changes.Count} properties on {canonical.ComponentId}";
                    return true;

                default:
                    failure = ValidationFailure(
                        $"Operation '{canonical.Kind}' is not exposed by Scene RPC v1",
                        "operation_not_exposed");
                    return false;
            }
        }

        private bool TryExecuteOperation(
            SceneOperation command,
            List<IUndoAction> inverseActions,
            Dictionary<string, string> tempIdMap,
            out RpcFailure failure)
        {
            failure = null;
            switch (command.Kind)
            {
                case "spawn":
                    if (prefabCatalog == null ||
                        !prefabCatalog.TryGet(command.PrefabId, out RuntimePrefabEntry entry))
                    {
                        failure = ValidationFailure(
                            $"Prefab '{command.PrefabId}' disappeared after preview",
                            "prefab_not_found");
                        return false;
                    }
                    Transform parent = SandboxRoot;
                    if (!string.IsNullOrWhiteSpace(command.ParentId))
                    {
                        if (!TryResolveWritableObject(
                                command.ParentId,
                                out RpcObjectIdentity parentIdentity,
                                out failure))
                            return false;
                        parent = parentIdentity.transform;
                    }

                    GameObject instance;
                    try
                    {
                        instance = Instantiate(entry.Prefab, parent, false);
                    }
                    catch (Exception exception)
                    {
                        failure = RpcFailure.Create(
                            SceneErrorCodes.Internal,
                            "Prefab instantiation failed: " + exception.Message,
                            "spawn_failed");
                        return false;
                    }
                    RpcObjectIdentity identity = instance.GetComponent<RpcObjectIdentity>();
                    if (identity == null)
                    {
                        instance.SetActive(false);
                        UnityEngine.Object.Destroy(instance);
                        failure = RpcFailure.Create(
                            SceneErrorCodes.Internal,
                            "Spawned prefab has no RpcObjectIdentity on its root",
                            "spawn_identity_missing");
                        return false;
                    }
                    string objectId;
                    do objectId = Guid.NewGuid().ToString("N");
                    while (objects.ContainsKey(objectId));
                    identity.AssignRuntimeIdentity(objectId, command.PrefabId);
                    if (!RegisterSpawned(identity, out string registrationError))
                    {
                        instance.SetActive(false);
                        Destroy(instance);
                        failure = RpcFailure.Create(
                            SceneErrorCodes.Internal,
                            registrationError,
                            "spawn_registration_failed");
                        return false;
                    }
                    inverseActions.Add(new SpawnUndoAction(this, identity));
                    ApplyTransformPatch(identity.transform, command.LocalTransform);
                    tempIdMap.Add(command.TempId, objectId);
                    return true;

                case "set_transform":
                    if (!TryResolveWritableObject(
                            command.ObjectId,
                            out RpcObjectIdentity transformIdentity,
                            out failure))
                        return false;
                    inverseActions.Add(new TransformUndoAction(transformIdentity.transform));
                    ApplyTransformPatch(transformIdentity.transform, command.LocalTransform);
                    return true;

                case "set_active":
                    if (!TryResolveWritableObject(
                            command.ObjectId,
                            out RpcObjectIdentity activeIdentity,
                            out failure))
                        return false;
                    inverseActions.Add(new ActiveUndoAction(activeIdentity.gameObject));
                    activeIdentity.gameObject.SetActive(command.Active.Value);
                    return true;

                case "set_properties":
                    if (!TryResolveWritableObject(
                            command.ObjectId,
                            out RpcObjectIdentity propertyIdentity,
                            out failure))
                        return false;
                    if (!TryResolveAdapter(propertyIdentity, command.ComponentId, out IRpcPropertyAdapter adapter, out failure))
                        return false;
                    var previous = new List<RpcPropertyChange>(command.Changes.Count);
                    foreach (RpcPropertyChange change in command.Changes)
                    {
                        if (!adapter.TryRead(change.PropertyId, out JToken value, out string readError))
                        {
                            failure = ValidationFailure(
                                $"Could not capture '{change.PropertyId}' for undo: {readError}",
                                "property_read_failed");
                            return false;
                        }
                        previous.Add(new RpcPropertyChange
                        {
                            PropertyId = change.PropertyId,
                            Value = value?.DeepClone() ?? JValue.CreateNull(),
                        });
                    }
                    inverseActions.Add(new PropertyUndoAction(adapter, previous));
                    foreach (RpcPropertyChange change in command.Changes)
                    {
                        // Validate again immediately before each write. Earlier writes
                        // in this same operation are now visible, so state-dependent
                        // cross-property invariants are checked against projected state.
                        if (!adapter.TryValidate(
                                change.PropertyId,
                                change.Value,
                                out JToken canonical,
                                out string validationError))
                        {
                            failure = ValidationFailure(
                                $"Property '{change.PropertyId}' changed validity after preview: {validationError}",
                                "property_revalidation_failed");
                            return false;
                        }
                        JToken canonicalValue = canonical ?? JValue.CreateNull();
                        if (!JToken.DeepEquals(canonicalValue, change.Value))
                        {
                            failure = ValidationFailure(
                                $"Property '{change.PropertyId}' canonical value changed after preview",
                                "property_canonicalization_changed");
                            return false;
                        }
                        if (!adapter.TryWrite(change.PropertyId, change.Value, out string writeError))
                        {
                            failure = ValidationFailure(
                                $"Could not write '{change.PropertyId}': {writeError}",
                                "property_write_failed");
                            return false;
                        }
                    }
                    return true;

                default:
                    failure = ValidationFailure("Unsupported operation reached executor", "executor_mismatch");
                    return false;
            }
        }

        private bool TryResolveWritableObject(
            string objectId,
            out RpcObjectIdentity identity,
            out RpcFailure failure)
        {
            if (!TryResolveObject(objectId, out identity, out failure)) return false;
            if (identity.AllowRemoteChanges) return true;

            failure = RpcFailure.Create(
                SceneErrorCodes.ValidationFailed,
                $"Runtime object '{objectId}' does not allow remote changes",
                "object_read_only");
            failure.Data["objectId"] = objectId;
            return false;
        }

        private bool TryCheckRevision(long expected, out RpcFailure failure)
        {
            if (expected < 0)
            {
                failure = RpcFailure.Create(
                    SceneErrorCodes.InvalidParams,
                    "expectedRevision must be non-negative",
                    "invalid_revision");
                return false;
            }
            if (expected == sceneRevision)
            {
                failure = null;
                return true;
            }
            failure = RpcFailure.Create(
                SceneErrorCodes.RevisionConflict,
                $"Scene revision conflict: expected {expected}, actual {sceneRevision}",
                "revision_conflict",
                true);
            failure.Data["expectedRevision"] = expected;
            failure.Data["actualRevision"] = sceneRevision;
            return false;
        }

        private static bool TryValidateTransformPatch(
            RpcTransformPatch patch,
            bool allowEmpty,
            out RpcFailure failure)
        {
            if (patch == null || patch.IsEmpty)
            {
                if (allowEmpty)
                {
                    failure = null;
                    return true;
                }
                failure = ValidationFailure("localTransform must change at least one field", "empty_transform");
                return false;
            }
            foreach (RpcVector3 vector in new[] { patch.Position, patch.RotationEuler, patch.Scale })
            {
                if (vector != null && (!IsFinite(vector.X) || !IsFinite(vector.Y) || !IsFinite(vector.Z)))
                {
                    failure = ValidationFailure("Transform values must be finite", "non_finite_transform");
                    return false;
                }
            }
            failure = null;
            return true;
        }

        private static void ApplyTransformPatch(Transform target, RpcTransformPatch patch)
        {
            if (patch == null) return;
            if (patch.Position != null)
                target.localPosition = new Vector3(patch.Position.X, patch.Position.Y, patch.Position.Z);
            if (patch.RotationEuler != null)
                target.localEulerAngles = new Vector3(
                    patch.RotationEuler.X,
                    patch.RotationEuler.Y,
                    patch.RotationEuler.Z);
            if (patch.Scale != null)
                target.localScale = new Vector3(patch.Scale.X, patch.Scale.Y, patch.Scale.Z);
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        private static bool HasAny(params string[] values)
        {
            return values.Any(value => value != null);
        }

        private static RpcFailure ValidationFailure(string message, string reason)
        {
            return RpcFailure.Create(SceneErrorCodes.ValidationFailed, message, reason);
        }

        private bool TryRollback(List<IUndoAction> actions, out string error)
        {
            var failures = new List<string>();
            for (int index = actions.Count - 1; index >= 0; index--)
            {
                if (!TryRevertSafely(actions[index], out string rollbackError))
                    failures.Add($"action {index}: {rollbackError}");
            }
            error = failures.Count == 0 ? null : string.Join("; ", failures);
            return failures.Count == 0;
        }

        private static bool TryCanRevertSafely(IUndoAction action, out string error)
        {
            if (action == null)
            {
                error = "Undo action is missing";
                return false;
            }
            try
            {
                return action.CanRevert(out error);
            }
            catch (Exception exception)
            {
                error = "Undo preflight threw " + exception.GetType().Name + ": " + exception.Message;
                return false;
            }
        }

        private static bool TryRevertSafely(IUndoAction action, out string error)
        {
            if (action == null)
            {
                error = "Undo action is missing";
                return false;
            }
            try
            {
                return action.Revert(out error);
            }
            catch (Exception exception)
            {
                error = "Undo action threw " + exception.GetType().Name + ": " + exception.Message;
                return false;
            }
        }

        private void CleanupExpiredProposals()
        {
            long now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            List<PendingProposal> expired = proposals.Values
                .Where(proposal => proposal.ExpiresAtUnixMs < now)
                .ToList();
            foreach (PendingProposal proposal in expired)
                RemoveProposal(proposal);
        }

        private void RemoveProposal(PendingProposal proposal)
        {
            if (proposal == null) return;
            proposals.Remove(proposal.Token);
            if (proposalByMutation.TryGetValue(proposal.MutationKey, out string token) &&
                string.Equals(token, proposal.Token, StringComparison.Ordinal))
                proposalByMutation.Remove(proposal.MutationKey);
        }

        private void RememberReceipt(string mutationKey, SceneMutationResult result)
        {
            mutationReceipts[mutationKey] = CloneReceipt(result);
            receiptOrder.Enqueue(mutationKey);
            while (receiptOrder.Count > MaxReceiptEntries)
            {
                string oldest = receiptOrder.Dequeue();
                mutationReceipts.Remove(oldest);
            }
        }

        private static SceneMutationResult CloneReceipt(SceneMutationResult result)
        {
            return new SceneMutationResult
            {
                SceneRevision = result.SceneRevision,
                ClientMutationId = result.ClientMutationId,
                UndoId = result.UndoId,
                TempIdMap = result.TempIdMap == null
                    ? new Dictionary<string, string>()
                    : new Dictionary<string, string>(result.TempIdMap, StringComparer.Ordinal),
                IdempotentReplay = result.IdempotentReplay,
            };
        }

        private sealed class PendingProposal
        {
            public string Token;
            public string SessionId;
            public string PrincipalId;
            public string MutationKey;
            public string ClientMutationId;
            public long BaseRevision;
            public long ExpiresAtUnixMs;
            public string CanonicalJson;
            public string Digest;
            public List<SceneOperation> Commands;
            public List<string> Summary;

            public ScenePreviewResult ToResult()
            {
                return new ScenePreviewResult
                {
                    PreviewToken = Token,
                    ClientMutationId = ClientMutationId,
                    BaseRevision = BaseRevision,
                    ExpiresAtUnixMs = ExpiresAtUnixMs,
                    Summary = new List<string>(Summary),
                };
            }
        }

        private enum MutationStatus
        {
            Applying,
            Applied,
            Rejected,
            RolledBack,
            Indeterminate,
            Undone,
        }

        private sealed class MutationTombstone
        {
            public string SessionId;
            public string PrincipalId;
            public string ClientMutationId;
            public long BaseRevision;
            public string Digest;
            public string PreviewToken;
            public MutationStatus Status;
            public long LastKnownRevision;
        }

        private sealed class UndoRecord
        {
            public string UndoId;
            public string MutationKey;
            public string OwnerPrincipalId;
            public string ClientMutationId;
            public List<IUndoAction> Actions;
        }

        private interface IUndoAction
        {
            bool CanRevert(out string error);
            bool Revert(out string error);
        }

        private sealed class TransformUndoAction : IUndoAction
        {
            private readonly Transform target;
            private readonly Vector3 position;
            private readonly Vector3 rotation;
            private readonly Vector3 scale;

            public TransformUndoAction(Transform target)
            {
                this.target = target;
                position = target.localPosition;
                rotation = target.localEulerAngles;
                scale = target.localScale;
            }

            public bool CanRevert(out string error)
            {
                error = target == null ? "Transform target no longer exists" : null;
                return target != null;
            }

            public bool Revert(out string error)
            {
                if (!CanRevert(out error)) return false;
                target.localPosition = position;
                target.localEulerAngles = rotation;
                target.localScale = scale;
                return true;
            }
        }

        private sealed class ActiveUndoAction : IUndoAction
        {
            private readonly GameObject target;
            private readonly bool previous;

            public ActiveUndoAction(GameObject target)
            {
                this.target = target;
                previous = target.activeSelf;
            }

            public bool CanRevert(out string error)
            {
                error = target == null ? "Active-state target no longer exists" : null;
                return target != null;
            }

            public bool Revert(out string error)
            {
                if (!CanRevert(out error)) return false;
                target.SetActive(previous);
                return true;
            }
        }

        private sealed class PropertyUndoAction : IUndoAction
        {
            private readonly IRpcPropertyAdapter adapter;
            private readonly List<RpcPropertyChange> previous;

            public PropertyUndoAction(IRpcPropertyAdapter adapter, List<RpcPropertyChange> previous)
            {
                this.adapter = adapter;
                this.previous = previous;
            }

            public bool CanRevert(out string error)
            {
                if (adapter is UnityEngine.Object unityObject && unityObject == null)
                {
                    error = "Property adapter no longer exists";
                    return false;
                }
                foreach (RpcPropertyChange change in previous)
                {
                    if (!adapter.TryValidate(change.PropertyId, change.Value, out _, out error))
                        return false;
                }
                error = null;
                return true;
            }

            public bool Revert(out string error)
            {
                foreach (RpcPropertyChange change in previous)
                {
                    if (!adapter.TryValidate(change.PropertyId, change.Value, out JToken canonical, out error) ||
                        !adapter.TryWrite(change.PropertyId, canonical, out error))
                        return false;
                }
                error = null;
                return true;
            }
        }

        private sealed class SpawnUndoAction : IUndoAction
        {
            private readonly RuntimeSceneController controller;
            private readonly RpcObjectIdentity identity;

            public SpawnUndoAction(RuntimeSceneController controller, RpcObjectIdentity identity)
            {
                this.controller = controller;
                this.identity = identity;
            }

            public bool CanRevert(out string error)
            {
                error = identity == null ? "Spawned object no longer exists" : null;
                return identity != null;
            }

            public bool Revert(out string error)
            {
                if (!CanRevert(out error)) return false;
                controller.Unregister(identity);
                identity.gameObject.SetActive(false);
                UnityEngine.Object.Destroy(identity.gameObject);
                return true;
            }
        }
    }
}
