using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace BrainRegion.RuntimeBridge
{
    /// <summary>
    /// Pure managed WorldDocument storage. Callers may execute these methods on a
    /// worker thread after resolving persistentDataPath on the Unity main thread.
    /// This type never touches UnityEngine or mutable scene state.
    /// </summary>
    internal static class WorldDocumentStorage
    {
        internal const int MaxSlots = 32;
        internal const int MaxFileBytes = 1024 * 1024;
        internal const long MaxStorageBytes = 16L * 1024L * 1024L;
        private const string Extension = ".brworld.json";

        internal static string GetStorageRoot(string persistentDataPath)
        {
            if (string.IsNullOrWhiteSpace(persistentDataPath))
                throw new ArgumentException("persistentDataPath must not be empty", nameof(persistentDataPath));
            return Path.GetFullPath(Path.Combine(persistentDataPath, "BrainRegion", "Worlds"));
        }

        internal static bool TryList(
            string root,
            out JObject result,
            out RpcFailure failure)
        {
            result = null;
            failure = null;
            try
            {
                root = NormalizeRoot(root);
                var slots = new JArray();
                var corruptSlots = new JArray();
                if (Directory.Exists(root))
                {
                    string[] paths = EnumerateBoundedSlotFiles(root, out failure);
                    if (failure != null) return false;
                    foreach (string path in paths)
                    {
                        string fileName = Path.GetFileName(path);
                        string slot = fileName.Substring(0, fileName.Length - Extension.Length);
                        if (!TryRead(root, slot, out JObject envelope, out RpcFailure readFailure))
                        {
                            corruptSlots.Add(new JObject
                            {
                                ["slot"] = slot,
                                ["reason"] = (string)readFailure.Data?["reason"] ?? "invalid_world_document",
                            });
                            continue;
                        }
                        JObject document = (JObject)envelope["document"];
                        if (!TryGetSummary(
                                document,
                                out JToken savedRevision,
                                out JToken savedUnixMs,
                                out JToken label))
                        {
                            corruptSlots.Add(new JObject
                            {
                                ["slot"] = slot,
                                ["reason"] = "invalid_world_document",
                            });
                            continue;
                        }
                        slots.Add(new JObject
                        {
                            ["slot"] = slot,
                            ["digest"] = envelope["digest"].DeepClone(),
                            ["savedRevision"] = savedRevision,
                            ["savedUnixMs"] = savedUnixMs,
                            ["label"] = label,
                            ["bytes"] = new FileInfo(path).Length,
                        });
                    }
                }
                result = new JObject
                {
                    ["slots"] = slots,
                    ["corruptSlots"] = corruptSlots,
                    ["maximumSlots"] = MaxSlots,
                    ["maximumFileBytes"] = MaxFileBytes,
                    ["maximumStorageBytes"] = MaxStorageBytes,
                };
                return true;
            }
            catch (Exception exception)
            {
                failure = Failure(
                    "Could not enumerate WorldDocument slots: " + exception.Message,
                    "slot_list_failed",
                    true);
                return false;
            }
        }

        internal static bool TryWrite(
            string root,
            string slot,
            string expectedSlotDigest,
            JObject document,
            out JObject result,
            out RpcFailure failure)
        {
            result = null;
            failure = null;
            if (!IsSlot(slot) || document == null ||
                (!string.IsNullOrEmpty(expectedSlotDigest) && !IsDigest(expectedSlotDigest)))
            {
                failure = Failure("WorldDocument write input is invalid", "invalid_persistence_params");
                return false;
            }
            if (!TryGetSummary(
                    document,
                    out JToken savedRevision,
                    out _,
                    out _))
            {
                failure = Failure(
                    "WorldDocument summary fields are invalid",
                    "invalid_world_document");
                return false;
            }

            try
            {
                root = NormalizeRoot(root);
                string digest = ComputeDigest(document);
                var envelope = new JObject
                {
                    ["digest"] = digest,
                    ["document"] = document,
                };
                byte[] bytes = new UTF8Encoding(false, true).GetBytes(envelope.ToString(Formatting.None));
                if (bytes.Length > MaxFileBytes)
                {
                    failure = Failure(
                        $"WorldDocument requires {bytes.Length} bytes; limit is {MaxFileBytes}",
                        "world_document_too_large");
                    return false;
                }

                Directory.CreateDirectory(root);
                string target = GetSlotPath(root, slot);
                string[] existing = EnumerateBoundedSlotFiles(root, out failure);
                if (failure != null) return false;
                if (!File.Exists(target) && existing.Length >= MaxSlots)
                {
                    failure = Failure(
                        $"WorldDocument slot quota of {MaxSlots} is full",
                        "slot_quota_exceeded");
                    return false;
                }

                string currentDigest = null;
                long currentBytes = 0;
                if (File.Exists(target))
                {
                    currentBytes = new FileInfo(target).Length;
                    if (!TryRead(root, slot, out JObject current, out failure))
                        return false;
                    currentDigest = (string)current["digest"];
                }
                if (!string.IsNullOrEmpty(expectedSlotDigest) &&
                    !string.Equals(currentDigest, expectedSlotDigest, StringComparison.Ordinal))
                {
                    failure = Failure(
                        "WorldDocument slot digest changed before save",
                        "slot_digest_conflict",
                        true);
                    failure.Data["expectedSlotDigest"] = expectedSlotDigest;
                    failure.Data["actualSlotDigest"] = currentDigest == null
                        ? JValue.CreateNull()
                        : new JValue(currentDigest);
                    return false;
                }

                long totalBytes = existing.Sum(path => new FileInfo(path).Length) - currentBytes + bytes.Length;
                if (totalBytes > MaxStorageBytes)
                {
                    failure = Failure(
                        $"WorldDocument storage would exceed {MaxStorageBytes} bytes",
                        "storage_quota_exceeded");
                    return false;
                }

                AtomicWrite(target, bytes);
                result = new JObject
                {
                    ["slot"] = slot,
                    ["digest"] = digest,
                    ["savedRevision"] = savedRevision,
                    ["bytes"] = bytes.Length,
                    ["idempotentReplay"] = false,
                };
                return true;
            }
            catch (Exception exception)
            {
                failure = Failure(
                    "WorldDocument save failed: " + exception.Message,
                    "save_failed",
                    true);
                return false;
            }
        }

        internal static bool TryRead(
            string root,
            string slot,
            out JObject envelope,
            out RpcFailure failure)
        {
            envelope = null;
            failure = null;
            if (!IsSlot(slot))
            {
                failure = Failure("WorldDocument slot is invalid", "invalid_slot");
                return false;
            }
            try
            {
                root = NormalizeRoot(root);
                string path = GetSlotPath(root, slot);
                if (!File.Exists(path))
                {
                    failure = Failure(
                        $"WorldDocument slot '{slot}' does not exist",
                        "slot_not_found");
                    return false;
                }
                var info = new FileInfo(path);
                if (info.Length < 1 || info.Length > MaxFileBytes)
                {
                    failure = Failure(
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
                    !string.Equals((string)envelope["digest"], ComputeDigest(document), StringComparison.Ordinal))
                {
                    failure = Failure(
                        "WorldDocument digest or envelope is invalid",
                        "world_digest_mismatch");
                    envelope = null;
                    return false;
                }
                return true;
            }
            catch (Exception exception)
            {
                failure = Failure(
                    "WorldDocument could not be read: " + exception.Message,
                    "world_read_failed");
                envelope = null;
                return false;
            }
        }

        internal static bool IsSlot(string slot)
        {
            if (string.IsNullOrEmpty(slot) || slot.Length > 64 ||
                !char.IsLetterOrDigit(slot[0])) return false;
            return slot.All(character => character <= 127 &&
                (char.IsLetterOrDigit(character) || character == '_' || character == '-'));
        }

        internal static bool IsDigest(string digest)
        {
            if (digest == null || !digest.StartsWith("sha256:", StringComparison.Ordinal) ||
                digest.Length != "sha256:".Length + 64) return false;
            return digest.Skip("sha256:".Length)
                .All(character => character >= '0' && character <= '9' ||
                                  character >= 'a' && character <= 'f');
        }

        internal static string ComputeDigest(JObject document)
        {
            using (SHA256 sha256 = SHA256.Create())
            {
                byte[] bytes = Encoding.UTF8.GetBytes(document.ToString(Formatting.None));
                byte[] digest = sha256.ComputeHash(bytes);
                return "sha256:" + BitConverter.ToString(digest).Replace("-", string.Empty).ToLowerInvariant();
            }
        }

        private static string NormalizeRoot(string root)
        {
            if (string.IsNullOrWhiteSpace(root))
                throw new ArgumentException("WorldDocument storage root must not be empty", nameof(root));
            return Path.GetFullPath(root);
        }

        private static string GetSlotPath(string root, string slot)
        {
            if (!IsSlot(slot)) throw new ArgumentException("Invalid WorldDocument slot", nameof(slot));
            root = NormalizeRoot(root);
            string path = Path.GetFullPath(Path.Combine(root, slot + Extension));
            string prefix = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) +
                Path.DirectorySeparatorChar;
            if (!path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("WorldDocument slot escaped its storage root");
            return path;
        }

        private static void AtomicWrite(string target, byte[] bytes)
        {
            string temporary = target + ".tmp." + Guid.NewGuid().ToString("N");
            string backup = target + ".bak";
            try
            {
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
                        // The target is committed. A stale backup is safer than a
                        // false failure that could trigger an ambiguous retry.
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

        private static bool HasExactProperties(JObject value, params string[] names)
        {
            return value != null && value.Count == names.Length &&
                   names.All(name => value.Property(name) != null);
        }

        private static string[] EnumerateBoundedSlotFiles(
            string root,
            out RpcFailure failure)
        {
            failure = null;
            string[] paths = Directory.EnumerateFiles(root, "*" + Extension)
                .Take(MaxSlots + 1)
                .ToArray();
            if (paths.Length > MaxSlots)
            {
                failure = Failure(
                    $"WorldDocument storage contains more than {MaxSlots} slot files",
                    "slot_list_overflow");
                return Array.Empty<string>();
            }
            Array.Sort(paths, StringComparer.Ordinal);
            return paths;
        }

        private static bool TryGetSummary(
            JObject document,
            out JToken savedRevision,
            out JToken savedUnixMs,
            out JToken label)
        {
            savedRevision = null;
            savedUnixMs = null;
            label = null;
            if (document?["savedRevision"]?.Type != JTokenType.Integer ||
                document["savedUnixMs"]?.Type != JTokenType.Integer ||
                document["metadata"] is not JObject metadata ||
                metadata["label"] != null &&
                metadata["label"].Type != JTokenType.String &&
                metadata["label"].Type != JTokenType.Null)
                return false;

            savedRevision = document["savedRevision"].DeepClone();
            savedUnixMs = document["savedUnixMs"].DeepClone();
            label = metadata["label"]?.DeepClone() ?? JValue.CreateNull();
            return true;
        }

        private static RpcFailure Failure(string message, string reason, bool retryable = false)
        {
            return RpcFailure.Create(SceneErrorCodes.PersistenceError, message, reason, retryable);
        }
    }
}
