using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace BrainRegion.RuntimeBridge
{
    public static class SceneProtocol
    {
        public const string Version = "brainregion.scene.v1";
        public const int MaxCommandsPerTransaction = 64;
        public const int MaxPageSize = 500;

        public static bool IsIdentifier(string value, int maximum)
        {
            if (string.IsNullOrEmpty(value) || value.Length > maximum) return false;
            foreach (char character in value)
            {
                bool allowed = character <= 127 &&
                    (char.IsLetterOrDigit(character) || character == '.' || character == '_' ||
                     character == ':' || character == '/' || character == '-');
                if (!allowed) return false;
            }
            return true;
        }

        public static bool IsIdentifierSegment(string value, int maximum)
        {
            return IsIdentifier(value, maximum) && value.IndexOf('/') < 0 && value.IndexOf(':') < 0;
        }

        public static bool IsPropertyValue(JToken token, bool arrayItem = false)
        {
            if (token == null || token.Type == JTokenType.Null || token.Type == JTokenType.Boolean ||
                token.Type == JTokenType.Integer) return true;
            if (token.Type == JTokenType.Float)
            {
                double number = token.Value<double>();
                return !double.IsNaN(number) && !double.IsInfinity(number);
            }
            if (token.Type == JTokenType.String)
                return token.Value<string>().Length <= (arrayItem ? 4096 : 16384);
            if (token is JObject valueObject)
            {
                if (IsVectorObject(valueObject)) return true;
                if (arrayItem || valueObject.Count != 1) return false;
                if (valueObject.TryGetValue("objectRef", out JToken objectRef))
                    return objectRef.Type == JTokenType.String && IsIdentifier(objectRef.Value<string>(), 160);
                if (valueObject.TryGetValue("assetRef", out JToken assetRef))
                    return assetRef.Type == JTokenType.String &&
                           !string.IsNullOrEmpty(assetRef.Value<string>()) &&
                           assetRef.Value<string>().Length <= 256;
                return false;
            }
            if (token is JArray array)
                return !arrayItem && array.Count <= 64 && array.All(item => IsPropertyValue(item, true));
            return false;
        }

        private static bool IsVectorObject(JObject value)
        {
            if (value.Count != 3 || value["x"] == null || value["y"] == null || value["z"] == null)
                return false;
            foreach (string axis in new[] { "x", "y", "z" })
            {
                JToken component = value[axis];
                if (component.Type != JTokenType.Integer && component.Type != JTokenType.Float) return false;
                double number = component.Value<double>();
                if (double.IsNaN(number) || double.IsInfinity(number)) return false;
            }
            return true;
        }
    }

    /// <summary>
    /// Capability names understood by the Player-side dispatcher. Transports may
    /// only construct a peer context after authenticating the connection; the
    /// dispatcher treats every capability as case-sensitive and deny-by-default.
    /// </summary>
    public static class SceneCapabilities
    {
        public const string SceneRead = "scene.read";
        public const string SceneWrite = "scene.write";
        public const string SceneSpawn = "scene.spawn";
        public const string SceneUndo = "scene.undo";
        public const string LogsRead = "logs.read";

        public static bool IsKnown(string capability)
        {
            switch (capability)
            {
                case SceneRead:
                case SceneWrite:
                case SceneSpawn:
                case SceneUndo:
                case LogsRead:
                    return true;
                default:
                    return false;
            }
        }
    }

    /// <summary>
    /// Immutable authorization evidence supplied by an authenticated transport.
    /// ConnectionEpoch must increase whenever one principal reconnects; enqueuing
    /// a newer epoch invalidates requests still queued for an older connection.
    /// This type does not authenticate credentials by itself.
    /// </summary>
    public sealed class AuthenticatedPeerContext
    {
        private readonly HashSet<string> capabilitySet;
        private readonly ReadOnlyCollection<string> grantedCapabilities;

        public string PrincipalId { get; }
        public long ConnectionEpoch { get; }
        public IReadOnlyList<string> GrantedCapabilities => grantedCapabilities;

        public AuthenticatedPeerContext(
            string principalId,
            long connectionEpoch,
            IEnumerable<string> capabilities)
        {
            if (!SceneProtocol.IsIdentifier(principalId, 128))
                throw new ArgumentException("principalId is not a valid Scene RPC identifier", nameof(principalId));
            if (connectionEpoch <= 0)
                throw new ArgumentOutOfRangeException(nameof(connectionEpoch), "connectionEpoch must be positive");

            PrincipalId = principalId;
            ConnectionEpoch = connectionEpoch;
            capabilitySet = new HashSet<string>(StringComparer.Ordinal);
            if (capabilities != null)
            {
                foreach (string capability in capabilities)
                {
                    if (!SceneCapabilities.IsKnown(capability))
                        throw new ArgumentException($"Unknown Scene RPC capability '{capability}'", nameof(capabilities));
                    capabilitySet.Add(capability);
                }
            }

            var ordered = new List<string>(capabilitySet);
            ordered.Sort(StringComparer.Ordinal);
            grantedCapabilities = ordered.AsReadOnly();
        }

        public bool HasCapability(string capability)
        {
            return capability != null && capabilitySet.Contains(capability);
        }
    }

    [Serializable]
    public sealed class RpcVector3
    {
        [JsonProperty("x", Required = Required.Always)] public float X;
        [JsonProperty("y", Required = Required.Always)] public float Y;
        [JsonProperty("z", Required = Required.Always)] public float Z;
    }

    [Serializable]
    public sealed class RpcTransformPatch
    {
        [JsonProperty("position")] public RpcVector3 Position;
        [JsonProperty("rotationEuler")] public RpcVector3 RotationEuler;
        [JsonProperty("scale")] public RpcVector3 Scale;

        [JsonIgnore]
        public bool IsEmpty => Position == null && RotationEuler == null && Scale == null;
    }

    [Serializable]
    public sealed class RpcPropertyChange
    {
        [JsonProperty("propertyId", Required = Required.Always)] public string PropertyId;
        [JsonProperty("value", Required = Required.AllowNull)] public JToken Value;
    }

    [Serializable]
    public sealed class SceneOperation
    {
        [JsonProperty("kind", Required = Required.Always)] public string Kind;
        [JsonProperty("tempId")] public string TempId;
        [JsonProperty("prefabId")] public string PrefabId;
        [JsonProperty("objectId")] public string ObjectId;
        [JsonProperty("parentId")] public string ParentId;
        [JsonProperty("componentId")] public string ComponentId;
        [JsonProperty("localTransform")] public RpcTransformPatch LocalTransform;
        [JsonProperty("active")] public bool? Active;
        [JsonProperty("changes")] public List<RpcPropertyChange> Changes;

        public SceneOperation Clone()
        {
            return JObject.FromObject(this).ToObject<SceneOperation>();
        }
    }

    [Serializable]
    public sealed class ScenePreviewRequest
    {
        [JsonProperty("expectedRevision", Required = Required.Always)] public long ExpectedRevision;
        [JsonProperty("clientMutationId", Required = Required.Always)] public string ClientMutationId;
        [JsonProperty("commands", Required = Required.Always)] public List<SceneOperation> Commands;
    }

    [Serializable]
    public sealed class SceneApplyRequest
    {
        [JsonProperty("previewToken", Required = Required.Always)] public string PreviewToken;
        [JsonProperty("expectedRevision", Required = Required.Always)] public long ExpectedRevision;
        [JsonProperty("clientMutationId", Required = Required.Always)] public string ClientMutationId;
    }

    [Serializable]
    public sealed class SceneUndoRequest
    {
        [JsonProperty("expectedRevision", Required = Required.Always)] public long ExpectedRevision;
        [JsonProperty("undoId")] public string UndoId;
    }

    [Serializable]
    public sealed class SceneHierarchyRequest
    {
        [JsonProperty("rootId")] public string RootId;
        [JsonProperty("depth")] public int Depth = 4;
        [JsonProperty("includeInactive")] public bool IncludeInactive = true;
        [JsonProperty("cursor")] public string Cursor;
        [JsonProperty("limit")] public int Limit = 200;
        [JsonProperty("ifRevision")] public long? IfRevision;
    }

    [Serializable]
    public sealed class ObjectInspectRequest
    {
        [JsonProperty("objectId", Required = Required.Always)] public string ObjectId;
        [JsonProperty("componentIds")] public List<string> ComponentIds;
    }

    [Serializable]
    public sealed class PrefabListRequest
    {
        [JsonProperty("query")] public string Query;
        [JsonProperty("cursor")] public string Cursor;
        [JsonProperty("limit")] public int Limit = 200;
    }

    [Serializable]
    public sealed class LogsPollRequest
    {
        [JsonProperty("afterSeq")] public long AfterSeq;
        [JsonProperty("limit")] public int Limit = 200;
        [JsonProperty("levels")] public List<string> Levels;
    }

    [Serializable]
    public sealed class RpcPropertyDescriptor
    {
        [JsonProperty("propertyId")] public string PropertyId;
        [JsonProperty("displayName")] public string DisplayName;
        [JsonProperty("valueType")] public string ValueType;
        [JsonProperty("readOnly")] public bool ReadOnly;
        [JsonProperty("persistent")] public bool Persistent = true;
        [JsonProperty("minimum")] public double? Minimum;
        [JsonProperty("maximum")] public double? Maximum;
        [JsonProperty("enumValues")] public List<string> EnumValues;
    }

    [Serializable]
    public sealed class RpcPropertySnapshot
    {
        [JsonProperty("descriptor")] public RpcPropertyDescriptor Descriptor;
        [JsonProperty("value")] public JToken Value;
    }

    [Serializable]
    public sealed class RpcComponentSnapshot
    {
        [JsonProperty("componentId")] public string ComponentId;
        [JsonProperty("typeId")] public string TypeId;
        [JsonProperty("properties")] public List<RpcPropertySnapshot> Properties = new List<RpcPropertySnapshot>();
    }

    [Serializable]
    public sealed class RpcObjectSnapshot
    {
        [JsonProperty("objectId")] public string ObjectId;
        [JsonProperty("parentId")] public string ParentId;
        [JsonProperty("name")] public string Name;
        [JsonProperty("active")] public bool Active;
        [JsonProperty("layer")] public int Layer;
        [JsonProperty("tag")] public string Tag;
        [JsonProperty("prefabId")] public string PrefabId;
        [JsonProperty("childCount")] public int ChildCount;
        [JsonProperty("components")] public List<RpcComponentSnapshot> Components = new List<RpcComponentSnapshot>();
    }

    [Serializable]
    public sealed class SceneHierarchyResult
    {
        [JsonProperty("sceneRevision")] public long SceneRevision;
        [JsonProperty("nodes")] public List<RpcObjectSnapshot> Nodes = new List<RpcObjectSnapshot>();
        [JsonProperty("nextCursor")] public string NextCursor;
        [JsonProperty("notModified")] public bool NotModified;
    }

    [Serializable]
    public sealed class ScenePreviewResult
    {
        [JsonProperty("previewToken")] public string PreviewToken;
        [JsonProperty("baseRevision")] public long BaseRevision;
        [JsonProperty("expiresAtUnixMs")] public long ExpiresAtUnixMs;
        [JsonProperty("clientMutationId")] public string ClientMutationId;
        [JsonProperty("summary")] public List<string> Summary = new List<string>();
        [JsonProperty("warnings")] public List<string> Warnings = new List<string>();
    }

    [Serializable]
    public sealed class SceneMutationResult
    {
        [JsonProperty("sceneRevision")] public long SceneRevision;
        [JsonProperty("clientMutationId")] public string ClientMutationId;
        [JsonProperty("undoId")] public string UndoId;
        [JsonProperty("tempIdMap")] public Dictionary<string, string> TempIdMap = new Dictionary<string, string>();
        [JsonProperty("idempotentReplay")] public bool IdempotentReplay;
    }

    [Serializable]
    public sealed class RpcFailure
    {
        [JsonProperty("code")] public int Code;
        [JsonProperty("message")] public string Message;
        [JsonProperty("data")] public JObject Data;

        public static RpcFailure Create(int code, string message, string reason, bool retryable = false)
        {
            return new RpcFailure
            {
                Code = code,
                Message = message,
                Data = new JObject
                {
                    ["reason"] = reason,
                    ["retryable"] = retryable,
                },
            };
        }
    }

    public static class SceneErrorCodes
    {
        public const int InvalidRequest = -32600;
        public const int MethodNotFound = -32601;
        public const int InvalidParams = -32602;
        public const int Internal = -32603;
        public const int Unauthenticated = -32001;
        public const int Forbidden = -32002;
        public const int Busy = -32003;
        public const int RevisionConflict = -32010;
        public const int ObjectNotFound = -32011;
        public const int ValidationFailed = -32013;
        public const int PropertyNotExposed = -32014;
        public const int PreviewExpired = -32015;
        public const int NotReversible = -32016;
    }
}
