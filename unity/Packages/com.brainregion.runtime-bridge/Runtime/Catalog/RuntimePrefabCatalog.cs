using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace BrainRegion.RuntimeBridge
{
    [Serializable]
    public sealed class RuntimePrefabEntry
    {
        [SerializeField] private string prefabId;
        [SerializeField] private string displayName;
        [SerializeField] private GameObject prefab;
        [SerializeField] private List<string> tags = new List<string>();

        public string PrefabId => prefabId;
        public string DisplayName => string.IsNullOrWhiteSpace(displayName) && prefab != null ? prefab.name : displayName;
        public GameObject Prefab => prefab;
        public IReadOnlyList<string> Tags => tags;
    }

    [CreateAssetMenu(fileName = "BrainRegionPrefabCatalog", menuName = "BrainRegion/Runtime Prefab Catalog")]
    public sealed class RuntimePrefabCatalog : ScriptableObject
    {
        [SerializeField] private string schemaVersion = "1";
        [SerializeField] private List<RuntimePrefabEntry> entries = new List<RuntimePrefabEntry>();

        public string SchemaVersion => schemaVersion;
        public IReadOnlyList<RuntimePrefabEntry> Entries => entries;

        public bool TryGet(string prefabId, out RuntimePrefabEntry entry)
        {
            entry = entries.FirstOrDefault(candidate =>
                candidate != null && string.Equals(candidate.PrefabId, prefabId, StringComparison.Ordinal));
            return entry != null && entry.Prefab != null;
        }

        public bool TryValidate(out string error)
        {
            if (string.IsNullOrWhiteSpace(schemaVersion) || schemaVersion.Length > 128)
            {
                error = "Prefab catalog schemaVersion must contain 1..128 characters";
                return false;
            }
            var ids = new HashSet<string>(StringComparer.Ordinal);
            foreach (RuntimePrefabEntry entry in entries)
            {
                if (entry == null || !SceneProtocol.IsIdentifier(entry.PrefabId, 128))
                {
                    error = "Prefab catalog contains an empty prefabId";
                    return false;
                }
                if (!ids.Add(entry.PrefabId))
                {
                    error = $"Prefab catalog contains duplicate prefabId '{entry.PrefabId}'";
                    return false;
                }
                if (entry.Prefab == null)
                {
                    error = $"Prefab '{entry.PrefabId}' has no GameObject assigned";
                    return false;
                }
                if (entry.Prefab.GetComponent<RpcObjectIdentity>() == null)
                {
                    error = $"Prefab '{entry.PrefabId}' must contain RpcObjectIdentity on its root";
                    return false;
                }
                if (entry.Prefab.activeSelf)
                {
                    error = $"Prefab '{entry.PrefabId}' root must be inactive so BrainRegion can assign its runtime identity before Awake/OnEnable";
                    return false;
                }
                if (entry.Prefab.GetComponentsInChildren<RpcObjectIdentity>(true).Length != 1)
                {
                    error = $"Prefab '{entry.PrefabId}' must expose exactly one root RpcObjectIdentity; nested editable entities belong in separate catalog prefabs";
                    return false;
                }
                if (entry.DisplayName == null || entry.DisplayName.Length > 256 ||
                    entry.Tags == null || entry.Tags.Count > 64 ||
                    entry.Tags.Any(tag => tag == null || tag.Length > 128))
                {
                    error = $"Prefab '{entry.PrefabId}' display name or tags exceed the Scene RPC bounds";
                    return false;
                }
            }

            error = null;
            return true;
        }
    }
}
