using System;
using UnityEngine;

namespace BrainRegion.RuntimeBridge
{
    /// <summary>
    /// Stable protocol identity for an explicitly editable runtime object.
    /// Unity InstanceID is intentionally never exposed because it is process-local
    /// and is not stable across loads.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class RpcObjectIdentity : MonoBehaviour
    {
        [SerializeField] private string stableId;
        [SerializeField] private string prefabId;
        [SerializeField] private bool allowRemoteChanges;

        public string StableId => stableId;
        public string PrefabId => prefabId;
        public bool AllowRemoteChanges => allowRemoteChanges;

        internal void EnsureIdentity()
        {
            if (string.IsNullOrWhiteSpace(stableId))
                stableId = Guid.NewGuid().ToString("N");
        }

        internal void AssignRuntimeIdentity(string objectId, string sourcePrefabId)
        {
            if (string.IsNullOrWhiteSpace(objectId))
                throw new ArgumentException("objectId must not be empty", nameof(objectId));

            stableId = objectId;
            prefabId = sourcePrefabId ?? string.Empty;
        }

        internal void ReplaceDuplicateIdentity()
        {
            stableId = Guid.NewGuid().ToString("N");
        }
    }
}
