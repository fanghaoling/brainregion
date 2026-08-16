using System;
using UnityEngine;

namespace BrainRegion.RuntimeBridge
{
    /// <summary>
    /// Process-local lifecycle signal used by active controllers to reconcile their
    /// registries on the Unity main thread. It deliberately carries no protocol
    /// identity or transport state.
    /// </summary>
    internal static class RuntimeIdentityLifecycle
    {
        internal static event Action<RpcObjectIdentity> Changed;

        internal static void NotifyChanged(RpcObjectIdentity identity)
        {
            Delegate[] handlers = Changed?.GetInvocationList();
            if (handlers == null) return;
            foreach (Delegate handler in handlers)
            {
                try
                {
                    ((Action<RpcObjectIdentity>)handler)(identity);
                }
                catch (Exception exception)
                {
                    Debug.LogError(
                        "[BrainRegion] Runtime identity lifecycle subscriber failed: " +
                        exception.Message);
                }
            }
        }
    }

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
        private RuntimeSceneController owner;

        public string StableId => stableId;
        public string PrefabId => prefabId;
        public bool AllowRemoteChanges => allowRemoteChanges;

        internal RuntimeSceneController Owner => owner;

        private void OnEnable()
        {
            RuntimeIdentityLifecycle.NotifyChanged(this);
        }

        private void OnDestroy()
        {
            RuntimeSceneController currentOwner = owner;
            string currentStableId = stableId;
            if (currentOwner != null)
                currentOwner.NotifyIdentityDestroyed(this, currentStableId);
            RuntimeIdentityLifecycle.NotifyChanged(this);
        }

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

        internal bool TryClaim(RuntimeSceneController controller, out string error)
        {
            if (controller == null)
            {
                error = "Runtime identity owner must not be null";
                return false;
            }
            if (owner != null && owner != controller)
            {
                error = $"Runtime object '{stableId}' is already owned by another scene controller";
                return false;
            }
            owner = controller;
            error = null;
            return true;
        }

        internal void Release(RuntimeSceneController controller)
        {
            if (owner == controller)
                owner = null;
        }
    }
}
