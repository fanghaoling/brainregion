using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace BrainRegion.RuntimeBridge
{
    /// <summary>
    /// Explicit property surface exposed to Scene RPC. Implementations must not use
    /// arbitrary reflection. They validate and canonicalize input before any write.
    /// TryRead/TryValidate/TryWrite must not throw. Validation must be deterministic
    /// and independent of mutable scene state; model cross-field invariants as one
    /// compound exposed property. A successful write may only mutate that represented
    /// property and must be reversible by writing the value returned by TryRead.
    /// </summary>
    public interface IRpcPropertyAdapter
    {
        string ComponentKey { get; }
        string TypeId { get; }
        IReadOnlyList<RpcPropertyDescriptor> DescribeProperties();
        bool TryRead(string propertyId, out JToken value, out string error);
        bool TryValidate(string propertyId, JToken proposed, out JToken canonical, out string error);
        bool TryWrite(string propertyId, JToken canonical, out string error);
    }

    /// <summary>
    /// Convenience base class for project-defined, AOT-safe adapters. ComponentKey
    /// is local to one object; the wire component id is objectId/componentKey.
    /// </summary>
    public abstract class RpcPropertyAdapterBehaviour : MonoBehaviour, IRpcPropertyAdapter
    {
        [SerializeField] private string componentKey = "component";
        [SerializeField] private string typeId = "project.component";

        public string ComponentKey => componentKey;
        public string TypeId => typeId;

        public abstract IReadOnlyList<RpcPropertyDescriptor> DescribeProperties();
        public abstract bool TryRead(string propertyId, out JToken value, out string error);
        public abstract bool TryValidate(string propertyId, JToken proposed, out JToken canonical, out string error);
        public abstract bool TryWrite(string propertyId, JToken canonical, out string error);

        protected virtual void OnValidate()
        {
            componentKey = SanitizeIdentifier(componentKey, "component");
            typeId = SanitizeIdentifier(typeId, "project.component");
        }

        private static string SanitizeIdentifier(string value, string fallback)
        {
            if (string.IsNullOrWhiteSpace(value)) return fallback;
            return value.Trim().Replace("/", "_").Replace("\\", "_");
        }
    }
}
