using System;
using System.Collections.Generic;
using BrainRegion.RuntimeBridge;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace BrainRegion.ScenePipeSmoke
{
    /// <summary>
    /// Packaged-Player assertion surface proving that a catalog prefab receives its
    /// runtime identity before any project Awake/OnEnable callback can observe it.
    /// </summary>
    public sealed class StagedPrefabProbe : RpcPropertyAdapterBehaviour
    {
        private static readonly IReadOnlyList<RpcPropertyDescriptor> Properties =
            new[]
            {
                ReadOnlyString("awake_identity", "Identity observed by Awake"),
                ReadOnlyString("enable_identity", "Identity observed by OnEnable"),
            };

        private string awakeIdentity = string.Empty;
        private string enableIdentity = string.Empty;

        public override IReadOnlyList<RpcPropertyDescriptor> DescribeProperties()
        {
            return Properties;
        }

        public override bool TryRead(string propertyId, out JToken value, out string error)
        {
            if (string.Equals(propertyId, "awake_identity", StringComparison.Ordinal))
            {
                value = new JValue(awakeIdentity);
                error = null;
                return true;
            }
            if (string.Equals(propertyId, "enable_identity", StringComparison.Ordinal))
            {
                value = new JValue(enableIdentity);
                error = null;
                return true;
            }
            value = null;
            error = "Unknown staging probe property";
            return false;
        }

        public override bool TryValidate(
            string propertyId,
            JToken proposed,
            out JToken canonical,
            out string error)
        {
            canonical = null;
            error = "Staging probe properties are read-only";
            return false;
        }

        public override bool TryWrite(string propertyId, JToken canonical, out string error)
        {
            error = "Staging probe properties are read-only";
            return false;
        }

        private void Awake()
        {
            awakeIdentity = GetComponent<RpcObjectIdentity>()?.StableId ?? string.Empty;
        }

        private void OnEnable()
        {
            enableIdentity = GetComponent<RpcObjectIdentity>()?.StableId ?? string.Empty;
        }

        private static RpcPropertyDescriptor ReadOnlyString(string propertyId, string displayName)
        {
            return new RpcPropertyDescriptor
            {
                PropertyId = propertyId,
                DisplayName = displayName,
                ValueType = "string",
                ReadOnly = true,
                Persistent = false,
            };
        }
    }
}
