using System;
using System.Collections.Generic;
using BrainRegion.RuntimeBridge;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace BrainRegion.ScenePipeSmoke
{
    public sealed class ScenePipeSmokeBootstrap : RpcPropertyAdapterBehaviour
    {
        private static readonly IReadOnlyList<RpcPropertyDescriptor> Properties =
            new[]
            {
                new RpcPropertyDescriptor
                {
                    PropertyId = "counter",
                    DisplayName = "Smoke Counter",
                    ValueType = "integer",
                    ReadOnly = false,
                    Persistent = true,
                    Minimum = 0,
                    Maximum = 100,
                },
            };

        [SerializeField] private int counter = 1;

        private WindowsScenePipeTransport transport;
        private string observedError;
        private bool observedConnected;

        public override IReadOnlyList<RpcPropertyDescriptor> DescribeProperties()
        {
            return Properties;
        }

        public override bool TryRead(string propertyId, out JToken value, out string error)
        {
            if (!string.Equals(propertyId, "counter", StringComparison.Ordinal))
            {
                value = null;
                error = "Unknown smoke property";
                return false;
            }
            value = new JValue(counter);
            error = null;
            return true;
        }

        public override bool TryValidate(
            string propertyId,
            JToken proposed,
            out JToken canonical,
            out string error)
        {
            canonical = null;
            if (!string.Equals(propertyId, "counter", StringComparison.Ordinal) ||
                proposed == null || proposed.Type != JTokenType.Integer)
            {
                error = "counter must be an integer";
                return false;
            }

            long candidate = proposed.Value<long>();
            if (candidate < 0 || candidate > 100)
            {
                error = "counter must be between 0 and 100";
                return false;
            }
            canonical = new JValue(checked((int)candidate));
            error = null;
            return true;
        }

        public override bool TryWrite(string propertyId, JToken canonical, out string error)
        {
            if (!TryValidate(propertyId, canonical, out JToken normalized, out error))
                return false;
            counter = normalized.Value<int>();
            error = null;
            return true;
        }

        private void Start()
        {
            transport = GetComponent<WindowsScenePipeTransport>();
            Debug.Log("[BrainRegion Smoke] Windows Scene RPC pipe client started");
        }

        private void Update()
        {
            if (transport == null) return;
            bool connected = transport.IsConnected;
            if (connected != observedConnected)
            {
                observedConnected = connected;
                Debug.Log($"[BrainRegion Smoke] pipe connected={connected}");
            }

            string error = transport.LastError;
            if (!string.IsNullOrEmpty(error) && error != observedError)
            {
                observedError = error;
                Debug.LogError($"[BrainRegion Smoke] pipe error: {error}");
            }
        }
    }
}
