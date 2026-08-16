using System;
using System.Collections.Generic;
using BrainRegion.RuntimeBridge;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace BrainRegion.ScenePipeSmoke
{
    public sealed class ScenePipeSmokeBootstrap : RpcPropertyAdapterBehaviour
    {
        private const string CounterPropertyId = "counter";
        private const string InjectWriteFailurePropertyId = "inject_write_failure";
        private const string DisconnectAfterWritePropertyId = "disconnect_after_write";
        private const string LifecycleCommandPropertyId = "lifecycle_command";
        private const string DynamicObjectName = "BrainRegion Dynamic Lifecycle Object";
        private const string AdditiveObjectName = "BrainRegion Additive Lifecycle Object";
        private const string AdditiveSceneName = "BrainRegion Lifecycle Additive Scene";

        private static readonly IReadOnlyList<RpcPropertyDescriptor> Properties =
            new[]
            {
                new RpcPropertyDescriptor
                {
                    PropertyId = CounterPropertyId,
                    DisplayName = "Smoke Counter",
                    ValueType = "integer",
                    ReadOnly = false,
                    Persistent = true,
                    Minimum = 0,
                    Maximum = 100,
                },
                new RpcPropertyDescriptor
                {
                    PropertyId = InjectWriteFailurePropertyId,
                    DisplayName = "Inject Write Failure",
                    ValueType = "boolean",
                    ReadOnly = false,
                    Persistent = false,
                },
                new RpcPropertyDescriptor
                {
                    PropertyId = DisconnectAfterWritePropertyId,
                    DisplayName = "Disconnect After Write",
                    ValueType = "boolean",
                    ReadOnly = false,
                    Persistent = false,
                },
                new RpcPropertyDescriptor
                {
                    PropertyId = LifecycleCommandPropertyId,
                    DisplayName = "Lifecycle smoke command",
                    ValueType = "enum",
                    ReadOnly = false,
                    Persistent = false,
                    EnumValues = new List<string>
                    {
                        string.Empty,
                        "create_dynamic",
                        "destroy_dynamic",
                        "load_additive",
                        "unload_additive",
                    },
                },
            };

        [SerializeField] private int counter = 1;

        private WindowsScenePipeTransport transport;
        private string observedError;
        private bool observedConnected;
        private string pendingLifecycleCommand = string.Empty;
        private GameObject dynamicObject;
        private Scene additiveScene;

        public override IReadOnlyList<RpcPropertyDescriptor> DescribeProperties()
        {
            return Properties;
        }

        public override bool TryRead(string propertyId, out JToken value, out string error)
        {
            if (string.Equals(propertyId, CounterPropertyId, StringComparison.Ordinal))
            {
                value = new JValue(counter);
                error = null;
                return true;
            }
            if (string.Equals(propertyId, InjectWriteFailurePropertyId, StringComparison.Ordinal) ||
                string.Equals(propertyId, DisconnectAfterWritePropertyId, StringComparison.Ordinal))
            {
                value = new JValue(false);
                error = null;
                return true;
            }
            if (string.Equals(propertyId, LifecycleCommandPropertyId, StringComparison.Ordinal))
            {
                value = new JValue(pendingLifecycleCommand);
                error = null;
                return true;
            }

            value = null;
            error = "Unknown smoke property";
            return false;
        }

        public override bool TryValidate(
            string propertyId,
            JToken proposed,
            out JToken canonical,
            out string error)
        {
            canonical = null;
            if (string.Equals(propertyId, CounterPropertyId, StringComparison.Ordinal))
            {
                if (proposed == null || proposed.Type != JTokenType.Integer)
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

            if ((string.Equals(propertyId, InjectWriteFailurePropertyId, StringComparison.Ordinal) ||
                 string.Equals(propertyId, DisconnectAfterWritePropertyId, StringComparison.Ordinal)) &&
                proposed != null && proposed.Type == JTokenType.Boolean)
            {
                canonical = new JValue(proposed.Value<bool>());
                error = null;
                return true;
            }
            if (string.Equals(propertyId, LifecycleCommandPropertyId, StringComparison.Ordinal) &&
                proposed != null && proposed.Type == JTokenType.String)
            {
                string command = proposed.Value<string>() ?? string.Empty;
                if (command == string.Empty || command == "create_dynamic" ||
                    command == "destroy_dynamic" || command == "load_additive" ||
                    command == "unload_additive")
                {
                    canonical = new JValue(command);
                    error = null;
                    return true;
                }
            }

            error = "Unknown smoke property or invalid value type";
            return false;
        }

        public override bool TryWrite(string propertyId, JToken canonical, out string error)
        {
            if (!TryValidate(propertyId, canonical, out JToken normalized, out error))
                return false;

            if (string.Equals(propertyId, CounterPropertyId, StringComparison.Ordinal))
            {
                counter = normalized.Value<int>();
            }
            else if (string.Equals(propertyId, InjectWriteFailurePropertyId, StringComparison.Ordinal) &&
                     normalized.Value<bool>())
            {
                error = "Injected smoke adapter write failure";
                return false;
            }
            else if (string.Equals(propertyId, DisconnectAfterWritePropertyId, StringComparison.Ordinal) &&
                     normalized.Value<bool>())
            {
                if (transport == null)
                {
                    error = "Smoke transport is not initialized";
                    return false;
                }

                // This property must be the final change in its transaction. Closing
                // the pipe here discards that request's response, while Connect keeps
                // the Player eligible for a new authenticated connection epoch. The
                // controller completes the apply and records its receipt afterwards.
                transport.Disconnect();
                transport.Connect();
            }
            else if (string.Equals(propertyId, LifecycleCommandPropertyId, StringComparison.Ordinal))
            {
                pendingLifecycleCommand = normalized.Value<string>() ?? string.Empty;
            }

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
            ExecutePendingLifecycleCommand();
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

        private void ExecutePendingLifecycleCommand()
        {
            string command = pendingLifecycleCommand;
            if (string.IsNullOrEmpty(command)) return;
            pendingLifecycleCommand = string.Empty;

            switch (command)
            {
                case "create_dynamic":
                    if (dynamicObject != null) return;
                    dynamicObject = new GameObject(DynamicObjectName);
                    dynamicObject.transform.SetParent(transform, false);
                    dynamicObject.AddComponent<RpcObjectIdentity>();
                    Debug.Log("[BrainRegion Smoke] created dynamic editable object");
                    return;

                case "destroy_dynamic":
                    if (dynamicObject == null) return;
                    Destroy(dynamicObject);
                    dynamicObject = null;
                    Debug.Log("[BrainRegion Smoke] destroyed dynamic editable object");
                    return;

                case "load_additive":
                    if (additiveScene.IsValid() && additiveScene.isLoaded) return;
                    additiveScene = SceneManager.CreateScene(AdditiveSceneName);
                    var additiveObject = new GameObject(AdditiveObjectName);
                    SceneManager.MoveGameObjectToScene(additiveObject, additiveScene);
                    additiveObject.AddComponent<RpcObjectIdentity>();
                    Debug.Log("[BrainRegion Smoke] loaded additive editable scene");
                    return;

                case "unload_additive":
                    if (!additiveScene.IsValid() || !additiveScene.isLoaded) return;
                    SceneManager.UnloadSceneAsync(additiveScene);
                    Debug.Log("[BrainRegion Smoke] requested additive scene unload");
                    return;
            }
        }
    }
}
