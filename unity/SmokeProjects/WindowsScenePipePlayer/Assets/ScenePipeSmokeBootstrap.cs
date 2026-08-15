using BrainRegion.RuntimeBridge;
using UnityEngine;

namespace BrainRegion.ScenePipeSmoke
{
    internal static class ScenePipeSmokeBootstrap
    {
        private static bool initialized;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void Initialize()
        {
            if (initialized) return;
            initialized = true;
            Application.runInBackground = true;

            var root = new GameObject("BrainRegion Scene Pipe Smoke Root");
            Object.DontDestroyOnLoad(root);

            // Add in this order so the controller indexes one readable identity
            // before the dispatcher builds its first registration snapshot.
            root.AddComponent<RpcObjectIdentity>();
            root.AddComponent<RuntimeSceneController>();
            root.AddComponent<RuntimeLogBuffer>();
            root.AddComponent<SceneRpcDispatcher>();
            WindowsScenePipeTransport transport =
                root.AddComponent<WindowsScenePipeTransport>();
            root.AddComponent<ScenePipeSmokeReporter>().Bind(transport);
            transport.Connect();

            Debug.Log("[BrainRegion Smoke] Windows Scene RPC pipe client started");
        }
    }

    internal sealed class ScenePipeSmokeReporter : MonoBehaviour
    {
        private WindowsScenePipeTransport transport;
        private string observedError;
        private bool observedConnected;

        internal void Bind(WindowsScenePipeTransport value)
        {
            transport = value;
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
