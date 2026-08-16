using BrainRegion.RuntimeBridge;
using UnityEngine;

namespace BrainRegion.ScenePipeSmoke
{
    [RpcBindingTarget("generated", "brainregion.smoke.generated")]
    public sealed class GeneratedSmokeProperties : MonoBehaviour
    {
        [RpcExposedProperty(
            "brightness",
            DisplayName = "Generated Brightness",
            Minimum = 0,
            Maximum = 10)]
        public int Brightness = 2;
    }
}
