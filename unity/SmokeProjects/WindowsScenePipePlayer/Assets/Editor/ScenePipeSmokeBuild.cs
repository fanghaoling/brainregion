using System;
using System.IO;
using System.Reflection;
using BrainRegion.RuntimeBridge;
using BrainRegion.RuntimeBridge.Editor;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace BrainRegion.ScenePipeSmoke.Editor
{
    internal static class ScenePipeSmokeBuild
    {
        private const string OutputEnvironmentVariable =
            "BRAINREGION_UNITY_SMOKE_OUTPUT";
        private const string GeneratedScenePath =
            "Assets/__GeneratedSmokeScene.unity";
        private const string GeneratedPrefabPath =
            "Assets/Generated Runtime Prefab.prefab";
        private const string GeneratedAdapterTypeName =
            "BrainRegion.RuntimeBridge.Generated.BrainRegion_ScenePipeSmoke_GeneratedSmokePropertiesBrainRegionRpcAdapter, Assembly-CSharp";

        public static void BuildWindowsIl2Cpp()
        {
            string outputPath = Environment.GetEnvironmentVariable(
                OutputEnvironmentVariable);
            if (string.IsNullOrWhiteSpace(outputPath) ||
                !Path.IsPathFullyQualified(outputPath) ||
                !string.Equals(Path.GetExtension(outputPath), ".exe",
                    StringComparison.OrdinalIgnoreCase))
            {
                throw new BuildFailedException(
                    OutputEnvironmentVariable +
                    " must be an absolute Windows .exe path");
            }

            outputPath = Path.GetFullPath(outputPath);
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
            if (!EditorUserBuildSettings.SwitchActiveBuildTarget(
                    BuildTargetGroup.Standalone,
                    BuildTarget.StandaloneWindows64))
            {
                throw new BuildFailedException(
                    "Could not select StandaloneWindows64");
            }

            PlayerSettings.companyName = "BrainRegion";
            PlayerSettings.productName = "BrainRegion Scene Pipe Smoke";
            PlayerSettings.bundleVersion = "0.2.0-smoke";
            PlayerSettings.runInBackground = true;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.defaultScreenWidth = 640;
            PlayerSettings.defaultScreenHeight = 360;
            PlayerSettings.SetScriptingBackend(
                NamedBuildTarget.Standalone,
                ScriptingImplementation.IL2CPP);
            PlayerSettings.SetIl2CppCompilerConfiguration(
                NamedBuildTarget.Standalone,
                Il2CppCompilerConfiguration.Release);
            PlayerSettings.SetManagedStrippingLevel(
                NamedBuildTarget.Standalone,
                ManagedStrippingLevel.High);

            try
            {
                CreateRuntimePrefab();
                RuntimePrefabCatalogGenerator.RebuildDefaultCatalog();
                Scene scene = EditorSceneManager.NewScene(
                    NewSceneSetup.EmptyScene,
                    NewSceneMode.Single);
                RuntimePrefabCatalog catalog =
                    AssetDatabase.LoadAssetAtPath<RuntimePrefabCatalog>(
                        RuntimePrefabCatalogGenerator.DefaultCatalogPath);
                RuntimeSceneController sourceController = CreateRuntimeFixture(catalog);
                if (!EditorSceneManager.SaveScene(scene, GeneratedScenePath))
                    throw new BuildFailedException("Could not create smoke scene");
                AssetDatabase.SaveAssets();
                Debug.Log(
                    "[BrainRegion Smoke] source scene catalog=" +
                    (sourceController.PrefabCatalog == null
                        ? "null"
                        : sourceController.PrefabCatalog.SchemaVersion + "/" +
                          sourceController.PrefabCatalog.Entries.Count));

                BuildReport report = BuildPipeline.BuildPlayer(
                    new BuildPlayerOptions
                    {
                        scenes = new[] { GeneratedScenePath },
                        locationPathName = outputPath,
                        target = BuildTarget.StandaloneWindows64,
                        options = BuildOptions.Development,
                    });
                if (report.summary.result != BuildResult.Succeeded)
                {
                    throw new BuildFailedException(
                        $"Smoke Player build failed: {report.summary.result}; " +
                        $"errors={report.summary.totalErrors}");
                }

                Debug.Log(
                    $"[BrainRegion Smoke] IL2CPP Player built at {outputPath} " +
                    $"({report.summary.totalSize} bytes)");
            }
            finally
            {
                AssetDatabase.DeleteAsset(GeneratedScenePath);
                AssetDatabase.DeleteAsset(GeneratedPrefabPath);
                AssetDatabase.DeleteAsset(
                    RuntimePrefabCatalogGenerator.DefaultCatalogPath);
            }
        }

        private static RuntimeSceneController CreateRuntimeFixture(RuntimePrefabCatalog catalog)
        {
            if (catalog == null)
                throw new BuildFailedException(
                    "Generated Runtime prefab catalog returned an invalid Unity object handle");
            var root = new GameObject("BrainRegion Scene Pipe Smoke Root");

            RpcObjectIdentity identity = root.AddComponent<RpcObjectIdentity>();
            ScenePipeSmokeBootstrap adapter = root.AddComponent<ScenePipeSmokeBootstrap>();
            root.AddComponent<GeneratedSmokeProperties>();
            Type generatedAdapterType = Type.GetType(
                GeneratedAdapterTypeName,
                false);
            if (generatedAdapterType == null ||
                !typeof(IRpcPropertyAdapter).IsAssignableFrom(generatedAdapterType))
            {
                throw new BuildFailedException(
                    "Generated smoke property adapter is missing. Run the binding generator before building.");
            }
            root.AddComponent(generatedAdapterType);
            RuntimeSceneController controller = root.AddComponent<RuntimeSceneController>();
            root.AddComponent<RuntimeLogBuffer>();
            root.AddComponent<SceneRpcDispatcher>();
            WindowsScenePipeTransport transport =
                root.AddComponent<WindowsScenePipeTransport>();

            SetSerializedString(identity, "stableId", "smoke-object-01");
            SetSerializedBoolean(identity, "allowRemoteChanges", true);
            SetSerializedString(adapter, "componentKey", "smoke");
            SetSerializedString(adapter, "typeId", "brainregion.smoke");
            SetSerializedObject(controller, "prefabCatalog", catalog);
            SetSerializedBoolean(controller, "includeLoadedScenes", true);
            SetSerializedBoolean(transport, "connectOnEnable", true);
            return controller;
        }

        private static void CreateRuntimePrefab()
        {
            var root = new GameObject("Generated Runtime Prefab");
            try
            {
                RpcObjectIdentity identity = root.AddComponent<RpcObjectIdentity>();
                StagedPrefabProbe probe = root.AddComponent<StagedPrefabProbe>();
                SetSerializedString(identity, "stableId", "prefab-template-smoke");
                SetSerializedString(probe, "componentKey", "staging");
                SetSerializedString(probe, "typeId", "brainregion.smoke.staging");
                root.SetActive(false);
                GameObject prefab = PrefabUtility.SaveAsPrefabAsset(
                    root,
                    GeneratedPrefabPath);
                if (prefab == null)
                    throw new BuildFailedException("Could not create smoke Runtime prefab");
                AssetDatabase.SetLabels(
                    prefab,
                    new[]
                    {
                        RuntimePrefabCatalogGenerator.SourceLabel,
                        "smoke",
                    });
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void SetSerializedString(
            UnityEngine.Object target,
            string propertyName,
            string value)
        {
            var serialized = new SerializedObject(target);
            SerializedProperty property = serialized.FindProperty(propertyName);
            if (property == null)
                throw new BuildFailedException(
                    $"Could not configure serialized property '{propertyName}'");
            property.stringValue = value;
            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(target);
        }

        private static void SetSerializedBoolean(
            UnityEngine.Object target,
            string propertyName,
            bool value)
        {
            var serialized = new SerializedObject(target);
            SerializedProperty property = serialized.FindProperty(propertyName);
            if (property == null)
                throw new BuildFailedException(
                    $"Could not configure serialized property '{propertyName}'");
            property.boolValue = value;
            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(target);
        }

        private static void SetSerializedObject(
            UnityEngine.Object target,
            string propertyName,
            UnityEngine.Object value)
        {
            FieldInfo field = target.GetType().GetField(
                propertyName,
                BindingFlags.Instance | BindingFlags.NonPublic);
            if (field == null || !typeof(UnityEngine.Object).IsAssignableFrom(field.FieldType))
                throw new BuildFailedException(
                    $"Could not configure serialized property '{propertyName}'");
            field.SetValue(target, value);
            EditorUtility.SetDirty(target);
        }
    }

    internal sealed class ScenePipeSmokeBuildSceneValidator : IProcessSceneWithReport
    {
        public int callbackOrder => 1000;

        public void OnProcessScene(Scene scene, BuildReport report)
        {
            if (!string.Equals(
                    scene.path,
                    "Assets/__GeneratedSmokeScene.unity",
                    StringComparison.Ordinal))
                return;

            RuntimeSceneController controller = null;
            foreach (GameObject root in scene.GetRootGameObjects())
            {
                controller = root.GetComponentInChildren<RuntimeSceneController>(true);
                if (controller != null) break;
            }
            if (controller == null)
                throw new BuildFailedException(
                    "Smoke build scene lost its RuntimeSceneController");
            if (controller.PrefabCatalog == null)
                throw new BuildFailedException(
                    "Smoke build scene lost its generated Runtime prefab catalog reference");
            if (controller.PrefabCatalog.Entries.Count != 1)
            {
                throw new BuildFailedException(
                    "Smoke build scene catalog has " +
                    controller.PrefabCatalog.Entries.Count + " entries instead of one");
            }
            Debug.Log(
                "[BrainRegion Smoke] build scene catalog=" +
                controller.PrefabCatalog.SchemaVersion);
        }
    }
}
