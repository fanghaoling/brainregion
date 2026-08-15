using System;
using System.IO;
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
                ManagedStrippingLevel.Medium);

            Scene scene = EditorSceneManager.NewScene(
                NewSceneSetup.EmptyScene,
                NewSceneMode.Single);
            if (!EditorSceneManager.SaveScene(scene, GeneratedScenePath))
                throw new BuildFailedException("Could not create smoke scene");

            try
            {
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
            }
        }
    }
}
