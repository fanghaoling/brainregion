using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace BrainRegion.RuntimeBridge.Editor
{
    /// <summary>
    /// Build-time guard for persistent Runtime Scene RPC identities. Runtime-spawned
    /// entities receive a session identity from the controller; authored scene
    /// entities must already have a stable, unique identity before the Player build.
    /// </summary>
    public sealed class RuntimeIdentityBuildValidator :
        IPreprocessBuildWithReport,
        IProcessSceneWithReport
    {
        private static readonly HashSet<string> BuildObjectIds =
            new HashSet<string>(StringComparer.Ordinal);

        public int callbackOrder => -1000;

        public void OnPreprocessBuild(BuildReport report)
        {
            BuildObjectIds.Clear();
            ValidatePrefabCatalogs();
        }

        public void OnProcessScene(Scene scene, BuildReport report)
        {
            foreach (GameObject root in scene.GetRootGameObjects())
            {
                foreach (RpcObjectIdentity identity in root.GetComponentsInChildren<RpcObjectIdentity>(true))
                {
                    if (!SceneProtocol.IsIdentifier(identity.StableId, 160))
                    {
                        throw new BuildFailedException(
                            $"BrainRegion Runtime Scene RPC object '{BuildObjectPath(identity.transform)}' " +
                            $"in scene '{scene.path}' has no valid stable ID. Run " +
                            "Tools/BrainRegion/Assign Missing Runtime IDs, save the scene, and rebuild.");
                    }
                    if (!BuildObjectIds.Add(identity.StableId))
                    {
                        throw new BuildFailedException(
                            $"BrainRegion Runtime Scene RPC stable ID '{identity.StableId}' is duplicated " +
                            $"at '{BuildObjectPath(identity.transform)}' in scene '{scene.path}'.");
                    }
                }
            }
        }

        [MenuItem("Tools/BrainRegion/Assign Missing Runtime IDs")]
        private static void AssignMissingRuntimeIds()
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            int assigned = 0;
            for (int sceneIndex = 0; sceneIndex < SceneManager.sceneCount; sceneIndex++)
            {
                Scene scene = SceneManager.GetSceneAt(sceneIndex);
                int assignedInScene = 0;
                foreach (GameObject root in scene.GetRootGameObjects())
                {
                    foreach (RpcObjectIdentity identity in root.GetComponentsInChildren<RpcObjectIdentity>(true))
                    {
                        string current = identity.StableId;
                        if (SceneProtocol.IsIdentifier(current, 160) && seen.Add(current)) continue;

                        Undo.RecordObject(identity, "Assign BrainRegion Runtime ID");
                        var serialized = new SerializedObject(identity);
                        SerializedProperty property = serialized.FindProperty("stableId");
                        string replacement;
                        do
                        {
                            replacement = "scene:" + Guid.NewGuid().ToString("N");
                        }
                        while (!seen.Add(replacement));
                        property.stringValue = replacement;
                        serialized.ApplyModifiedPropertiesWithoutUndo();
                        EditorUtility.SetDirty(identity);
                        assigned++;
                        assignedInScene++;
                    }
                }
                if (assignedInScene > 0) EditorSceneManager.MarkSceneDirty(scene);
            }
            Debug.Log($"[BrainRegion] Assigned {assigned} missing or duplicate Runtime Scene RPC ID(s). Save modified scenes before building.");
        }

        private static void ValidatePrefabCatalogs()
        {
            foreach (string guid in AssetDatabase.FindAssets("t:RuntimePrefabCatalog"))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                RuntimePrefabCatalog catalog = AssetDatabase.LoadAssetAtPath<RuntimePrefabCatalog>(path);
                if (catalog != null && !catalog.TryValidate(out string error))
                    throw new BuildFailedException($"Invalid BrainRegion prefab catalog '{path}': {error}");
            }
        }

        private static string BuildObjectPath(Transform target)
        {
            string path = target.name;
            while (target.parent != null)
            {
                target = target.parent;
                path = target.name + "/" + path;
            }
            return path;
        }
    }
}
