using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace BrainRegion.RuntimeBridge.Editor
{
    /// <summary>
    /// Builds one deterministic application-owned catalog from prefab assets carrying
    /// the BrainRegionRuntimePrefab label. Prefab IDs derive from Unity asset GUIDs,
    /// so renames and moves do not change the wire identity.
    /// </summary>
    public sealed class RuntimePrefabCatalogGenerator : IPreprocessBuildWithReport
    {
        public const string SourceLabel = "BrainRegionRuntimePrefab";
        public const string DefaultCatalogPath =
            "Assets/BrainRegionGenerated/RuntimePrefabCatalog.asset";

        public int callbackOrder => -1100;

        public void OnPreprocessBuild(BuildReport report)
        {
            if (AssetDatabase.FindAssets($"l:{SourceLabel} t:Prefab").Length > 0 ||
                AssetDatabase.LoadAssetAtPath<RuntimePrefabCatalog>(DefaultCatalogPath) != null)
                RebuildDefaultCatalog();
        }

        [MenuItem("Tools/BrainRegion/Rebuild Runtime Prefab Catalog")]
        public static RuntimePrefabCatalog RebuildDefaultCatalog()
        {
            return RebuildCatalog(DefaultCatalogPath, SourceLabel);
        }

        public static RuntimePrefabCatalog RebuildCatalog(
            string catalogPath,
            string sourceLabel)
        {
            ValidateAssetPath(catalogPath);
            if (string.IsNullOrWhiteSpace(sourceLabel) || sourceLabel.Length > 128)
                throw new ArgumentException("Catalog source label must contain 1..128 characters", nameof(sourceLabel));

            List<CatalogEntry> generated = AssetDatabase
                .FindAssets($"l:{sourceLabel} t:Prefab")
                .Select(guid => BuildEntry(guid, sourceLabel))
                .OrderBy(entry => entry.PrefabId, StringComparer.Ordinal)
                .ToList();

            EnsureAssetFolder(catalogPath);
            RuntimePrefabCatalog catalog =
                AssetDatabase.LoadAssetAtPath<RuntimePrefabCatalog>(catalogPath);
            if (catalog == null)
            {
                catalog = ScriptableObject.CreateInstance<RuntimePrefabCatalog>();
                AssetDatabase.CreateAsset(catalog, catalogPath);
            }

            var serialized = new SerializedObject(catalog);
            serialized.FindProperty("schemaVersion").stringValue =
                "1-" + ComputeDigest(generated).Substring(0, 16);
            SerializedProperty entries = serialized.FindProperty("entries");
            entries.arraySize = generated.Count;
            for (int index = 0; index < generated.Count; index++)
            {
                CatalogEntry source = generated[index];
                SerializedProperty target = entries.GetArrayElementAtIndex(index);
                target.FindPropertyRelative("prefabId").stringValue = source.PrefabId;
                target.FindPropertyRelative("displayName").stringValue = source.DisplayName;
                target.FindPropertyRelative("prefab").objectReferenceValue = source.Prefab;
                SerializedProperty tags = target.FindPropertyRelative("tags");
                tags.arraySize = source.Tags.Count;
                for (int tagIndex = 0; tagIndex < source.Tags.Count; tagIndex++)
                    tags.GetArrayElementAtIndex(tagIndex).stringValue = source.Tags[tagIndex];
            }
            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(catalog);
            AssetDatabase.SaveAssets();
            AssetDatabase.ImportAsset(
                catalogPath,
                ImportAssetOptions.ForceSynchronousImport |
                ImportAssetOptions.ForceUpdate);
            catalog = AssetDatabase.LoadAssetAtPath<RuntimePrefabCatalog>(catalogPath);
            if (catalog == null)
                throw new BuildFailedException(
                    $"Generated BrainRegion prefab catalog could not be reloaded from '{catalogPath}'");

            if (!catalog.TryValidate(out string error))
                throw new BuildFailedException($"Generated BrainRegion prefab catalog is invalid: {error}");
            Debug.Log($"[BrainRegion] Generated {generated.Count} runtime prefab catalog entries at {catalogPath}");
            return catalog;
        }

        private static CatalogEntry BuildEntry(string guid, string sourceLabel)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (prefab == null)
                throw new BuildFailedException($"BrainRegion catalog source '{path}' is not a GameObject prefab");
            RpcObjectIdentity rootIdentity = prefab.GetComponent<RpcObjectIdentity>();
            if (rootIdentity == null ||
                prefab.GetComponentsInChildren<RpcObjectIdentity>(true).Length != 1)
            {
                throw new BuildFailedException(
                    $"BrainRegion catalog prefab '{path}' must have exactly one root RpcObjectIdentity");
            }

            List<string> tags = AssetDatabase.GetLabels(prefab)
                .Where(label => !string.Equals(label, sourceLabel, StringComparison.Ordinal))
                .OrderBy(label => label, StringComparer.Ordinal)
                .ToList();
            if (tags.Count > 64 || tags.Any(tag => tag == null || tag.Length > 128))
                throw new BuildFailedException($"BrainRegion catalog prefab '{path}' has invalid labels");
            return new CatalogEntry
            {
                PrefabId = "prefab:" + guid,
                DisplayName = prefab.name,
                Prefab = prefab,
                Tags = tags,
            };
        }

        private static string ComputeDigest(IEnumerable<CatalogEntry> entries)
        {
            var canonical = new StringBuilder();
            foreach (CatalogEntry entry in entries)
            {
                canonical.Append(entry.PrefabId).Append('\n')
                    .Append(entry.DisplayName).Append('\n')
                    .Append(string.Join("\u001f", entry.Tags)).Append('\n');
            }
            using (SHA256 sha256 = SHA256.Create())
            {
                byte[] digest = sha256.ComputeHash(Encoding.UTF8.GetBytes(canonical.ToString()));
                return BitConverter.ToString(digest).Replace("-", string.Empty).ToLowerInvariant();
            }
        }

        private static void ValidateAssetPath(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || !path.StartsWith("Assets/", StringComparison.Ordinal) ||
                !path.EndsWith(".asset", StringComparison.OrdinalIgnoreCase) || path.Contains(".."))
                throw new ArgumentException("Catalog path must be a normalized .asset path below Assets", nameof(path));
        }

        private static void EnsureAssetFolder(string assetPath)
        {
            string folder = assetPath.Substring(0, assetPath.LastIndexOf('/'));
            string[] segments = folder.Split('/');
            string current = segments[0];
            for (int index = 1; index < segments.Length; index++)
            {
                string next = current + "/" + segments[index];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(current, segments[index]);
                current = next;
            }
        }

        private sealed class CatalogEntry
        {
            public string PrefabId;
            public string DisplayName;
            public GameObject Prefab;
            public List<string> Tags;
        }
    }
}
