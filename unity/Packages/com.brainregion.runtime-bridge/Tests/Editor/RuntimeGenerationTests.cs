using System;
using BrainRegion.RuntimeBridge.Editor;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Build;
using UnityEngine;

namespace BrainRegion.RuntimeBridge.EditorTests
{
    public sealed class RuntimeGenerationTests
    {
        [RpcBindingTarget("generated-test", "brainregion.test.generated")]
        public sealed class GeneratedTarget : MonoBehaviour
        {
            [RpcExposedProperty(
                "intensity",
                DisplayName = "Intensity",
                Minimum = 0,
                Maximum = 5)]
            public int Intensity = 1;
        }

        [Test]
        public void BindingGeneratorEmitsDeterministicDirectMemberAccess()
        {
            string first = RuntimePropertyBindingGenerator.GenerateSource(
                typeof(GeneratedTarget));
            string second = RuntimePropertyBindingGenerator.GenerateSource(
                typeof(GeneratedTarget));

            Assert.That(second, Is.EqualTo(first));
            StringAssert.Contains("target.@Intensity", first);
            StringAssert.Contains(
                "RpcGeneratedBindingValues.TryInteger(proposed, 0D, 5D",
                first);
            StringAssert.DoesNotContain("System.Reflection", first);
            StringAssert.DoesNotContain("GetValue(", first);
            StringAssert.DoesNotContain("SetValue(", first);
        }

        [Test]
        public void CatalogGeneratorUsesStableGuidAndSortedAssetLabels()
        {
            string suffix = Guid.NewGuid().ToString("N");
            string folderName = "__BrainRegionCatalogTest_" + suffix;
            string folderPath = "Assets/" + folderName;
            string prefabPath = folderPath + "/Runtime.prefab";
            string catalogPath = folderPath + "/Catalog.asset";
            string sourceLabel = "BrainRegionTest" + suffix;
            AssetDatabase.CreateFolder("Assets", folderName);
            var root = new GameObject("Generated Catalog Fixture");
            try
            {
                RpcObjectIdentity identity = root.AddComponent<RpcObjectIdentity>();
                var serialized = new SerializedObject(identity);
                serialized.FindProperty("stableId").stringValue = "catalog-test-object";
                serialized.ApplyModifiedPropertiesWithoutUndo();
                root.SetActive(false);
                GameObject prefab = PrefabUtility.SaveAsPrefabAsset(root, prefabPath);
                Assert.That(prefab, Is.Not.Null);
                AssetDatabase.SetLabels(prefab, new[] { sourceLabel, "zeta", "alpha" });

                RuntimePrefabCatalog catalog =
                    RuntimePrefabCatalogGenerator.RebuildCatalog(catalogPath, sourceLabel);
                string firstSchema = catalog.SchemaVersion;
                string guid = AssetDatabase.AssetPathToGUID(prefabPath);
                Assert.That(catalog.Entries.Count, Is.EqualTo(1));
                Assert.That(catalog.Entries[0].PrefabId, Is.EqualTo("prefab:" + guid));
                Assert.That(catalog.Entries[0].Tags, Is.EqualTo(new[] { "alpha", "zeta" }));
                Assert.That(catalog.TryValidate(out string error), Is.True, error);

                RuntimePrefabCatalog rebuilt =
                    RuntimePrefabCatalogGenerator.RebuildCatalog(catalogPath, sourceLabel);
                Assert.That(rebuilt.SchemaVersion, Is.EqualTo(firstSchema));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
                AssetDatabase.DeleteAsset(folderPath);
            }
        }

        [Test]
        public void CatalogGeneratorRejectsActivePrefabRoot()
        {
            string suffix = Guid.NewGuid().ToString("N");
            string folderName = "__BrainRegionActivePrefabTest_" + suffix;
            string folderPath = "Assets/" + folderName;
            string prefabPath = folderPath + "/UnsafeRuntime.prefab";
            string catalogPath = folderPath + "/Catalog.asset";
            string sourceLabel = "BrainRegionActivePrefab" + suffix;
            AssetDatabase.CreateFolder("Assets", folderName);
            var root = new GameObject("Unsafe Active Runtime Prefab");
            try
            {
                root.AddComponent<RpcObjectIdentity>();
                GameObject prefab = PrefabUtility.SaveAsPrefabAsset(root, prefabPath);
                Assert.That(prefab, Is.Not.Null);
                AssetDatabase.SetLabels(prefab, new[] { sourceLabel });

                BuildFailedException exception = Assert.Throws<BuildFailedException>(() =>
                    RuntimePrefabCatalogGenerator.RebuildCatalog(catalogPath, sourceLabel));
                StringAssert.Contains("inactive root", exception.Message);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
                AssetDatabase.DeleteAsset(folderPath);
            }
        }
    }
}
