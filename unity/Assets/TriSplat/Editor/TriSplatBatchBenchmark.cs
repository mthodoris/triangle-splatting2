// Batch-mode physics benchmark (plan step 9.8): runs physproxy's PhysProxyBenchmark in Play mode without a window,
// then quits. Environment: TRISPLAT_PHYS (folder under StreamingAssets with *.ppm, drops.csv, walks.csv),
// TRISPLAT_BENCH_OUT (CSV folder), TRISPLAT_RADII (e.g. "0.01,0.02,0.05"), TRISPLAT_COLLIDERS (default: physproxy's).
//   Unity -batchmode -projectPath <project> -executeMethod TriSplat.TriSplatBatchBenchmark.Run -logFile -
// (no -quit: Unity exits by itself when the benchmark is done)
using UnityEditor;
using UnityEngine;

namespace TriSplat
{
    [InitializeOnLoad]
    public static class TriSplatBatchBenchmark
    {
        const string Key = "TRISPLAT_BENCH_STARTED";

        static TriSplatBatchBenchmark()
        {
            // Re-registered after every domain reload (entering Play mode reloads scripts).
            if (!Application.isBatchMode || !SessionState.GetBool(Key, false)) return;
            EditorApplication.update += Watch;
        }

        public static void Run()
        {
            var scene = UnityEditor.SceneManagement.EditorSceneManager.NewScene(
                UnityEditor.SceneManagement.NewSceneSetup.EmptyScene, UnityEditor.SceneManagement.NewSceneMode.Single);
            var go = new GameObject("benchmark");
            var b = go.AddComponent<PhysProxy.PhysProxyBenchmark>();
            b.dataFolder = System.Environment.GetEnvironmentVariable("TRISPLAT_PHYS");
            b.outputFolder = System.Environment.GetEnvironmentVariable("TRISPLAT_BENCH_OUT");
            var cols = System.Environment.GetEnvironmentVariable("TRISPLAT_COLLIDERS");   // e.g. "collision_ext,soup_all"
            if (!string.IsNullOrEmpty(cols)) b.colliders = cols.Split(',');
            var radii = System.Environment.GetEnvironmentVariable("TRISPLAT_RADII");
            if (!string.IsNullOrEmpty(radii))
                b.sphereRadii = System.Array.ConvertAll(radii.Split(','), x => float.Parse(x, System.Globalization.CultureInfo.InvariantCulture));
            SessionState.SetBool(Key, true);
            Debug.Log($"[TriSplatBatchBenchmark] {b.dataFolder} -> {b.outputFolder}, entering Play mode");
            EditorApplication.EnterPlaymode();
        }

        static void Watch()
        {
            if (!EditorApplication.isPlaying || !PhysProxy.PhysProxyBenchmark.Done) return;
            Debug.Log("[TriSplatBatchBenchmark] done, quitting");
            SessionState.SetBool(Key, false);
            EditorApplication.Exit(0);
        }
    }
}
