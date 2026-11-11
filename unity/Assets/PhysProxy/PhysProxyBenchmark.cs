// physproxy Phase 7: engine benchmark. For each collider, in its own local physics scene:
//   Drop test  - spheres released at drops.csv points (seen-as-empty space), simulated with a fixed
//                timestep; logs where each one rests and whether it fell through the world.
//   Walk test  - a CharacterController capsule walks a straight line from start to goal (walks.csv)
//                with gravity; logs reached / stuck / fell through.
// No randomness: inputs come from the CSVs and the timestep is fixed, so runs are repeatable.
// Output: <outputFolder>/drops_<collider>.csv, walks_<collider>.csv (default: Application.persistentDataPath).
// Summarise with: python unity/summarize.py <outputFolder>
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PhysProxy
{
    public class PhysProxyBenchmark : MonoBehaviour
    {
        public string dataFolder = "physproxy/garden";
        public string[] colliders = { "collision", "naive", "soup" };
        public string outputFolder = "";

        [Header("Drop test")]
        public float sphereRadius = 0.05f;
        public float dropSeconds = 4f;
        public int dropBatch = 500;
        public CollisionDetectionMode detection = CollisionDetectionMode.ContinuousDynamic;

        [Header("Walk test")]
        public float capsuleRadius = 0.2f;
        public float capsuleHeight = 1.0f;
        public float stepOffset = 0.15f;
        public float slopeLimit = 45f;
        public float walkSpeed = 1.0f;
        public float walkSeconds = 12f;

        [Header("Common")]
        public float fixedDt = 0.01f;
        public float fellBelowY = -0.5f;     // the box floor is y = 0
        const int ProbeLayer = 31;
        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        IEnumerator Start()
        {
            string outDir = string.IsNullOrEmpty(outputFolder) ? Application.persistentDataPath : outputFolder;
            Directory.CreateDirectory(outDir);
            Physics.IgnoreLayerCollision(ProbeLayer, ProbeLayer);   // probes do not hit each other
            var drops = ReadCsv(PpmMesh.DataPath(dataFolder, "drops.csv"));
            var walks = ReadCsv(PpmMesh.DataPath(dataFolder, "walks.csv"));
            foreach (var name in colliders)
            {
                var mesh = PpmMesh.Load(PpmMesh.DataPath(dataFolder, name + ".ppm"), name);
                yield return RunDrops(name, mesh, drops, Path.Combine(outDir, $"drops_{name}.csv"));
                yield return RunWalks(name, mesh, walks, Path.Combine(outDir, $"walks_{name}.csv"));
            }
            Debug.Log($"[physproxy] benchmark done; CSVs in {outDir}");
        }

        Scene NewScene(string tag, Mesh mesh)
        {
            var scene = SceneManager.CreateScene($"pp_{tag}_{Time.frameCount}", new CreateSceneParameters(LocalPhysicsMode.Physics3D));
            var go = new GameObject("collider");
            SceneManager.MoveGameObjectToScene(go, scene);
            go.AddComponent<MeshCollider>().sharedMesh = mesh;
            return scene;
        }

        IEnumerator RunDrops(string name, Mesh mesh, List<Dictionary<string, float>> drops, string path)
        {
            var sb = new StringBuilder("id,start_x,start_y,start_z,ref_land_y,started_overlapping,end_x,end_y,end_z,rest_minus_ref,drift_xz,fell_through,end_speed\n");
            int fell = 0;
            for (int b0 = 0; b0 < drops.Count; b0 += dropBatch)
            {
                var scene = NewScene(name + "_drop", mesh);
                var ps = scene.GetPhysicsScene();
                Physics.SyncTransforms();
                var bodies = new List<(Dictionary<string, float> row, Rigidbody rb, bool overlap)>();
                var hits = new Collider[4];
                for (int i = b0; i < Mathf.Min(b0 + dropBatch, drops.Count); i++)
                {
                    var r = drops[i];
                    var p = new Vector3(r["x"], r["y"], r["z"]);
                    bool overlap = ps.OverlapSphere(p, sphereRadius, hits, ~(1 << ProbeLayer), QueryTriggerInteraction.Ignore) > 0;
                    var go = new GameObject("probe");
                    SceneManager.MoveGameObjectToScene(go, scene);
                    go.layer = ProbeLayer;
                    go.transform.position = p;
                    go.AddComponent<SphereCollider>().radius = sphereRadius;
                    var rb = go.AddComponent<Rigidbody>();
                    rb.collisionDetectionMode = detection;
                    rb.interpolation = RigidbodyInterpolation.None;
                    bodies.Add((r, rb, overlap));
                }
                int steps = Mathf.RoundToInt(dropSeconds / fixedDt);
                for (int s = 0; s < steps; s++) ps.Simulate(fixedDt);
                foreach (var (r, rb, overlap) in bodies)
                {
                    var e = rb.position;
                    bool f = e.y < fellBelowY;
                    if (f) fell++;
                    float rest = e.y - sphereRadius - r["ref_land_y"];
                    float drift = new Vector2(e.x - r["x"], e.z - r["z"]).magnitude;
                    sb.AppendLine(string.Join(",", F(r["id"]), F(r["x"]), F(r["y"]), F(r["z"]), F(r["ref_land_y"]), overlap ? "1" : "0",
                        F(e.x), F(e.y), F(e.z), F(rest), F(drift), f ? "1" : "0", F(rb.velocity.magnitude)));
                }
                yield return SceneManager.UnloadSceneAsync(scene);
            }
            File.WriteAllText(path, sb.ToString());
            Debug.Log($"[physproxy] drop {name}: {fell}/{drops.Count} fell through -> {path}");
        }

        IEnumerator RunWalks(string name, Mesh mesh, List<Dictionary<string, float>> walks, string path)
        {
            var sb = new StringBuilder("id,reached,stuck,fell_through,time,final_dist_xz,path_len,max_y_above_start\n");
            int reached = 0, stuck = 0, fell = 0;
            var scene = NewScene(name + "_walk", mesh);
            Physics.SyncTransforms();
            float g = Physics.gravity.y;
            int steps = Mathf.RoundToInt(walkSeconds / fixedDt);
            int window = Mathf.RoundToInt(2f / fixedDt);
            foreach (var w in walks)
            {
                var start = new Vector3(w["sx"], w["sy"], w["sz"]);
                var goal = new Vector3(w["gx"], w["gy"], w["gz"]);
                var go = new GameObject("walker");
                SceneManager.MoveGameObjectToScene(go, scene);
                go.layer = ProbeLayer;
                var cc = go.AddComponent<CharacterController>();
                cc.radius = capsuleRadius; cc.height = capsuleHeight; cc.stepOffset = stepOffset;
                cc.slopeLimit = slopeLimit; cc.skinWidth = 0.02f; cc.center = Vector3.zero;
                go.transform.position = start + Vector3.up * (capsuleHeight * 0.5f + cc.skinWidth + 0.05f);
                Physics.SyncTransforms();
                float vy = 0f, t = 0f, pathLen = 0f, maxUp = 0f;
                bool ok = false, isStuck = false, isFell = false;
                var hist = new Queue<float>();
                for (int s = 0; s < steps; s++)
                {
                    var pos = go.transform.position;
                    var to = goal - pos; to.y = 0f;
                    float d = to.magnitude;
                    if (d < capsuleRadius) { ok = true; break; }
                    if (pos.y < fellBelowY) { isFell = true; break; }
                    hist.Enqueue(d);
                    if (hist.Count > window && hist.Dequeue() - d < 0.05f) { isStuck = true; break; }
                    vy = cc.isGrounded ? -1f : vy + g * fixedDt;
                    var move = to / d * walkSpeed;
                    move.y = vy;
                    cc.Move(move * fixedDt);
                    var np = go.transform.position;
                    pathLen += (np - pos).magnitude;
                    maxUp = Mathf.Max(maxUp, np.y - (start.y + capsuleHeight * 0.5f));
                    t += fixedDt;
                }
                var fp = go.transform.position;
                if (ok) reached++; if (isStuck) stuck++; if (isFell) fell++;
                sb.AppendLine(string.Join(",", F(w["id"]), ok ? "1" : "0", isStuck ? "1" : "0", isFell ? "1" : "0", F(t),
                    F(new Vector2(goal.x - fp.x, goal.z - fp.z).magnitude), F(pathLen), F(maxUp)));
                Destroy(go);
            }
            yield return SceneManager.UnloadSceneAsync(scene);
            File.WriteAllText(path, sb.ToString());
            Debug.Log($"[physproxy] walk {name}: reached {reached}, stuck {stuck}, fell {fell} of {walks.Count} -> {path}");
        }

        static string F(float v) => v.ToString("0.#####", Inv);

        static List<Dictionary<string, float>> ReadCsv(string path)
        {
            var lines = File.ReadAllLines(path);
            var head = lines[0].Split(',');
            var rows = new List<Dictionary<string, float>>();
            for (int i = 1; i < lines.Length; i++)
            {
                if (string.IsNullOrWhiteSpace(lines[i])) continue;
                var c = lines[i].Split(',');
                var r = new Dictionary<string, float>();
                for (int k = 0; k < head.Length; k++) r[head[k]] = float.Parse(c[k], Inv);
                rows.Add(r);
            }
            return rows;
        }
    }
}
