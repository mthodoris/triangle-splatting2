// physproxy Phase 7: interactive viewer. Render mesh as the visual, one collider as a static MeshCollider.
// Play mode: Space throws a ball from the main camera; C toggles showing the collider; 1/2/3 switch collider.
using System.Collections.Generic;
using UnityEngine;

namespace PhysProxy
{
    public class PhysProxyViewer : MonoBehaviour
    {
        [Tooltip("Folder under Assets/StreamingAssets holding render.ppm, collision.ppm, ...")]
        public string dataFolder = "physproxy/garden";
        public string[] colliders = { "collision", "naive", "soup" };
        public int activeCollider = 0;
        public float ballRadius = 0.08f;
        public float throwSpeed = 4f;

        GameObject visual, colliderGo;
        MeshRenderer colliderRenderer;
        readonly Dictionary<string, Mesh> cache = new Dictionary<string, Mesh>();
        Material vcMat, colliderMat;
        readonly List<GameObject> balls = new List<GameObject>();

        void Start()
        {
            var shader = Shader.Find("PhysProxy/VertexColorUnlit");
            vcMat = new Material(shader);
            colliderMat = new Material(shader);
            colliderMat.SetColor("_Tint", new Color(0.4f, 0.85f, 0.55f, 1f));

            visual = new GameObject("render (visual)");
            visual.AddComponent<MeshFilter>().sharedMesh = Get("render");
            visual.AddComponent<MeshRenderer>().sharedMaterial = vcMat;

            colliderGo = new GameObject("collider");
            colliderGo.AddComponent<MeshFilter>();
            colliderRenderer = colliderGo.AddComponent<MeshRenderer>();
            colliderRenderer.sharedMaterial = colliderMat;
            colliderRenderer.enabled = false;
            colliderGo.AddComponent<MeshCollider>();
            Use(activeCollider);
        }

        Mesh Get(string name)
        {
            if (!cache.TryGetValue(name, out var m))
            {
                m = PpmMesh.Load(PpmMesh.DataPath(dataFolder, name + ".ppm"), name);
                if (name != "render")   // colliders have no colors: paint them white so the tint shows
                {
                    var c = new Color32[m.vertexCount];
                    for (int i = 0; i < c.Length; i++) c[i] = new Color32(255, 255, 255, 255);
                    m.SetColors(c);
                }
                cache[name] = m;
            }
            return m;
        }

        void Use(int i)
        {
            activeCollider = Mathf.Clamp(i, 0, colliders.Length - 1);
            var m = Get(colliders[activeCollider]);
            colliderGo.GetComponent<MeshFilter>().sharedMesh = m;
            colliderGo.GetComponent<MeshCollider>().sharedMesh = m;
            colliderGo.name = "collider: " + colliders[activeCollider];
            Debug.Log($"[physproxy] collider = {colliders[activeCollider]} ({m.triangles.Length / 3} triangles)");
        }

        void Update()
        {
            if (Input.GetKeyDown(KeyCode.C)) colliderRenderer.enabled = !colliderRenderer.enabled;
            for (int k = 0; k < colliders.Length && k < 9; k++)
                if (Input.GetKeyDown(KeyCode.Alpha1 + k)) Use(k);
            if (Input.GetKeyDown(KeyCode.Space) && Camera.main != null)
            {
                var b = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                b.transform.localScale = Vector3.one * (2 * ballRadius);
                b.transform.position = Camera.main.transform.position + Camera.main.transform.forward * 0.5f;
                var rb = b.AddComponent<Rigidbody>();
                rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
                rb.velocity = Camera.main.transform.forward * throwSpeed;
                balls.Add(b);
                if (balls.Count > 200) { Destroy(balls[0]); balls.RemoveAt(0); }
            }
        }
    }
}
