// physproxy Phase 7: interactive viewer. Render mesh as the visual, one collider as a static MeshCollider.
// Play mode: Space throws a ball from the main camera; R rains balls over the whole box;
// C toggles showing the collider; 1/2/3 switch collider.
using System.Collections.Generic;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

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
        [Tooltip("0 = no bounce, 1 = perfectly elastic")]
        [Range(0f, 1f)] public float ballBounciness = 0.3f;

        [Header("Rain (R key)")]
        public int rainCount = 50;
        [Tooltip("Balls start between top-of-box + this and + this + rainSpread (m)")]
        public float rainHeightAboveBox = 0.5f;
        public float rainSpread = 1.0f;
        public bool rainRandomColors = true;
        Vector3 boxSize = new Vector3(6.47f, 2.6f, 5.9f);   // overwritten from meta.json

#if UNITY_6000_0_OR_NEWER
        PhysicsMaterial ballMat;
#else
        PhysicMaterial ballMat;
#endif

        [System.Serializable] class Meta { public float[] box_size_unity; }

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
            try
            {
                var meta = JsonUtility.FromJson<Meta>(System.IO.File.ReadAllText(PpmMesh.DataPath(dataFolder, "meta.json")));
                if (meta.box_size_unity != null && meta.box_size_unity.Length == 3)
                    boxSize = new Vector3(meta.box_size_unity[0], meta.box_size_unity[1], meta.box_size_unity[2]);
            }
            catch (System.Exception e) { Debug.LogWarning($"[physproxy] meta.json not read, using default box size: {e.Message}"); }
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

        // Works with the new Input System (Unity 6 default) and with the legacy Input Manager.
        static bool KeyC() =>
#if ENABLE_INPUT_SYSTEM
            Keyboard.current != null && Keyboard.current.cKey.wasPressedThisFrame;
#else
            Input.GetKeyDown(KeyCode.C);
#endif
        static bool KeySpace() =>
#if ENABLE_INPUT_SYSTEM
            Keyboard.current != null && Keyboard.current.spaceKey.wasPressedThisFrame;
#else
            Input.GetKeyDown(KeyCode.Space);
#endif
        static bool KeyR() =>
#if ENABLE_INPUT_SYSTEM
            Keyboard.current != null && Keyboard.current.rKey.wasPressedThisFrame;
#else
            Input.GetKeyDown(KeyCode.R);
#endif
        static bool KeyDigit(int k) =>   // k = 0 -> "1"
#if ENABLE_INPUT_SYSTEM
            Keyboard.current != null && Keyboard.current[(Key)((int)Key.Digit1 + k)].wasPressedThisFrame;
#else
            Input.GetKeyDown(KeyCode.Alpha1 + k);
#endif

        void Update()
        {
            if (KeyC()) colliderRenderer.enabled = !colliderRenderer.enabled;
            for (int k = 0; k < colliders.Length && k < 9; k++)
                if (KeyDigit(k)) Use(k);
            if (KeySpace() && Camera.main != null)
                Spawn(Camera.main.transform.position + Camera.main.transform.forward * 0.5f,
                      Camera.main.transform.forward * throwSpeed, Color.white);
            if (KeyR()) Rain();
        }

        // Box in Unity frame: x in [-sx/2, sx/2], z in [-sz/2, sz/2], y in [0, sy].
        void Rain()
        {
            for (int i = 0; i < rainCount; i++)
            {
                var p = new Vector3(Random.Range(-0.5f, 0.5f) * boxSize.x,
                                    boxSize.y + rainHeightAboveBox + Random.Range(0f, rainSpread),
                                    Random.Range(-0.5f, 0.5f) * boxSize.z);
                var c = rainRandomColors ? Color.HSVToRGB(Random.value, 0.8f, 1f) : Color.white;
                Spawn(p, Vector3.zero, c);
            }
            Debug.Log($"[physproxy] rain: {rainCount} balls over {boxSize.x:0.0} x {boxSize.z:0.0} m");
        }

        void Spawn(Vector3 pos, Vector3 vel, Color color)
        {
            var b = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            b.transform.localScale = Vector3.one * (2 * ballRadius);
            b.transform.position = pos;
            b.GetComponent<Renderer>().material.color = color;
            if (ballMat == null)
            {
#if UNITY_6000_0_OR_NEWER
                ballMat = new PhysicsMaterial("ball") { bounceCombine = PhysicsMaterialCombine.Maximum };
#else
                ballMat = new PhysicMaterial("ball") { bounceCombine = PhysicMaterialCombine.Maximum };
#endif
            }
            ballMat.bounciness = ballBounciness;   // Maximum: the collider's default 0 does not cancel it
            b.GetComponent<Collider>().sharedMaterial = ballMat;
            var rb = b.AddComponent<Rigidbody>();
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
#if UNITY_6000_0_OR_NEWER
            rb.linearVelocity = vel;
#else
            rb.velocity = vel;
#endif
            balls.Add(b);
            while (balls.Count > 300) { Destroy(balls[0]); balls.RemoveAt(0); }
        }
    }
}
