// Shows an export_unity.py scene (plan step 9). Put this on an empty GameObject and press Play.
// Files: Assets/StreamingAssets/<sceneFolder>/{render.tsm, atlas.png, meta.json, cameras.json}.
//
// Controls (Play mode):
//   hold right mouse + move mouse   look around          W A S D / Q E   move / down, up      Shift   faster
//   V / B                          next / previous photo camera (exact training-photo viewpoint)
//   Space                          throw a ball (needs a collider: "Soup Collider" on, or the SDF collider later)
//   H                              hide / show this help
using System.IO;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
#endif

namespace TriSplat
{
    public class TriSplatViewer : MonoBehaviour
    {
        [Tooltip("Folder inside Assets/StreamingAssets")]
        public string sceneFolder = "trisplat/garden";
        [Tooltip("Leave empty: found automatically (TriSplat/SHAtlas)")]
        public Shader shader;
        [Tooltip("Use the rendered triangles themselves as the collider (triangle soup). Has gaps: small balls can fall through.")]
        public bool soupCollider = false;
        [Tooltip("Colliders for keys 1/2/3 (set on the PhysProxyViewer of this object at start): 1 = soup, all triangles; " +
                 "2 = SDF collision mesh only (inside the proxy box); 3 = SDF mesh + triangles outside the box")]
        public string[] physicsColliders = { "soup_all", "collision", "collision_ext" };
        [Tooltip("Collider active at start: 0 = key 1, 1 = key 2, 2 = key 3")]
        public int startCollider = 2;
        public float moveSpeed = 1.5f, lookSpeed = 0.15f, ballRadius = 0.05f, throwSpeed = 6f;

        [System.Serializable] class Cam { public string name; public float[] pos, forward, up; public float fovY, aspect, fx_n, fy_n; }
        [System.Serializable] class CamList { public Cam[] cams; }

        GraphicsBuffer shBuffer;
        Camera cam;
        CamList cams;
        int camIndex = -1;
        float yaw, pitch;
        bool help = true;
        string status = "";

        void Awake()
        {
            // Runs before PhysProxyViewer.Start, so these override whatever the scene file stored.
            var pp = GetComponent<PhysProxy.PhysProxyViewer>();
            if (pp != null && physicsColliders != null && physicsColliders.Length > 0)
            {
                pp.colliders = physicsColliders;
                pp.activeCollider = Mathf.Clamp(startCollider, 0, physicsColliders.Length - 1);
            }
        }

        void Start()
        {
            string dir = Path.Combine(Application.streamingAssetsPath, sceneFolder);
            var t0 = Time.realtimeSinceStartup;
            var go = TriSplatBuilder.Build(dir, shader, transform, out shBuffer, out var info);
            if (soupCollider) go.AddComponent<MeshCollider>().sharedMesh = go.GetComponent<MeshFilter>().sharedMesh;

            cam = Camera.main;
            if (cam == null) cam = new GameObject("Main Camera").AddComponent<Camera>();
            cam.nearClipPlane = 0.05f; cam.farClipPlane = 300f;
            cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = Color.black;   // TS+ background
            string cf = Path.Combine(dir, "cameras.json");
            if (File.Exists(cf)) cams = JsonUtility.FromJson<CamList>(File.ReadAllText(cf));
            if (cams != null && cams.cams.Length > 0) GoToCam(0);
            status = info +
                     $"loaded in {Time.realtimeSinceStartup - t0:F1} s" + (soupCollider ? ", soup collider ON" : "");
            Debug.Log("[TriSplat] " + status);
        }

        void GoToCam(int i)
        {
            var c = cams.cams[(i % cams.cams.Length + cams.cams.Length) % cams.cams.Length];
            camIndex = System.Array.IndexOf(cams.cams, c);
            cam.transform.position = transform.TransformPoint(new Vector3(c.pos[0], c.pos[1], c.pos[2]));
            cam.transform.rotation = transform.rotation * Quaternion.LookRotation(
                new Vector3(c.forward[0], c.forward[1], c.forward[2]), new Vector3(c.up[0], c.up[1], c.up[2]));
            cam.ResetProjectionMatrix(); cam.fieldOfView = c.fovY;   // window aspect may differ from the photo
            var e = cam.transform.eulerAngles; yaw = e.y; pitch = Mathf.Clamp(e.x > 180 ? e.x - 360 : e.x, -89f, 89f);
        }

        void Update()
        {
            if (cam == null) return;
            var pp = GetComponent<PhysProxy.PhysProxyViewer>();          // physics viewer on the same object (optional)
            if (pp != null && pp.firstPersonActive) return;               // first person (F) controls the camera
            if (Down("h")) help = !help;
            if (cams != null && Down("v")) GoToCam(camIndex + 1);
            if (cams != null && Down("b")) GoToCam(camIndex - 1);
            if (RightMouse())
            {
                var d = MouseDelta();
                if (float.IsFinite(d.x) && float.IsFinite(d.y))
                {
                    yaw = Mathf.Repeat(yaw + d.x * lookSpeed, 360f);
                    pitch = Mathf.Clamp(pitch - d.y * lookSpeed, -89f, 89f);
                }
                if (!float.IsFinite(yaw) || !float.IsFinite(pitch)) { yaw = 0f; pitch = 0f; }   // never pass NaN to Euler
                cam.transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
            }
            var mv = new Vector3(Axis("d", "a"), Axis("e", "q"), Axis("w", "s"));
            float sp = moveSpeed * (Held("leftShift") ? 4f : 1f);
            cam.transform.position += cam.transform.TransformDirection(mv) * sp * Time.deltaTime;
            if (pp == null && Down("space")) Throw();                     // with physproxy, its Space/R throw balls
        }

        void Throw()
        {
            var b = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            b.transform.position = cam.transform.position + cam.transform.forward * 0.3f;
            b.transform.localScale = Vector3.one * 2f * ballRadius;
            var rb = b.AddComponent<Rigidbody>();
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
#if UNITY_6000_0_OR_NEWER
            rb.linearVelocity = cam.transform.forward * throwSpeed;
#else
            rb.velocity = cam.transform.forward * throwSpeed;
#endif
            b.GetComponent<Renderer>().material.color = Color.HSVToRGB(Random.value, 0.8f, 1f);
            Destroy(b, 20f);
        }

        void OnGUI()
        {
            if (!help) return;
            string c = (cams != null && camIndex >= 0) ? $"photo camera {camIndex + 1}/{cams.cams.Length} ({cams.cams[camIndex].name})" : "";
            GUI.Box(new Rect(10, 50, 700, 130), "");
            GUI.Label(new Rect(20, 55, 680, 125),
                "TriSplat viewer   " + status + "\n" +
                "Right mouse + move: look    WASD / QE: move    Shift: faster\n" +
                "V / B: next / previous photo camera   " + c + "\n" +
                (GetComponent<PhysProxy.PhysProxyViewer>() != null
                    ? "1: soup (all triangles)   2: SDF only   3: SDF + soup outside   C: show collider\n" +
                      "Space: throw ball   R: ball rain   F: walk (first person)   H: hide help"
                    : "Space: throw a ball    H: hide this help"));
        }

        void OnDestroy() { shBuffer?.Release(); }

        // ---- input: new Input System (this project) or the legacy Input Manager ----
#if ENABLE_INPUT_SYSTEM
        static KeyControl KeyOf(string k) => Keyboard.current == null ? null : (KeyControl)Keyboard.current[k];
        static bool Down(string k) { var c = KeyOf(k); return c != null && c.wasPressedThisFrame; }
        static bool Held(string k) { var c = KeyOf(k); return c != null && c.isPressed; }
        static bool RightMouse() => Mouse.current != null && Mouse.current.rightButton.isPressed;
        static Vector2 MouseDelta() => Mouse.current != null ? Mouse.current.delta.ReadValue() : Vector2.zero;
#else
        static KeyCode Code(string k) => k == "leftShift" ? KeyCode.LeftShift : k == "space" ? KeyCode.Space
            : (KeyCode)System.Enum.Parse(typeof(KeyCode), k.ToUpper());
        static bool Down(string k) => Input.GetKeyDown(Code(k));
        static bool Held(string k) => Input.GetKey(Code(k));
        static bool RightMouse() => Input.GetMouseButton(1);
        static Vector2 MouseDelta() => new Vector2(Input.GetAxisRaw("Mouse X"), Input.GetAxisRaw("Mouse Y")) * 10f;
#endif
        static float Axis(string pos, string neg) => (Held(pos) ? 1f : 0f) - (Held(neg) ? 1f : 0f);
    }
}
