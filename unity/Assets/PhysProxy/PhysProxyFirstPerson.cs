// physproxy: first-person walker. Add next to PhysProxyViewer (same GameObject).
// F toggles first person. WASD move, mouse look, Shift run, Space jump, left click throws a ball,
// Esc frees the mouse (click to capture again). The body is a CharacterController, so it collides with
// whichever collider the viewer has active (1/2/3) - you walk on exactly what physics sees.
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace PhysProxy
{
    [RequireComponent(typeof(PhysProxyViewer))]
    public class PhysProxyFirstPerson : MonoBehaviour
    {
        public bool startInFirstPerson = false;
        [Header("Body (metres)")]
        public float height = 1.7f;
        public float radius = 0.25f;
        public float eyeHeight = 1.6f;          // above the feet
        public float stepOffset = 0.3f;
        public float slopeLimit = 45f;
        [Header("Motion")]
        public float walkSpeed = 1.5f;
        public float runSpeed = 3.5f;
        public float jumpSpeed = 3.5f;
        public float mouseSensitivity = 0.1f;   // degrees per pixel of mouse motion
        [Tooltip("Spawn on the ground below this point (x, z in Unity metres; y is ignored)")]
        public Vector2 spawnXZ = new Vector2(0f, -2.5f);
        public float throwSpeed = 6f;

        PhysProxyViewer viewer;
        CharacterController body;
        Transform cam;
        Vector3 camPos0; Quaternion camRot0; Transform camParent0; float camNear0;
        float yaw, pitch, vy;
        bool active;

        void Start()
        {
            viewer = GetComponent<PhysProxyViewer>();
            if (startInFirstPerson) Invoke(nameof(Enter), 0.1f);   // after the viewer has built its collider
        }

        void Enter()
        {
            if (Camera.main == null) { Debug.LogWarning("[physproxy] first person needs a Main Camera"); return; }
            cam = Camera.main.transform;
            camPos0 = cam.position; camRot0 = cam.rotation; camParent0 = cam.parent; camNear0 = Camera.main.nearClipPlane;
            if (body == null)
            {
                var go = new GameObject("first person");
                body = go.AddComponent<CharacterController>();
                body.height = height; body.radius = radius; body.stepOffset = stepOffset; body.slopeLimit = slopeLimit;
                body.skinWidth = 0.02f; body.center = new Vector3(0f, height / 2f, 0f);   // transform = feet
            }
            body.enabled = false;
            body.transform.position = GroundBelow(spawnXZ);
            body.transform.rotation = Quaternion.identity;
            body.enabled = true;
            yaw = 0f; pitch = 0f; vy = 0f;
            cam.SetParent(body.transform, false);
            cam.localPosition = new Vector3(0f, eyeHeight, 0f);
            cam.localRotation = Quaternion.identity;
            Camera.main.nearClipPlane = 0.02f;
            active = viewer.firstPersonActive = true;
            Lock(true);
            Debug.Log("[physproxy] first person: WASD move, mouse look, Shift run, Space jump, click throws, F exits");
        }

        void Exit()
        {
            active = viewer.firstPersonActive = false;
            cam.SetParent(camParent0, false);
            cam.SetPositionAndRotation(camPos0, camRot0);
            Camera.main.nearClipPlane = camNear0;
            Lock(false);
        }

        Vector3 GroundBelow(Vector2 xz)
        {
            var top = new Vector3(xz.x, viewer.BoxSize.y + 2f, xz.y);
            return Physics.Raycast(top, Vector3.down, out var hit, 100f) ? hit.point + Vector3.up * 0.05f
                                                                          : new Vector3(xz.x, 0.5f, xz.y);
        }

        static void Lock(bool on)
        {
            Cursor.lockState = on ? CursorLockMode.Locked : CursorLockMode.None;
            Cursor.visible = !on;
        }

        void Update()
        {
            if (Pressed(KeyF())) { if (active) Exit(); else Enter(); }
            if (!active) return;
            if (KeyEsc()) Lock(false);
            bool locked = Cursor.lockState == CursorLockMode.Locked;
            if (Click())
            {
                if (!locked) Lock(true);
                else viewer.Spawn(cam.position + cam.forward * 0.4f, cam.forward * throwSpeed,
                                  Color.HSVToRGB(Random.value, 0.8f, 1f), viewer.ballRadius);
            }

            if (locked)
            {
                var d = MouseDelta();
                yaw += d.x * mouseSensitivity;
                pitch = Mathf.Clamp(pitch - d.y * mouseSensitivity, -89f, 89f);
                body.transform.rotation = Quaternion.Euler(0f, yaw, 0f);
                cam.localRotation = Quaternion.Euler(pitch, 0f, 0f);
            }

            var m = Move();
            var dir = body.transform.right * m.x + body.transform.forward * m.y;
            if (dir.sqrMagnitude > 1f) dir.Normalize();
            float speed = Run() ? runSpeed : walkSpeed;
            if (body.isGrounded) vy = Jump() ? jumpSpeed : -1f;
            else vy += Physics.gravity.y * Time.deltaTime;
            body.Move((dir * speed + Vector3.up * vy) * Time.deltaTime);

            if (body.transform.position.y < -1f)          // fell through or off the edge: respawn
            {
                Debug.Log("[physproxy] first person fell below the scene, respawning");
                body.enabled = false; body.transform.position = GroundBelow(spawnXZ); body.enabled = true; vy = 0f;
            }
        }

        // ---- input, both the new Input System and the legacy Input Manager ----
        static bool Pressed(bool b) => b;
#if ENABLE_INPUT_SYSTEM
        static Keyboard K => Keyboard.current;
        static bool KeyF() => K != null && K.fKey.wasPressedThisFrame;
        static bool KeyEsc() => K != null && K.escapeKey.wasPressedThisFrame;
        static bool Jump() => K != null && K.spaceKey.wasPressedThisFrame;
        static bool Run() => K != null && K.leftShiftKey.isPressed;
        static bool Click() => Mouse.current != null && Mouse.current.leftButton.wasPressedThisFrame;
        static Vector2 MouseDelta() => Mouse.current != null ? Mouse.current.delta.ReadValue() : Vector2.zero;
        static Vector2 Move()
        {
            if (K == null) return Vector2.zero;
            return new Vector2((K.dKey.isPressed ? 1 : 0) - (K.aKey.isPressed ? 1 : 0),
                               (K.wKey.isPressed ? 1 : 0) - (K.sKey.isPressed ? 1 : 0));
        }
#else
        static bool KeyF() => Input.GetKeyDown(KeyCode.F);
        static bool KeyEsc() => Input.GetKeyDown(KeyCode.Escape);
        static bool Jump() => Input.GetKeyDown(KeyCode.Space);
        static bool Run() => Input.GetKey(KeyCode.LeftShift);
        static bool Click() => Input.GetMouseButtonDown(0);
        static Vector2 MouseDelta() => new Vector2(Input.GetAxisRaw("Mouse X"), Input.GetAxisRaw("Mouse Y")) * 10f;
        static Vector2 Move() => new Vector2(Input.GetAxisRaw("Horizontal"), Input.GetAxisRaw("Vertical"));
#endif
    }
}
