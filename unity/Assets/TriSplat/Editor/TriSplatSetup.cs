// Creates Assets/Scenes/TriSplatViewer.unity: one camera + one "TriSplat" object (textured viewer + physproxy
// physics viewer + first person), ready for Play.
// Menu: TriSplat > Create viewer scene. Batch: -executeMethod TriSplat.TriSplatSetup.CreateScene
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace TriSplat
{
    public static class TriSplatSetup
    {
        [MenuItem("TriSplat/Create viewer scene")]
        public static void CreateScene()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = Color.black;
            camGo.AddComponent<AudioListener>();
            var go = new GameObject("TriSplat");
            var v = go.AddComponent<TriSplatViewer>();
            v.sceneFolder = "trisplat/garden_tex8wd1_mp300k";
            v.shader = AssetDatabase.LoadAssetAtPath<Shader>("Assets/TriSplat/SHAtlas.shader");
            // Physics (physproxy viewer: balls, rain, collider display, first person).
            var pv = go.AddComponent<PhysProxy.PhysProxyViewer>();
            pv.dataFolder = v.sceneFolder + "/physics";
            pv.drawRenderMesh = false;           // the textured TriSplat mesh is the visual
            // colliders 1/2/3 are set by TriSplatViewer.physicsColliders at start (soup / SDF only / SDF + soup outside)
            pv.ballRadius = 0.05f;
            go.AddComponent<PhysProxy.PhysProxyFirstPerson>();
            EditorSceneManager.SaveScene(scene, "Assets/Scenes/TriSplatViewer.unity");
            Debug.Log("[TriSplatSetup] wrote Assets/Scenes/TriSplatViewer.unity");
        }
    }
}
