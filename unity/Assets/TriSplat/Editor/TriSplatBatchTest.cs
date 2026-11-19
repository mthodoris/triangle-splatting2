// Batch-mode render test (plan step 9.4): renders test views of an export_unity.py folder with Unity's real
// pipeline and shader, to PNGs, for comparison with combined/scripts/eval_unity_export.py.
//   Unity -batchmode -projectPath <project> -executeMethod TriSplat.TriSplatBatchTest.Run -quit -logFile -
// Environment: TRISPLAT_DIR (export folder), TRISPLAT_OUT (PNG folder), TRISPLAT_VIEWS ("0,5,10"), TRISPLAT_W (width).
using System.IO;
using UnityEditor;
using UnityEngine;

namespace TriSplat
{
    public static class TriSplatBatchTest
    {
        [System.Serializable] class Cam { public string name; public float[] pos, forward, up; public float fovY, aspect, fx_n, fy_n; }
        [System.Serializable] class CamList { public Cam[] cams; }

        public static void Run()
        {
            string dir = System.Environment.GetEnvironmentVariable("TRISPLAT_DIR");
            string outDir = System.Environment.GetEnvironmentVariable("TRISPLAT_OUT");
            string views = System.Environment.GetEnvironmentVariable("TRISPLAT_VIEWS") ?? "0";
            int W = int.Parse(System.Environment.GetEnvironmentVariable("TRISPLAT_W") ?? "1297");
            Directory.CreateDirectory(outDir);
            Debug.Log($"[TriSplatBatchTest] colour space {PlayerSettings.colorSpace}, pipeline {UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline}");
            var shader = AssetDatabase.LoadAssetAtPath<Shader>("Assets/TriSplat/SHAtlas.shader");
            var go = TriSplatBuilder.Build(dir, shader, null, out var buf, out var info);
            Debug.Log("[TriSplatBatchTest] " + info);
            var cams = JsonUtility.FromJson<CamList>(File.ReadAllText(Path.Combine(dir, "cameras.json")));
            var cam = new GameObject("test cam").AddComponent<Camera>();
            cam.nearClipPlane = 0.05f; cam.farClipPlane = 300f;
            cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = Color.black;
            foreach (var s in views.Split(','))
            {
                var c = cams.cams[int.Parse(s)];
                int H = Mathf.RoundToInt(W / c.aspect);
                cam.transform.position = new Vector3(c.pos[0], c.pos[1], c.pos[2]);
                cam.transform.rotation = Quaternion.LookRotation(new Vector3(c.forward[0], c.forward[1], c.forward[2]),
                                                                 new Vector3(c.up[0], c.up[1], c.up[2]));
                TriSplatBuilder.SetPhotoProjection(cam, c.fx_n, c.fy_n, c.fovY, c.aspect);
                var rt = new RenderTexture(W, H, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB) { antiAliasing = 1 };
                cam.targetTexture = rt;
                cam.Render();
                RenderTexture.active = rt;
                var img = new Texture2D(W, H, TextureFormat.RGB24, false);
                img.ReadPixels(new Rect(0, 0, W, H), 0, 0);
                img.Apply();
                RenderTexture.active = null; cam.targetTexture = null;
                File.WriteAllBytes(Path.Combine(outDir, c.name + ".png"), img.EncodeToPNG());
                Debug.Log($"[TriSplatBatchTest] wrote {c.name}.png ({W}x{H})");
                Object.DestroyImmediate(img); rt.Release();
            }
            buf.Release();
            Object.DestroyImmediate(go);
        }
    }
}
