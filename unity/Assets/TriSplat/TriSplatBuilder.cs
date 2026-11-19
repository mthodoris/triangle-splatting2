// Builds the render object of an export_unity.py folder: mesh + TriSplat/SHAtlas material (atlas, SH buffer, frame).
// Used by TriSplatViewer (Play mode) and by the batch-mode render test (Editor/TriSplatBatchTest.cs).
using System.IO;
using UnityEngine;

namespace TriSplat
{
    public static class TriSplatBuilder
    {
        /// Exact pinhole projection of a photo camera (fx != fy, centred principal point), as TS+ renders it.
        /// fxN = fx / width, fyN = fy / height. Falls back to Unity's own FOV if fxN is 0 (old cameras.json).
        public static void SetPhotoProjection(Camera cam, float fxN, float fyN, float fovY, float aspect)
        {
            cam.ResetProjectionMatrix();
            cam.fieldOfView = fovY; cam.aspect = aspect;
            if (fxN <= 0) return;
            float n = cam.nearClipPlane, f = cam.farClipPlane;
            var P = Matrix4x4.zero;
            P[0, 0] = 2f * fxN; P[1, 1] = 2f * fyN;
            P[2, 2] = -(f + n) / (f - n); P[2, 3] = -2f * f * n / (f - n); P[3, 2] = -1f;
            cam.projectionMatrix = P;
        }

        [System.Serializable] class Meta { public float[] scene_dir_from_unity_flat; public int n_triangles; }

        public static GameObject Build(string dir, Shader shader, Transform parent, out GraphicsBuffer shBuffer, out string info)
        {
            var meta = JsonUtility.FromJson<Meta>(File.ReadAllText(Path.Combine(dir, "meta.json")));
            var data = TsmMesh.Load(Path.Combine(dir, "render.tsm"));

            // Atlas as a LINEAR texture = the trained numbers unchanged (the shader converts to linear at the end).
            // Decode, then copy the raw bytes into a texture created as linear, so no sRGB conversion can happen.
            var png = new Texture2D(2, 2, TextureFormat.RGB24, false);
            png.LoadImage(File.ReadAllBytes(Path.Combine(dir, "atlas.png")));
            var atlas = new Texture2D(png.width, png.height, png.format, false, true);
            atlas.SetPixelData(png.GetRawTextureData<byte>(), 0);
            atlas.filterMode = FilterMode.Bilinear;
            atlas.wrapMode = TextureWrapMode.Clamp;
            atlas.Apply(false, true);
            Object.DestroyImmediate(png);

            shBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, data.shWords.Length, 4);
            shBuffer.SetData(data.shWords);
            if (shader == null) shader = Shader.Find("TriSplat/SHAtlas");
            var mat = new Material(shader) { name = "TriSplat SHAtlas" };
            mat.SetTexture("_MainTex", atlas);
            mat.SetBuffer("_SH", shBuffer);
            mat.SetInt("_ShCoeffs", data.shCoeffs);
            var f = meta.scene_dir_from_unity_flat;
            var m = Matrix4x4.identity;
            for (int r = 0; r < 3; r++) m.SetRow(r, new Vector4(f[3 * r], f[3 * r + 1], f[3 * r + 2], 0));
            mat.SetMatrix("_SceneDirFromUnity", m);

            var go = new GameObject("TriSplat render");
            if (parent != null) go.transform.SetParent(parent, false);
            go.AddComponent<MeshFilter>().sharedMesh = data.mesh;
            go.AddComponent<MeshRenderer>().sharedMaterial = mat;
            info = $"{meta.n_triangles:N0} triangles, SH coeffs {data.shCoeffs}, atlas {data.atlasW}x{data.atlasH}";
            return go;
        }
    }
}
