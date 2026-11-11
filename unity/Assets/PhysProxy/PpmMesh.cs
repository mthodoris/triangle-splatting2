// physproxy Phase 7: loader for the .ppm meshes written by unity/export_for_unity.py.
// Format (little endian): "PPM1", int32 nV, int32 nF, int32 hasColor,
// float32[nV*3] positions (already in Unity's frame, metres), uint8[nV*4] RGBA if hasColor, int32[nF*3] indices.
using System;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace PhysProxy
{
    public static class PpmMesh
    {
        public static Mesh Load(string path, string name = null)
        {
            byte[] bytes = File.ReadAllBytes(path);
            using (var br = new BinaryReader(new MemoryStream(bytes)))
            {
                var magic = new string(br.ReadChars(4));
                if (magic != "PPM1") throw new InvalidDataException($"{path}: not a PPM1 mesh");
                int nV = br.ReadInt32(), nF = br.ReadInt32();
                bool hasColor = br.ReadInt32() != 0;
                var verts = new Vector3[nV];
                for (int i = 0; i < nV; i++) verts[i] = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                Color32[] colors = null;
                if (hasColor)
                {
                    colors = new Color32[nV];
                    for (int i = 0; i < nV; i++)
                        colors[i] = new Color32(br.ReadByte(), br.ReadByte(), br.ReadByte(), br.ReadByte());
                }
                var idx = new int[nF * 3];
                for (int i = 0; i < idx.Length; i++) idx[i] = br.ReadInt32();

                var mesh = new Mesh { name = name ?? Path.GetFileNameWithoutExtension(path), indexFormat = IndexFormat.UInt32 };
                mesh.SetVertices(verts);
                if (colors != null) mesh.SetColors(colors);
                mesh.SetIndices(idx, MeshTopology.Triangles, 0);
                mesh.RecalculateNormals();
                mesh.RecalculateBounds();
                return mesh;
            }
        }

        public static string DataPath(string folder, string file) =>
            Path.Combine(Application.streamingAssetsPath, folder, file);
    }
}
