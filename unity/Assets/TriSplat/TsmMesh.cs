// Loader for render.tsm written by combined/scripts/export_unity.py (format in that file's docstring).
// Returns the mesh (positions, atlas UV in uv0, source vertex id in uv1.x) and the packed float16 SH words.
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace TriSplat
{
    public class TsmData
    {
        public Mesh mesh;
        public uint[] shWords;     // float16 pairs, [source vertex][coeff][rgb]
        public int shCoeffs;
        public int atlasW, atlasH;
    }

    public static class TsmMesh
    {
        public static TsmData Load(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            int o = 0;
            if (bytes[0] != 'T' || bytes[1] != 'S' || bytes[2] != 'M' || bytes[3] != '1')
                throw new InvalidDataException($"{path}: not a TSM1 file");
            o = 4;
            int nV = BitConverter.ToInt32(bytes, o), nT = BitConverter.ToInt32(bytes, o + 4);
            int K = BitConverter.ToInt32(bytes, o + 8), aw = BitConverter.ToInt32(bytes, o + 12), ah = BitConverter.ToInt32(bytes, o + 16);
            o += 20;
            int nC = nT * 3;

            var pos = new float[nC * 3]; Buffer.BlockCopy(bytes, o, pos, 0, nC * 12); o += nC * 12;
            var uv = new float[nC * 2]; Buffer.BlockCopy(bytes, o, uv, 0, nC * 8); o += nC * 8;
            var src = new int[nC]; Buffer.BlockCopy(bytes, o, src, 0, nC * 4); o += nC * 4;
            int nHalf = nV * K * 3;
            var words = new uint[(nHalf + 1) / 2];
            Buffer.BlockCopy(bytes, o, words, 0, nHalf * 2);

            var verts = new Vector3[nC]; var uv0 = new Vector2[nC]; var uv1 = new Vector2[nC]; var idx = new int[nC];
            for (int i = 0; i < nC; i++)
            {
                verts[i] = new Vector3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
                uv0[i] = new Vector2(uv[2 * i], uv[2 * i + 1]);
                uv1[i] = new Vector2(src[i], 0f);      // exact as float up to 16.7M vertices
                idx[i] = i;
            }
            var mesh = new Mesh { name = Path.GetFileNameWithoutExtension(path), indexFormat = IndexFormat.UInt32 };
            mesh.SetVertices(verts);
            mesh.SetUVs(0, uv0);
            mesh.SetUVs(1, uv1);
            mesh.SetIndices(idx, MeshTopology.Triangles, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return new TsmData { mesh = mesh, shWords = words, shCoeffs = K, atlasW = aw, atlasH = ah };
        }
    }
}
