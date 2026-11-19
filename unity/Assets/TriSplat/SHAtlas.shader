// Textured triangles with view-dependent colour (plan step 9.3). Same formula as TS+ / eval_unity_export.py:
//   vertex:   r = max(SH(dir) + 0.5, 0) - max(C0 * dc + 0.5, 0)      dir = vertex - camera, in the SCENE frame
//   fragment: colour = saturate(atlas(uv) + r)                        atlas = view-independent colour + texture
// Unlit (colours come from the photos), Cull Off (TS+ triangles have no reliable facing).
// SH coefficients: StructuredBuffer of packed float16 pairs, [source vertex][coeff][rgb]; the source vertex id is
// in TEXCOORD1.x. The atlas is loaded as a LINEAR texture, so it holds the trained (sRGB) numbers unchanged;
// the sum is converted to linear at the end when the project uses Linear colour space.
Shader "TriSplat/SHAtlas"
{
    Properties { _MainTex ("Atlas", 2D) = "white" {} }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Cull Off
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            StructuredBuffer<uint> _SH;
            int _ShCoeffs;               // (degree + 1)^2, 1..16
            float4x4 _SceneDirFromUnity; // 3x3 used: object-space (= exported Unity frame) direction -> scene frame

            struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; float2 vid : TEXCOORD1; };
            struct v2f { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; float3 r : TEXCOORD1; };

            float sh(uint v, uint j, uint c)
            {
                uint i = (v * (uint)_ShCoeffs + j) * 3u + c;
                uint w = _SH[i >> 1];
                return f16tof32((i & 1u) ? (w >> 16) : (w & 0xFFFFu));
            }
            float3 shc(uint v, uint j) { return float3(sh(v, j, 0), sh(v, j, 1), sh(v, j, 2)); }

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                uint id = (uint)(v.vid.x + 0.5);
                // direction in object space = the exported Unity frame (works if the object is moved / rotated)
                float3 camObj = mul(unity_WorldToObject, float4(_WorldSpaceCameraPos, 1)).xyz;
                float3 d = normalize(mul((float3x3)_SceneDirFromUnity, v.vertex.xyz - camObj));
                float x = d.x, y = d.y, z = d.z;
                float3 dc = shc(id, 0);
                float3 res = 0.28209479177387814 * dc;
                if (_ShCoeffs > 1)
                {
                    res += -0.4886025119029199 * y * shc(id, 1) + 0.4886025119029199 * z * shc(id, 2)
                           - 0.4886025119029199 * x * shc(id, 3);
                }
                if (_ShCoeffs > 4)
                {
                    float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
                    res += 1.0925484305920792 * xy * shc(id, 4) - 1.0925484305920792 * yz * shc(id, 5)
                         + 0.31539156525252005 * (2.0 * zz - xx - yy) * shc(id, 6)
                         - 1.0925484305920792 * xz * shc(id, 7) + 0.5462742152960396 * (xx - yy) * shc(id, 8);
                    if (_ShCoeffs > 9)
                    {
                        res += -0.5900435899266435 * y * (3 * xx - yy) * shc(id, 9)
                             + 2.890611442640554 * xy * z * shc(id, 10)
                             - 0.4570457994644658 * y * (4 * zz - xx - yy) * shc(id, 11)
                             + 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy) * shc(id, 12)
                             - 0.4570457994644658 * x * (4 * zz - xx - yy) * shc(id, 13)
                             + 1.445305721320277 * z * (xx - yy) * shc(id, 14)
                             - 0.5900435899266435 * x * (xx - 3 * yy) * shc(id, 15);
                    }
                }
                o.r = max(res + 0.5, 0) - max(0.28209479177387814 * dc + 0.5, 0);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float3 c = saturate(tex2D(_MainTex, i.uv).rgb + i.r);
                #ifndef UNITY_COLORSPACE_GAMMA
                c = GammaToLinearSpace(c);
                #endif
                return float4(c, 1);
            }
            ENDCG
        }
    }
}
