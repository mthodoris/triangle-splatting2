// physproxy Phase 7: unlit per-vertex color (the render mesh carries SH DC colors per vertex).
// Cull Off: render-mesh triangle winding is arbitrary (indices are stored sorted upstream).
Shader "PhysProxy/VertexColorUnlit"
{
    Properties { _Tint ("Tint", Color) = (1,1,1,1) }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Cull Off
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            fixed4 _Tint;
            struct appdata { float4 vertex : POSITION; fixed4 color : COLOR; float3 normal : NORMAL; };
            struct v2f { float4 pos : SV_POSITION; fixed4 color : COLOR; };
            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.color = v.color * _Tint;
                // Vertex colors are sRGB (as trained). In a Linear-color-space project Unity does not convert
                // them, so the sRGB output would brighten and wash them out: convert to linear here.
                #ifndef UNITY_COLORSPACE_GAMMA
                o.color.rgb = GammaToLinearSpace(o.color.rgb);
                #endif
                return o;
            }
            fixed4 frag (v2f i) : SV_Target { return i.color; }
            ENDCG
        }
    }
}
