/*
 * Bilinear texture sampling with clamp-to-edge, forward and backward.
 *
 * Ported from BBSplat's cuda_rasterizer/grid_sample.h (biliniar_texture_sampler /
 * backward_biliniar_texture_sampler), with the changes decided in notes/decisions.md:
 *   - clamp-to-edge instead of zero padding: the sample coordinate is clamped to the
 *     texture before interpolating, exactly like PyTorch
 *     grid_sample(mode='bilinear', padding_mode='border', align_corners=True).
 *     So a texture never fades to 0 near a triangle's edge (no dark seams).
 *   - texture coordinates (u, v) are in [0, 1] (BBSplat: [-1, 1]);
 *     texel centres sit at u = i / (S - 1), i = 0..S-1 (align_corners=True).
 *   - texture size S is a parameter (S >= 2).
 *
 * Layout (same as BBSplat): one texture = C channels of S x S floats, channel-first,
 * row-major: texture[c * S * S + y * S + x]. u selects the column x, v the row y.
 */
#ifndef CUDA_TEXTURE_SAMPLER_H_INCLUDED
#define CUDA_TEXTURE_SAMPLER_H_INCLUDED

#include <cuda_runtime.h>

// Where (u, v) lands in the texture: the top-left texel (x0, y0) of the 2x2 cell
// and the fractions (tx, ty) in [0, 1] inside that cell.
// gx / gy are 1 if the coordinate was inside [0, S-1] and 0 if it was clamped
// (then moving u or v does not change the sample, so its gradient is 0).
struct TexelCell {
    int x0, y0;
    float tx, ty;
    float gx, gy;
};

__forceinline__ __device__ TexelCell texel_cell(float u, float v, int S)
{
    const float lim = (float)(S - 1);
    float x = u * lim;
    float y = v * lim;
    TexelCell c;
    c.gx = (x >= 0.f && x <= lim) ? 1.f : 0.f;
    c.gy = (y >= 0.f && y <= lim) ? 1.f : 0.f;
    x = fminf(fmaxf(x, 0.f), lim);
    y = fminf(fmaxf(y, 0.f), lim);
    // Keep x0 <= S-2 so that x0+1 is always a valid texel. At x == S-1 this gives
    // x0 = S-2, tx = 1, i.e. exactly texel S-1 (same value as PyTorch's border mode).
    c.x0 = min((int)floorf(x), S - 2);
    c.y0 = min((int)floorf(y), S - 2);
    c.tx = x - (float)c.x0;
    c.ty = y - (float)c.y0;
    return c;
}

// out[c] = bilinear sample of channel c at (u, v), c = 0..C-1.
__forceinline__ __device__ void sample_texture_bilinear(
    float* out, float u, float v,
    const float* __restrict__ texture, int S, int C)
{
    const TexelCell k = texel_cell(u, v, S);
    const float w00 = (1.f - k.tx) * (1.f - k.ty);   // (x0,   y0)
    const float w10 = k.tx * (1.f - k.ty);           // (x0+1, y0)
    const float w01 = (1.f - k.tx) * k.ty;           // (x0,   y0+1)
    const float w11 = k.tx * k.ty;                   // (x0+1, y0+1)
    const int i00 = k.y0 * S + k.x0;
    const int SS = S * S;
    for (int c = 0; c < C; ++c) {
        const float* t = texture + c * SS + i00;
        out[c] = w00 * t[0] + w10 * t[1] + w01 * t[S] + w11 * t[S + 1];
    }
}

// Backward of sample_texture_bilinear.
//   dL_dout[c]  : gradient of the loss w.r.t. the sampled value of channel c
//   dL_dtexture : gradient buffer of this texture (same layout); the 4 texels of the cell
//                 get weight * dL_dout[c] via atomicAdd (many pixels hit the same texel)
// Returns dL/d(u, v). For textured triangles this is NOT propagated to the vertices
// (decision 2 in notes/decisions.md), but it is computed so it can be tested and used later.
__forceinline__ __device__ float2 backward_sample_texture_bilinear(
    const float* dL_dout, float u, float v,
    const float* __restrict__ texture, float* __restrict__ dL_dtexture, int S, int C)
{
    const TexelCell k = texel_cell(u, v, S);
    const float w00 = (1.f - k.tx) * (1.f - k.ty);
    const float w10 = k.tx * (1.f - k.ty);
    const float w01 = (1.f - k.tx) * k.ty;
    const float w11 = k.tx * k.ty;
    const int i00 = k.y0 * S + k.x0;
    const int SS = S * S;
    float dtx = 0.f, dty = 0.f;   // d sample / d tx, d sample / d ty, weighted by dL_dout
    for (int c = 0; c < C; ++c) {
        const float g = dL_dout[c];
        float* d = dL_dtexture + c * SS + i00;
        atomicAdd(d,         w00 * g);
        atomicAdd(d + 1,     w10 * g);
        atomicAdd(d + S,     w01 * g);
        atomicAdd(d + S + 1, w11 * g);
        const float* t = texture + c * SS + i00;
        dtx += g * ((t[1] - t[0]) * (1.f - k.ty) + (t[S + 1] - t[S]) * k.ty);
        dty += g * ((t[S] - t[0]) * (1.f - k.tx) + (t[S + 1] - t[1]) * k.tx);
    }
    // x = u * (S-1), so d/du = (S-1) * d/dx; zero where the coordinate was clamped.
    const float lim = (float)(S - 1);
    return make_float2(dtx * lim * k.gx, dty * lim * k.gy);
}

#endif
