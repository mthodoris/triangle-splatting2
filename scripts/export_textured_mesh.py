#!/usr/bin/env python3
"""Export a textured-triangle checkpoint as a textured mesh (OBJ+MTL+PNG and/or GLB). CPU only.

Each triangle's S x S RGB residual texture becomes one S x S tile of a texture atlas. The tile
holds the *full* colour (base vertex colour + residual), so any standard mesh viewer shows what
the renderer shows (view-independent part only, see below).

Mapping (matches cuda_rasterizer/texture_sampler.h, align_corners=True):
  texel (x, y) of triangle t sits at UV (u, v) = (x/(S-1), y/(S-1)), where u is the weight of
  vertex 1 and v the weight of vertex 2 (vertex 0 has weight 1-u-v). Vertex 0 -> texel (0,0),
  vertex 1 -> texel (S-1, 0), vertex 2 -> texel (0, S-1).
  In the atlas the corners are put on the *pixel centres* of those texels, so a viewer's normal
  bilinear sampling reproduces the renderer's bilinear sampling exactly inside each tile.
  Tile pixel value = linear vertex colour at (u, v) (extrapolated for u+v > 1, those texels are
  needed by bilinear cells on the diagonal) + texture[:, y, x].

Differences from the CUDA render (expected, small):
  - vertex colour = SH degree 0 only (no view dependence), like TS+'s create_ply.py;
  - viewers interpolate the vertex colour perspective-correctly, TS+ in screen space;
  - 8-bit colour, clamped to [0, 1];
  - mip-mapping in a viewer can bleed between neighbouring tiles when zoomed far out.
Vertices are not shared in the output (each triangle has its own UVs) -> triangle soup.

Usage:
  python scripts/export_textured_mesh.py <model_dir or state_dict.pt> --out <prefix> [--obj] [--glb] [--check]
"""
import argparse
import math
import os

import numpy as np
import torch
from PIL import Image

SH_C0 = 0.28209479177387814


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "point_cloud", "iteration_30000", "point_cloud_state_dict.pt")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    verts = sd["triangles_points"].detach().float().numpy()
    faces = sd["_triangle_indices"].detach().long().numpy()
    dc = sd["features_dc"].detach().float().numpy()[:, 0, :]                 # [V, 3]
    tex = sd.get("texture")
    if tex is None:
        raise SystemExit("checkpoint has no 'texture' (plain TS+ run?) -> use create_ply.py")
    tex = tex.detach().float().numpy()                                       # [T, 3, S, S]  (c, y, x)
    return verts, faces, dc, tex


def vertex_colours(dc):
    # computeColorFromSH at degree 0: SH_C0 * dc + 0.5, then max(0) (no upper clamp in the renderer)
    return np.maximum(SH_C0 * dc + 0.5, 0.0)


def atlas_layout(T, P):
    tiles_per_row = math.ceil(math.sqrt(T))
    W = tiles_per_row * P
    H = math.ceil(T / tiles_per_row) * P
    return tiles_per_row, W, H


def auto_upsample(T, S, max_side=16384, k_max=4):
    for k in range(k_max, 0, -1):
        if max(atlas_layout(T, (S - 1) * k + 1)[1:]) <= max_side:
            return k
    return 1


def upsample_matrix(S, P):
    """[P*P, S*S] bilinear weights (align_corners=True) from an S x S texture to a P x P tile."""
    g = np.arange(P) / (P - 1) * (S - 1)
    i0 = np.minimum(np.floor(g).astype(int), S - 2)
    f = g - i0
    A = np.zeros((P, S))
    A[np.arange(P), i0] = 1 - f
    A[np.arange(P), i0 + 1] += f
    return np.kron(A, A).astype(np.float32)          # rows: y*P + x, cols: j*S + i


def bake(faces, col, tex, k=1):
    """Tile = P x P pixels, P = (S-1)*k + 1. k = 1: the raw texels (unclamped float, exact under
    bilinear). k > 1: the renderer's colour evaluated on a k-times finer grid and clamped to [0, 1]
    per pixel (like the renderer clamps the final pixel), so texel overshoot is not cut off.
    Returns atlas [H, W, 3] and per-corner UVs [T, 3, 2] (OBJ convention, v up)."""
    T, _, S, _ = tex.shape
    P = (S - 1) * k + 1
    tpr, W, H = atlas_layout(T, P)
    g = np.arange(P, dtype=np.float32) / (P - 1)
    u = g[None, :].repeat(P, 0)       # [y, x] -> u = x/(P-1)
    v = g[:, None].repeat(P, 1)       # [y, x] -> v = y/(P-1)
    w0, w1, w2 = (1 - u - v), u, v    # weights of vertex 0, 1, 2
    M = upsample_matrix(S, P)
    atlas = np.zeros((H, W, 3), np.float32)
    tx = np.arange(T) % tpr
    ty = np.arange(T) // tpr
    chunk = max(1000, 30_000_000 // (P * P))
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        c0, c1, c2 = (col[faces[s:e, j]] for j in range(3))     # [n, 3]
        lin = (c0[:, :, None, None] * w0 + c1[:, :, None, None] * w1 + c2[:, :, None, None] * w2)
        smp = (tex[s:e].reshape(e - s, 3, S * S) @ M.T).reshape(e - s, 3, P, P)
        tile = (lin + smp).transpose(0, 2, 3, 1)                # [n, P(y), P(x), 3]
        if k > 1:
            tile = np.clip(tile, 0.0, 1.0)
        # scatter tiles row by row of tiles
        for r in np.unique(ty[s:e]):
            idx = np.nonzero(ty[s:e] == r)[0]
            x0 = tx[s + idx[0]] * P
            atlas[r * P:(r + 1) * P, x0:x0 + len(idx) * P] = tile[idx].transpose(1, 0, 2, 3).reshape(P, len(idx) * P, 3)
    # corner pixel centres
    px = np.stack([tx * P + 0.5, tx * P + (P - 1) + 0.5, tx * P + 0.5], 1)          # [T, 3]
    py = np.stack([ty * P + 0.5, ty * P + 0.5, ty * P + (P - 1) + 0.5], 1)
    uv = np.stack([px / W, 1.0 - py / H], -1).astype(np.float32)                   # [T, 3, 2]
    return atlas, uv


def bilinear_clamp(img, x, y):
    """img [H, W, C]; pixel-centre coordinates x, y (0 = centre of first pixel)."""
    H, W, _ = img.shape
    x = np.clip(x, 0, W - 1); y = np.clip(y, 0, H - 1)
    x0 = np.minimum(np.floor(x).astype(int), W - 2); y0 = np.minimum(np.floor(y).astype(int), H - 2)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1]
            + (1 - fx) * fy * img[y0 + 1, x0] + fx * fy * img[y0 + 1, x0 + 1])


def check(faces, col, tex, atlas, uv, n=200_000, seed=0):
    """Random points inside random triangles: CUDA-style colour vs sampling the atlas like a viewer."""
    rng = np.random.default_rng(seed)
    T, _, S, _ = tex.shape
    t = rng.integers(0, T, n)
    a, b = rng.random(n), rng.random(n)
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]                 # (b0, b1) inside the triangle
    # reference = renderer: interp vertex colours + bilinear(tex, x=u*(S-1), y=v*(S-1)), clamp-to-edge
    w = np.stack([1 - a - b, a, b], 1)
    lin = sum(w[:, k:k + 1] * col[faces[t, k]] for k in range(3))
    x, y = a * (S - 1), b * (S - 1)
    x0 = np.minimum(np.floor(x).astype(int), S - 2); y0 = np.minimum(np.floor(y).astype(int), S - 2)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    tt = tex[t]                                                  # [n, 3, S, S]
    ar = np.arange(n)
    smp = ((1 - fx) * (1 - fy) * tt[ar, :, y0, x0] + fx * (1 - fy) * tt[ar, :, y0, x0 + 1]
           + (1 - fx) * fy * tt[ar, :, y0 + 1, x0] + fx * fy * tt[ar, :, y0 + 1, x0 + 1])
    ref = lin + smp
    # viewer: interpolate corner UVs, bilinear in the atlas
    H, W, _ = atlas.shape
    U = (w[:, :, None] * uv[t]).sum(1)
    got = bilinear_clamp(atlas, U[:, 0] * W - 0.5, (1 - U[:, 1]) * H - 0.5)
    refc = np.clip(ref, 0, 1)                                    # renderer clamps the final pixel
    got8 = bilinear_clamp(np.round(np.clip(atlas, 0, 1) * 255) / 255, U[:, 0] * W - 0.5, (1 - U[:, 1]) * H - 0.5)
    err = np.abs(np.clip(got, 0, 1) - refc).max()
    psnr8 = -10 * np.log10(((got8 - refc) ** 2).mean())
    print(f"check ({n} random surface points): max |float atlas - renderer| = {err:.3f}; "
          f"8-bit PNG vs renderer: PSNR {psnr8:.1f} dB, mean abs {np.abs(got8 - refc).mean():.4f}")
    return psnr8


def write_obj(prefix, soup, uv, png_name):
    T = uv.shape[0]
    base = os.path.basename(prefix)
    with open(prefix + ".mtl", "w") as f:
        f.write(f"newmtl tex\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {png_name}\n")
    with open(prefix + ".obj", "w") as f:
        f.write(f"mtllib {base}.mtl\nusemtl tex\n")
        np.savetxt(f, soup.reshape(-1, 3), fmt="v %.6f %.6f %.6f")
        np.savetxt(f, uv.reshape(-1, 2), fmt="vt %.7f %.7f")
        i = np.arange(1, 3 * T + 1).reshape(T, 3)
        np.savetxt(f, np.repeat(i, 2, axis=1), fmt="f %d/%d %d/%d %d/%d")


def write_glb(prefix, soup, uv, img):
    import trimesh
    T = uv.shape[0]
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img, metallicFactor=0.0, roughnessFactor=1.0)
    mesh = trimesh.Trimesh(vertices=soup.reshape(-1, 3), faces=np.arange(3 * T).reshape(T, 3),
                           visual=trimesh.visual.TextureVisuals(uv=uv.reshape(-1, 2), material=mat),
                           process=False)
    mesh.export(prefix + ".glb")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="model dir (uses iteration_30000) or point_cloud_state_dict.pt")
    p.add_argument("--out", required=True, help="output prefix, e.g. runs/cluster_meshes/foo_textured")
    p.add_argument("--obj", action="store_true", help="write OBJ+MTL+PNG (MeshLab)")
    p.add_argument("--glb", action="store_true", help="write GLB (Blender / Unity)")
    p.add_argument("--check", action="store_true", help="numeric check atlas vs renderer formula")
    p.add_argument("--upsample", type=int, default=0,
                   help="tile = (S-1)*k+1 px, clamped after interpolation (1 = raw texels, clamped per texel); "
                        "0 = auto: largest k <= 4 with atlas side <= 16384")
    a = p.parse_args()

    verts, faces, dc, tex = load(a.model)
    T, _, S, _ = tex.shape
    col = vertex_colours(dc)
    k = a.upsample or auto_upsample(T, S)
    atlas, uv = bake(faces, col, tex, k)
    print(f"{T} triangles, S = {S}, upsample {k}, atlas {atlas.shape[1]} x {atlas.shape[0]}")
    if a.check:
        check(faces, col, tex, atlas, uv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None
    img = Image.fromarray((np.clip(atlas, 0, 1) * 255 + 0.5).astype(np.uint8))
    png = a.out + "_atlas.png"
    img.save(png, compress_level=1)
    soup = verts[faces]                                          # [T, 3, 3]
    if a.obj:
        write_obj(a.out, soup, uv, os.path.basename(png))
    if a.glb:
        write_glb(a.out, soup, uv, img)
    print("wrote", a.out + ".*")


if __name__ == "__main__":
    main()
