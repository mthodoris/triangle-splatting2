"""Step 6 of notes/plan.md: perspective-correct texture coordinates.

A single large triangle, seen at a steep angle (its corners are 1.5 and 8 units from the camera), with
colour 0 and a texture. With background 0 the rendered image is alpha * texture(uv), so image / alpha is
the texture value the rasterizer used at each pixel.

Reference (float64 PyTorch, no rasterizer code): for every pixel, the exact camera ray through the pixel
centre (same pixel convention as TS+: pixel x <-> NDC (2x+1)/W - 1) is intersected with the triangle's
plane; the 3D barycentric weights of the hit point are the UV (v0 -> (0,0), v1 -> (1,0), v2 -> (0,1)),
and the texture is sampled with grid_sample(bilinear, border, align_corners=True), which is what
texture_sampler.h implements.

Checks:
  1. perspective_uv=True matches the reference inside the triangle (pixels not at its edge);
  2. negative control: perspective_uv=False (steps 4-5, screen space) does NOT match it;
  3. zero texture: both modes give exactly the no-texture image;
  4. a triangle seen head-on (all corners at the same depth): both modes agree (the correction is then
     exactly the identity), so the difference in 1-2 really comes from the perspective;
  5. writes a picture: checkerboard, screen space | perspective | reference.

Usage (from combined/, venv active):
  python tests/test_texture_perspective.py --cameras <cameras.json> [--view 0] [--downscale 8] [--out <dir>]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compare_rasterizers import camera_from_json  # noqa: E402
import diff_textured_triangle_rasterization as ours  # noqa: E402

dev = "cuda"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")


def triangle(cam_json, corners_cam):
    """One triangle given in the camera frame (x right, y down, z forward) -> world-space model."""
    R = torch.tensor(np.array(cam_json["rotation"]), dtype=torch.float32, device=dev)   # camera -> world
    p = torch.tensor(np.array(cam_json["position"]), dtype=torch.float32, device=dev)
    verts = (torch.tensor(corners_cam, dtype=torch.float32, device=dev) @ R.T + p).contiguous()
    return dict(vertices=verts, triangles_indices=torch.arange(3, device=dev, dtype=torch.int32).view(1, 3),
                vertex_weights=torch.ones(3, device=dev), sigma=0.01)


def render(m, cam, texture, perspective_uv):
    settings = ours.TriangleRasterizationSettings(
        image_height=cam["H"], image_width=cam["W"], tanfovx=cam["tanfovx"], tanfovy=cam["tanfovy"],
        bg=torch.zeros(3, device=dev), scale_modifier=1.0,
        viewmatrix=cam["viewmatrix"], projmatrix=cam["projmatrix"],
        sh_degree=0, campos=cam["campos"], prefiltered=False, debug=False, perspective_uv=perspective_uv)
    rast = ours.TriangleRasterizer(raster_settings=settings)
    kw = {} if texture is None else dict(texture=texture)
    image, _, _, allmap, _ = rast(vertices=m["vertices"], triangles_indices=m["triangles_indices"],
                                  vertex_weights=m["vertex_weights"], sigma=m["sigma"],
                                  scaling=torch.zeros(1, device=dev), shs=None,
                                  colors_precomp=torch.zeros(3, 3, device=dev), **kw)
    return image, allmap[1]


def reference(m, cam, texture):
    """Exact ray-triangle UV per pixel and the bilinear texture sample there (float64).
    Returns (texture value [3,H,W], inside mask with a margin, min barycentric weight)."""
    H, W = cam["H"], cam["W"]
    P = cam["projmatrix"].double()                      # row-vector convention: [x y z 1] @ P
    Pinv = torch.linalg.inv(P)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float64),
                            torch.arange(W, device=dev, dtype=torch.float64), indexing="ij")
    nx, ny = (2 * xs + 1) / W - 1, (2 * ys + 1) / H - 1
    def unproject(z):
        q = torch.stack([nx, ny, torch.full_like(nx, z), torch.ones_like(nx)], -1) @ Pinv
        return q[..., :3] / q[..., 3:]
    o, d = unproject(0.0), unproject(0.5)
    d = d - o
    v = m["vertices"].double()
    e1, e2 = v[1] - v[0], v[2] - v[0]
    n = torch.linalg.cross(e1, e2)
    t = ((v[0] - o) @ n) / (d @ n)
    x = o + t[..., None] * d                            # hit point
    # barycentric weights of x: x = v0 + l1 e1 + l2 e2
    G = torch.stack([e1, e2], 1)                        # [3, 2]
    l12 = torch.linalg.lstsq(G.expand(H * W, 3, 2), (x - v[0]).reshape(-1, 3, 1)).solution.reshape(H, W, 2)
    l1, l2 = l12[..., 0], l12[..., 1]
    l0 = 1 - l1 - l2
    bmin = torch.minimum(torch.minimum(l0, l1), l2)
    grid = torch.stack([l1 * 2 - 1, l2 * 2 - 1], -1)[None]           # x <- u, y <- v
    val = F.grid_sample(texture.double(), grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
    return val, bmin


def compare(m, cam, tex, label):
    val, bmin = reference(m, cam, tex)
    out = {}
    for mode in (True, False):
        img, alpha = render(m, cam, tex, mode)
        ok = (alpha > 0.5) & (bmin > 0.02)               # inside, away from the edges
        got = img.double() / alpha.double().clamp_min(1e-6)
        out[mode] = ((got - val).abs().amax(0)[ok].max().item(), int(ok.sum()), got)
    return out, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--downscale", type=int, default=8)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cams = json.load(open(args.cameras))
    cam = camera_from_json(cams[args.view], args.downscale)
    S = args.S
    g = torch.Generator(device=dev).manual_seed(0)
    print(f"camera {args.view}: {cam['W']}x{cam['H']}, texture {S}x{S}")

    steep = triangle(cams[args.view], [[-0.9, 0.55, 1.5], [0.9, 0.55, 1.5], [0.0, -2.5, 8.0]])
    flat = triangle(cams[args.view], [[-1.5, 0.9, 4.0], [1.5, 0.9, 4.0], [0.2, -1.2, 4.0]])
    tex = torch.rand(1, 3, S, S, device=dev, generator=g)

    print("\nsteep triangle (corners at depth 1.5, 1.5, 8)")
    out, _ = compare(steep, cam, tex, "steep")
    e_p, n, _ = out[True]
    e_s, _, _ = out[False]
    check("perspective_uv=True == exact ray-triangle reference", e_p <= 1e-3,
          f"(max err {e_p:.1e} over {n} pixels)")
    check("negative control: screen-space UV (steps 4-5) does not match", e_s > 0.05, f"(max err {e_s:.2f})")

    print("\nhead-on triangle (all corners at depth 4)")
    out, _ = compare(flat, cam, tex, "flat")
    check("both modes match the reference (correction is the identity at equal depths)",
          out[True][0] <= 1e-3 and out[False][0] <= 1e-3,
          f"(max err perspective {out[True][0]:.1e}, screen space {out[False][0]:.1e})")

    print("\nzero texture")
    for mode in (True, False):
        a, _ = render(steep, cam, None, mode)
        b, _ = render(steep, cam, torch.zeros(1, 3, S, S, device=dev), mode)
        check(f"perspective_uv={mode}: zero texture == no texture, bit-identical", torch.equal(a, b))

    if args.out:
        import torchvision
        os.makedirs(args.out, exist_ok=True)
        ys, xs = torch.meshgrid(torch.arange(S, device=dev), torch.arange(S, device=dev), indexing="ij")
        board = (((xs // 2) + (ys // 2)) % 2).float()
        board = torch.stack([board, 0.3 + 0.4 * board, 1 - board])[None]
        out, val = compare(steep, cam, board, "board")
        _, alpha = render(steep, cam, board, True)
        ok = (alpha > 0.5)[None]
        pics = [out[False][2] * ok, out[True][2] * ok, val * ok]
        path = os.path.join(args.out, f"perspective_view{args.view}.png")
        torchvision.utils.save_image(torch.cat(pics, 2).clamp(0, 1).float(), path)
        print(f"\nwrote {path} (left: screen-space UV, middle: perspective-correct, right: exact reference)")

    print("\nRESULT:", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)} checks)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
