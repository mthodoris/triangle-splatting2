"""Step 5 of notes/plan.md: gradients of the per-triangle texture (backward pass).

All gradient comparisons run with atomic_vertex_grads=True, so the vertex gradients are reproducible
and can be compared tightly (decision A: training keeps the default False = original TS+).

Checks:
  1. fixture, zero texture: vertex / vertex-weight / SH gradients are the same as with no texture
     (the texture must not change plain TS+), and the texture itself gets a non-zero gradient;
  2. tiny synthetic scene (a few big overlapping triangles in front of a fixture camera):
     dL/dtexture == finite differences, for a zero and a random texture. The image is linear in the
     texture (the texture only adds to the colour, alpha does not depend on it), so a step of 1 is exact
     up to float rounding;
  3. tiny scene: texels no pixel can read get exactly 0 (the u+v>1 corner of the square, and every
     texel of a triangle that is behind the camera);
  4. fixture, non-zero texture changes the vertex / opacity gradients correctly: on a copy with
     unshared vertices, a texture linear in (u, v) must give the same vertex, vertex-weight and colour
     gradients as adding those values to the vertex colours (this checks that backward adds the texture
     sample to the colour used for dL/dalpha);
  5. fixture: dL/dtexture predicts the change of the loss for a random texture change (linearity, all
     texels at once).

Usage (from combined/, venv active):
  python tests/test_texture_backward.py --model <dir with point_cloud_state_dict.pt> --cameras <cameras.json> \
      [--views 0 50] [--downscale 8] [--scaling 4] [--S 8]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compare_rasterizers import load_model, camera_from_json  # noqa: E402
from test_texture_forward import unshare_vertices  # noqa: E402
import diff_textured_triangle_rasterization as ours  # noqa: E402

dev = "cuda"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")


def run(m, cam, scaling, texture=None, colors_precomp=None, loss_seed=0):
    """Render, apply a fixed pseudo-random loss (image + aux terms, as compare_rasterizers.py), backward.
    Returns the loss (float64) and the gradients of all leaves."""
    settings = ours.TriangleRasterizationSettings(
        image_height=cam["H"] * scaling, image_width=cam["W"] * scaling,
        tanfovx=cam["tanfovx"], tanfovy=cam["tanfovy"],
        bg=torch.zeros(3, device=dev), scale_modifier=1.0,
        viewmatrix=cam["viewmatrix"], projmatrix=cam["projmatrix"],
        sh_degree=m.get("sh_degree", 0), campos=cam["campos"], prefiltered=False, debug=False,
        atomic_vertex_grads=True)
    rast = ours.TriangleRasterizer(raster_settings=settings)
    leaf = lambda t: t.detach().clone().requires_grad_(True)  # noqa: E731
    lv = dict(vertices=leaf(m["vertices"]), vertex_weights=leaf(m["vertex_weights"]))
    kw = dict(triangles_indices=m["triangles_indices"], sigma=m["sigma"],
              scaling=torch.zeros(m["triangles_indices"].shape[0], device=dev))
    if colors_precomp is not None:
        lv["colors"] = leaf(colors_precomp)
        kw.update(shs=None, colors_precomp=lv["colors"])
    else:
        lv["shs"] = leaf(m["shs"])
        kw.update(shs=lv["shs"], colors_precomp=None)
    if texture is not None:
        lv["texture"] = leaf(texture)
        kw["texture"] = lv["texture"]
    image, _, _, allmap, _ = rast(vertices=lv["vertices"], vertex_weights=lv["vertex_weights"], **kw)
    g = torch.Generator(device=dev).manual_seed(loss_seed)
    w_img = torch.rand(image.shape, device=dev, generator=g)
    w_aux = torch.rand(allmap.shape, device=dev, generator=g)
    w_aux[6] = 0   # triangle-id channel is not differentiable
    loss = (image * w_img).sum() + (allmap * w_aux).sum()
    loss.backward()
    lossd = (image.double() * w_img.double()).sum() + (allmap.double() * w_aux.double()).sum()
    return lossd.item(), {k: (v.grad.detach().clone() if v.grad is not None else None) for k, v in lv.items()}


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def tiny_scene(cam_json, downscale):
    """A few big, overlapping triangles 2-4 units in front of a fixture camera, plus one behind it."""
    cam = camera_from_json(cam_json, downscale)
    R = torch.tensor(np.array(cam_json["rotation"]), dtype=torch.float32, device=dev)   # camera -> world
    p = torch.tensor(np.array(cam_json["position"]), dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev).manual_seed(1)
    tris = []
    for i in range(6):
        z = 2.0 + 0.4 * i
        c = (torch.rand(2, device=dev, generator=g) - 0.5) * 0.6 * z
        corners = torch.tensor([[-0.6, -0.5], [0.7, -0.4], [-0.1, 0.7]], device=dev) * z * 0.5
        corners = corners + (torch.rand(3, 2, device=dev, generator=g) - 0.5) * 0.2 * z
        tris.append(torch.cat([c + corners, torch.full((3, 1), z, device=dev)], 1))
    tris.append(torch.tensor([[-1.0, -1.0, -3.0], [1.0, -1.0, -3.0], [0.0, 1.0, -3.0]], device=dev))  # behind
    cam_pts = torch.cat(tris, 0)                                   # [3T, 3] camera frame
    verts = (cam_pts @ R.T + p).contiguous()
    T = len(tris)
    m = dict(vertices=verts,
             triangles_indices=torch.arange(3 * T, device=dev, dtype=torch.int32).view(T, 3),
             vertex_weights=torch.full((3 * T,), 0.7, device=dev),   # semi-transparent: several layers count
             sigma=1.0, sh_degree=0)
    colors = torch.rand(3 * T, 3, device=dev, generator=g)
    return m, cam, colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--views", type=int, nargs="+", default=[0, 50])
    ap.add_argument("--downscale", type=int, default=8)
    ap.add_argument("--scaling", type=int, default=4)
    ap.add_argument("--S", type=int, default=8, help="texture resolution")
    args = ap.parse_args()

    m = load_model(args.model)
    T, S = m["triangles_indices"].shape[0], args.S
    cams = json.load(open(args.cameras))
    print(f"model: {m['vertices'].shape[0]} vertices, {T} triangles, texture {S}x{S}")
    g = torch.Generator(device=dev).manual_seed(0)

    # 1. zero texture on the fixture
    for v in args.views:
        cam = camera_from_json(cams[v], args.downscale)
        print(f"\nfixture view {v}: zero texture")
        _, a = run(m, cam, args.scaling)
        _, b = run(m, cam, args.scaling, texture=torch.zeros(T, 3, S, S, device=dev))
        _, a2 = run(m, cam, args.scaling)
        for k in ("vertices", "vertex_weights", "shs"):
            noise, d = rel(a2[k], a[k]), rel(b[k], a[k])
            check(f"zero texture: d/d{k} == no texture", d <= max(1e-5, 10 * noise),
                  f"(rel diff {d:.1e}; run-to-run {noise:.1e})")
        nz = (b["texture"].abs().sum((1, 2, 3)) > 0).float().mean().item()
        check("zero texture: the texture gets a gradient", b["texture"].abs().max() > 0,
              f"({100 * nz:.1f}% of triangles have a non-zero texture gradient)")
        check("no texture -> no texture gradient entry", "texture" not in a)

    # 2.-3. tiny synthetic scene, finite differences
    print("\ntiny scene: finite differences")
    ms, cs, colors = tiny_scene(cams[args.views[0]], 16)
    Ts = ms["triangles_indices"].shape[0]
    for name, tex in [("zero", torch.zeros(Ts, 3, S, S, device=dev)),
                      ("random", (torch.rand(Ts, 3, S, S, device=dev, generator=g) - 0.5))]:
        L0, gr = run(ms, cs, 1, texture=tex, colors_precomp=colors)
        gt = gr["texture"]
        flat = gt.flatten()
        # texels: the 60 with the largest gradient + 60 random ones
        idx = torch.cat([flat.abs().topk(60).indices,
                         torch.randint(0, flat.numel(), (60,), device=dev, generator=g)])
        fd = []
        for i in idx.tolist():
            t2 = tex.clone().flatten()
            t2[i] += 1.0
            L1, _ = run(ms, cs, 1, texture=t2.view_as(tex), colors_precomp=colors)
            fd.append(L1 - L0)
        fd = torch.tensor(fd, device=dev, dtype=torch.float64)
        an = flat[idx].double()
        err = ((an - fd).abs().max() / fd.abs().max()).item()
        check(f"{name} texture: dL/dtexture == finite differences ({len(idx)} texels)", err <= 1e-3,
              f"(max err {err:.1e} of the largest gradient {fd.abs().max().item():.2e})")
        covered = (gt.abs().sum((1, 2, 3)) > 0)
        check(f"{name} texture: the 6 visible triangles get a gradient, the one behind the camera none",
              bool(covered[:6].all()) and not bool(covered[6]), f"({covered.tolist()})")
        ys, xs = torch.meshgrid(torch.arange(S, device=dev), torch.arange(S, device=dev), indexing="ij")
        unreachable = (xs + ys) > S                                   # whole bilinear cell has u + v > 1
        check(f"{name} texture: texels in the u+v>1 corner get exactly 0",
              bool((gt[:, :, unreachable] == 0).all()), f"({int(unreachable.sum())} texels per channel)")

    # 4. linear texture == vertex colours: vertex / opacity / colour gradients
    v = args.views[0]
    cam = camera_from_json(cams[v], args.downscale)
    print(f"\nfixture view {v}: non-zero texture, vertex gradients")
    mu = unshare_vertices(m)
    corner = torch.rand(T, 2, 3, device=dev, generator=g) - 0.5
    lin = torch.linspace(0, 1, S, device=dev)
    tex = (corner[:, 0, :, None, None] * lin.view(1, 1, 1, S)
           + corner[:, 1, :, None, None] * lin.view(1, 1, S, 1)).contiguous()
    base_col = torch.rand(3 * T, 3, device=dev, generator=g) * 0.5
    vert_col = base_col.clone().view(T, 3, 3)
    vert_col[:, 1] += corner[:, 0]
    vert_col[:, 2] += corner[:, 1]
    _, ga = run(mu, cam, args.scaling, texture=tex, colors_precomp=base_col)
    _, gb = run(mu, cam, args.scaling, colors_precomp=vert_col.view(-1, 3).contiguous())
    _, g0 = run(mu, cam, args.scaling, colors_precomp=base_col)                  # texture ignored
    for k in ("vertices", "vertex_weights", "colors"):
        d, d0 = rel(ga[k], gb[k]), rel(g0[k], gb[k])
        # (a few pixels differ because the texture clamps barycentrics slightly outside the triangle)
        check(f"linear texture: d/d{k} == same values in the vertex colours",
              d <= 1e-3 and (k == "colors" or d0 > 10 * d),
              f"(rel diff {d:.1e}; without the texture it would be {d0:.1e})")

    # 5. linearity on the fixture, all texels at once
    tex = (torch.rand(T, 3, S, S, device=dev, generator=g) - 0.5) * 0.2
    dtex = (torch.rand(T, 3, S, S, device=dev, generator=g) - 0.5) * 0.2
    L0, gr = run(m, cam, args.scaling, texture=tex)
    L1, _ = run(m, cam, args.scaling, texture=tex + dtex)
    pred = (gr["texture"].double() * dtex.double()).sum().item()
    err = abs(pred - (L1 - L0)) / abs(L1 - L0)
    check("fixture: <dL/dtexture, dtex> == L(tex + dtex) - L(tex)", err <= 1e-3,
          f"(predicted {pred:.5f}, actual {L1 - L0:.5f}, rel err {err:.1e})")

    print("\nRESULT:", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)} checks)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
