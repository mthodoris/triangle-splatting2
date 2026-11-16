"""Pre-step 5 of notes/plan.md: TS+ vertex gradients are not reproducible because of a data race.

backward.cu preprocessCUDA adds each triangle's gradient to its 3 vertices with a plain '+=';
triangles that share a vertex run in parallel and overwrite each other's additions. Our rasterizer
has a switch `atomic_vertex_grads` (atomicAdd). This test checks, on the local fixture:
  1. switch ON: repeated backward passes give the same vertex gradient (up to float-atomic noise);
  2. switch ON gives the CORRECT gradient: reference = the ORIGINAL rasterizer on a copy of the model
     where every triangle has its own 3 vertices (no shared vertices -> no race), with the copies'
     gradients summed back onto the shared vertices;
  3. switch OFF still behaves like the original TS+ (same spread vs the reference as the original).
Forward outputs don't depend on the switch (backward only).

Usage (from combined/, venv active):
  python tests/test_vertex_grad_race.py --model <dir with point_cloud_state_dict.pt> --cameras <cameras.json> \
      [--views 0 50] [--downscale 8] [--scaling 4] [--repeats 4]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compare_rasterizers import load_model, camera_from_json  # noqa: E402
from test_texture_forward import unshare_vertices  # noqa: E402
import diff_triangle_rasterization as orig  # noqa: E402
import diff_textured_triangle_rasterization as ours  # noqa: E402

dev = "cuda"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")


def vertex_grads(pkg, m, cam, scaling, atomic=None):
    kw = {} if atomic is None else dict(atomic_vertex_grads=atomic)
    settings = pkg.TriangleRasterizationSettings(
        image_height=cam["H"] * scaling, image_width=cam["W"] * scaling,
        tanfovx=cam["tanfovx"], tanfovy=cam["tanfovy"],
        bg=torch.zeros(3, device=dev), scale_modifier=1.0,
        viewmatrix=cam["viewmatrix"], projmatrix=cam["projmatrix"],
        sh_degree=m["sh_degree"], campos=cam["campos"], prefiltered=False, debug=False, **kw)
    rast = pkg.TriangleRasterizer(raster_settings=settings)
    verts = m["vertices"].detach().clone().requires_grad_(True)
    image, _, _, allmap, _ = rast(vertices=verts, triangles_indices=m["triangles_indices"],
                                  vertex_weights=m["vertex_weights"].detach(), sigma=m["sigma"], shs=m["shs"].detach(),
                                  colors_precomp=None,
                                  scaling=torch.zeros(m["triangles_indices"].shape[0], device=dev))
    # same fixed pseudo-random loss as compare_rasterizers.py (all image + aux terms)
    g = torch.Generator(device=dev).manual_seed(0)
    w_img = torch.rand(image.shape, device=dev, generator=g)
    w_aux = torch.rand(allmap.shape, device=dev, generator=g)
    w_aux[6] = 0
    ((image * w_img).sum() + (allmap * w_aux).sum()).backward()
    return verts.grad.detach().clone()


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--views", type=int, nargs="+", default=[0, 50])
    ap.add_argument("--downscale", type=int, default=8)
    ap.add_argument("--scaling", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=4)
    args = ap.parse_args()

    m = load_model(args.model)
    mu = unshare_vertices(m)
    idx = m["triangles_indices"].long().reshape(-1)          # unshared vertex k -> shared vertex idx[k]
    cams = json.load(open(args.cameras))
    print(f"model: {m['vertices'].shape[0]} vertices, {m['triangles_indices'].shape[0]} triangles")

    for v in args.views:
        cam = camera_from_json(cams[v], args.downscale)
        print(f"\nview {v}")
        # reference: original rasterizer, no shared vertices, summed back
        g_u = vertex_grads(orig, mu, cam, args.scaling)
        ref = torch.zeros_like(m["vertices"]).index_add_(0, idx, g_u)
        g_u2 = vertex_grads(orig, mu, cam, args.scaling)
        ref2 = torch.zeros_like(m["vertices"]).index_add_(0, idx, g_u2)
        noise = rel(ref2, ref)
        print(f"  reference reproducible to {noise:.1e} (float-atomic noise)")

        on = [vertex_grads(ours, m, cam, args.scaling, atomic=True) for _ in range(args.repeats)]
        spread_on = max(rel(g, on[0]) for g in on[1:])
        check("switch ON: repeated backward gives the same vertex gradient", spread_on <= 1e-5,
              f"(max rel spread {spread_on:.1e})")
        err_on = max(rel(g, ref) for g in on)
        check("switch ON == race-free reference", err_on <= 1e-5, f"(max rel diff {err_on:.1e})")

        off = [vertex_grads(ours, m, cam, args.scaling, atomic=False) for _ in range(args.repeats)]
        default = vertex_grads(ours, m, cam, args.scaling)          # settings without the field
        org = [vertex_grads(orig, m, cam, args.scaling) for _ in range(args.repeats)]
        err_off = max(rel(g, ref) for g in off + [default])
        err_org = max(rel(g, ref) for g in org)
        check("switch OFF (and default) behaves like original TS+ (racy, same size of error)",
              err_off > 1e-3 and err_org > 1e-3 and 0.2 < err_off / err_org < 5,
              f"(max rel diff to reference: ours-off {err_off:.1e}, original {err_org:.1e})")

    print("\nRESULT:", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)} checks)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
