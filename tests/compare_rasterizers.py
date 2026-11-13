"""Compare the original TS+ rasterizer with our renamed/modified copy.

Step 0 of notes/plan.md: the copy must give bit-identical forward outputs, and
backward gradients equal up to float-atomic noise.

Both packages must be installed in the same venv:
  diff_triangle_rasterization           (original, built from triangle-splatting2/)
  diff_textured_triangle_rasterization  (ours, built from combined/)

Usage (from combined/, venv active):
  python tests/compare_rasterizers.py --model <dir with point_cloud_state_dict.pt> \
      --cameras <cameras.json> [--views 0 10 20] [--downscale 4] [--scaling 1]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diff_triangle_rasterization as orig  # noqa: E402
import diff_textured_triangle_rasterization as ours  # noqa: E402
from utils.graphics_utils import getProjectionMatrix  # noqa: E402


def load_model(model_dir, device="cuda"):
    s = torch.load(os.path.join(model_dir, "point_cloud_state_dict.pt"), map_location=device)
    floor = 0.9999  # what TriangleModel.load_parameters uses for rendering
    weights = floor + (1.0 - floor) * torch.sigmoid(s["vertex_weight"].float())
    return dict(
        vertices=s["triangles_points"].float().contiguous(),
        triangles_indices=s["_triangle_indices"].int().contiguous(),
        vertex_weights=weights.squeeze().contiguous(),
        sigma=math.exp(float(s["sigma"])),
        shs=torch.cat((s["features_dc"], s["features_rest"]), dim=1).float().contiguous(),
        sh_degree=int(s["active_sh_degree"]),
    )


def camera_from_json(c, downscale, device="cuda"):
    """cameras.json stores camera-to-world rotation and the camera centre (3DGS format)."""
    W, H = int(c["width"]) // downscale, int(c["height"]) // downscale
    fx, fy = c["fx"] / downscale, c["fy"] / downscale
    c2w = np.eye(4)
    c2w[:3, :3] = np.array(c["rotation"])
    c2w[:3, 3] = np.array(c["position"])
    w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)
    fovx, fovy = 2 * math.atan(W / (2 * fx)), 2 * math.atan(H / (2 * fy))
    world_view = w2c.transpose(0, 1)
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).to(device)
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    campos = world_view.inverse()[3, :3]
    return dict(W=W, H=H, tanfovx=math.tan(fovx / 2), tanfovy=math.tan(fovy / 2),
                viewmatrix=world_view, projmatrix=full_proj, campos=campos)


def render(pkg, m, cam, scaling, need_grad):
    settings = pkg.TriangleRasterizationSettings(
        image_height=cam["H"] * scaling, image_width=cam["W"] * scaling,
        tanfovx=cam["tanfovx"], tanfovy=cam["tanfovy"],
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
        viewmatrix=cam["viewmatrix"], projmatrix=cam["projmatrix"],
        sh_degree=m["sh_degree"], campos=cam["campos"], prefiltered=False, debug=False)
    rast = pkg.TriangleRasterizer(raster_settings=settings)
    leaves = {k: m[k].detach().clone().requires_grad_(need_grad)
              for k in ("vertices", "vertex_weights", "shs")}
    out = rast(vertices=leaves["vertices"], triangles_indices=m["triangles_indices"],
               vertex_weights=leaves["vertex_weights"], sigma=m["sigma"], shs=leaves["shs"],
               colors_precomp=None,
               scaling=torch.zeros(m["triangles_indices"].shape[0], device="cuda"))
    image, radii, scr_size, allmap, max_blending = out
    grads = {}
    if need_grad:
        # fixed pseudo-random loss weights so both runs get the same upstream gradient
        g = torch.Generator(device="cuda").manual_seed(0)
        w_img = torch.rand(image.shape, device="cuda", generator=g)
        w_aux = torch.rand(allmap.shape, device="cuda", generator=g)
        w_aux[6] = 0  # channel 6 = triangle id, not differentiable (see method1.md §4.2)
        ((image * w_img).sum() + (allmap * w_aux).sum()).backward()
        grads = {k: v.grad.detach().clone() for k, v in leaves.items()}
    fwd = dict(image=image.detach(), radii=radii.detach(), allmap=allmap.detach(),
               max_blending=max_blending.detach())
    return fwd, grads


def max_rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--views", type=int, nargs="+", default=[0, 50, 100])
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--scaling", type=int, default=1, help="TS+ supersampling factor")
    args = ap.parse_args()

    m = load_model(args.model)
    cams = json.load(open(args.cameras))
    print(f"model: {m['vertices'].shape[0]} vertices, {m['triangles_indices'].shape[0]} triangles, "
          f"sh_degree {m['sh_degree']}, sigma {m['sigma']:.3g}")
    ok = True
    for v in args.views:
        cam = camera_from_json(cams[v], args.downscale)
        f_o, g_o = render(orig, m, cam, args.scaling, need_grad=True)
        f_n, g_n = render(ours, m, cam, args.scaling, need_grad=True)
        f_o2, g_o2 = render(orig, m, cam, args.scaling, need_grad=True)  # run-to-run noise of TS+ itself
        print(f"\nview {v} ({cam['W'] * args.scaling}x{cam['H'] * args.scaling}), "
              f"covered px {(f_o['allmap'][1] > 0).float().mean().item():.1%}")
        for k in f_o:
            if k == "allmap":
                # channels 0-5: depth, alpha, normal xyz, median depth -> must be bit-identical.
                same = torch.equal(f_o[k][:6], f_n[k][:6])
                # channel 6 (triangle id) is only written where some triangle reaches T > 0.5,
                # i.e. alpha > 0.5; elsewhere it is uninitialised memory (method1.md §4.1).
                valid = f_o[k][1] > 0.5
                same_id = torch.equal(f_o[k][6][valid], f_n[k][6][valid])
                ok &= same and same_id
                print(f"  forward allmap[0:6]    bit-identical: {same}")
                print(f"  forward allmap[6] id   bit-identical where alpha>0.5: {same_id} "
                      f"({(~valid).sum().item()} px undefined)")
                continue
            same = torch.equal(f_o[k], f_n[k])
            ok &= same
            print(f"  forward {k:13s} bit-identical: {same}")
        for k in g_o:
            r_new, r_self = max_rel(g_n[k], g_o[k]), max_rel(g_o2[k], g_o[k])
            # level 2: ours vs original must be within ~TS+'s own run-to-run noise
            good = r_new <= max(10 * r_self, 1e-6)
            ok &= good
            print(f"  grad    {k:13s} max rel diff ours-vs-orig {r_new:.2e} | orig-vs-orig {r_self:.2e} "
                  f"| bit-identical {torch.equal(g_n[k], g_o[k])} | {'OK' if good else 'FAIL'}")
            if r_self > 1e-3:
                print(f"    note: TS+ itself is not reproducible for '{k}' (orig-vs-orig {r_self:.1e}); "
                      f"this check is weak until that is understood (notes/decisions.md).")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
