#!/usr/bin/env python3
"""Does the proxy match what is seen? Depth of the proxy mesh vs the render surface.

For views of a trained model: the exact TS+ depth (physproxy.depth) and the
z-buffer of the proxy mesh (pytorch3d), compared on pixels whose visible render
surface lies inside the proxy box (the proxy only exists there):
  agree       |z_proxy - z_render| <= tol * z_render     (what you see is what you hit)
  blocks      proxy in front of the visible surface by more than tol
              (an invisible wall / floater in the collider)
  see_through proxy behind the visible surface by more than tol, or no proxy hit
              (a ray would pass through something visible)
Saves RGB | render depth | proxy depth | error class (green agree / red blocks /
blue see-through) per view, and a JSON summary.

Usage:
  python physproxy/eval/proxy_views.py -s SCENE -i images_8 -m MODEL --proxy_mesh MESH.ply --proxy_sdf SDF.pt
"""
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physproxy.compat import use_cached_metric3d  # noqa: E402
use_cached_metric3d()
from arguments import ModelParams, PipelineParams  # noqa: E402
from scene import Scene  # noqa: E402
from scene.triangle_model import TriangleModel  # noqa: E402
from triangle_renderer import render  # noqa: E402
from physproxy.depth import exact_depth, pixel_alpha, view_rays  # noqa: E402
from physproxy.eval.zbuffer import p3d_zbuffer  # noqa: E402
from physproxy.eval.check_depth import colorize  # noqa: E402


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--proxy_mesh", required=True)
    parser.add_argument("--proxy_sdf", required=True, help="sdf_<it>.pt (for the box and frame)")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--views", type=int, nargs="*", default=None, help="default: all views of the split")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--tol", type=float, default=0.02, help="relative depth tolerance")
    parser.add_argument("--save_images", type=int, default=6, help="save pictures for the first N views")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    dataset, pipe = lp.extract(args), pp.extract(args)
    dataset.eval = True
    out_dir = args.out or os.path.join(os.path.dirname(args.proxy_mesh), "views_" + os.path.splitext(os.path.basename(args.proxy_mesh))[0])
    os.makedirs(out_dir, exist_ok=True)

    sd = torch.load(args.proxy_sdf, map_location="cuda")["state_dict"]
    frame, lo, hi = sd["frame"], sd["aabb_min"], sd["aabb_max"]
    m = trimesh.load(args.proxy_mesh, process=False)
    PV = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device="cuda")
    PF = torch.tensor(np.asarray(m.faces), dtype=torch.int64, device="cuda")

    with torch.no_grad():
        tri = TriangleModel(dataset.sh_degree)
        scene = Scene(dataset, tri, None, None, load_iteration=args.iteration, shuffle=False)
        tri.scaling = args.scale
        cams = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
        summary = compare_views(cams, tri, pipe, PV, PF, frame, lo, hi, tol=args.tol, out_dir=out_dir,
                                save_images=args.save_images, views=args.views)
        summary["proxy_mesh"] = args.proxy_mesh
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
        print(json.dumps({k: v for k, v in summary.items() if k != "per_view"}, indent=1))


@torch.no_grad()
def compare_views(cams, tri, pipe, PV, PF, frame, lo, hi, tol=0.02, out_dir=None, save_images=0, views=None):
    """Classify in-box pixels of each view as agree / blocks / see_through (see module doc).
    Renders at tri.scaling. Returns the summary dict."""
    from PIL import Image
    idx = views if views else list(range(len(cams)))
    V, F = tri.get_vertices, tri.get_triangle_indices
    tot = {"agree": 0, "blocks": 0, "see_through": 0, "n": 0}
    per_view = []
    gaps = []   # distance along the ray (scene units) by which the proxy is in front, on "blocks" pixels
    for k, vi in enumerate(idx):
        view = cams[vi]
        pkg = render(view, tri, pipe, torch.zeros(3, device="cuda"))
        ex = exact_depth(view, V, F, pkg["rend_ids"], pixel_alpha(pkg))
        z, valid = ex["depth"], ex["valid"]
        H, W = z.shape
        pl = ex["points_world"] @ frame.T
        in_box = valid & ((pl >= lo) & (pl <= hi)).all(-1)
        zp, fp = p3d_zbuffer(view, PV, PF, H, W)
        hit = fp >= 0
        rel = (zp - z) / z.clamp_min(1e-6)
        agree = in_box & hit & (rel.abs() <= tol)
        blocks = in_box & hit & (rel < -tol)
        see = in_box & ~(agree | blocks)
        n = int(in_box.sum())
        dn = view_rays(view, H, W, z.device).norm(dim=-1)
        gaps.append(((z - zp) * dn)[blocks])
        per_view.append({"view": view.image_name, "in_box_frac": n / (H * W),
                         "agree": int(agree.sum()) / max(n, 1), "blocks": int(blocks.sum()) / max(n, 1),
                         "see_through": int(see.sum()) / max(n, 1)})
        for key, msk in [("agree", agree), ("blocks", blocks), ("see_through", see)]:
            tot[key] += int(msk.sum())
        tot["n"] += n
        if out_dir is not None and k < save_images:
            os.makedirs(out_dir, exist_ok=True)
            st = tri.scaling
            rgb = (pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            zr = (z * valid)[::st, ::st].cpu().numpy()
            zpp = (zp * hit)[::st, ::st].cpu().numpy()
            lo_d, hi_d = np.percentile(zr[zr > 0], [2, 98])
            cls = np.zeros((H, W, 3), np.uint8)
            cls[agree.cpu().numpy()] = (40, 170, 60)
            cls[blocks.cpu().numpy()] = (220, 50, 40)
            cls[see.cpu().numpy()] = (50, 90, 230)
            cls[(~in_box).cpu().numpy()] = (60, 60, 60)
            row = np.concatenate([rgb, colorize(zr, lo_d, hi_d), colorize(zpp, lo_d, hi_d), cls[::st, ::st]], axis=1)
            Image.fromarray(row).save(os.path.join(out_dir, f"{view.image_name}.png"))
    g = torch.cat(gaps) if gaps else torch.zeros(0, device="cuda")
    gq = {}
    if g.numel():
        sub = g[torch.randperm(g.numel(), device=g.device)[:2_000_000]]
        gq = {f"blocks_gap_p{q}": float(torch.quantile(sub, q / 100)) for q in (50, 90, 99)}
        for t in (0.02, 0.05, 0.1, 0.2):
            gq[f"blocks_frac_gap_over_{t}"] = float((g > t).float().mean()) * tot["blocks"] / max(tot["n"], 1)
    return {**gq, "tol": tol, "views": len(idx), "in_box_pixels": tot["n"],
            "agree": tot["agree"] / max(tot["n"], 1), "blocks": tot["blocks"] / max(tot["n"], 1),
            "see_through": tot["see_through"] / max(tot["n"], 1), "per_view": per_view}


if __name__ == "__main__":
    main()
