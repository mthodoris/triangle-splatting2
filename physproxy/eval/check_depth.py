#!/usr/bin/env python3
"""Phase 1 check: exact per-pixel depth from rend_ids + ray/plane intersection.

For a few views of a trained model it
  1. renders with the upstream renderer and computes physproxy.depth.exact_depth,
  2. checks the hit lies on the visible triangle (barycentrics >= -tol),
  3. cross-checks against an independent z-buffer (pytorch3d rasterize_meshes,
     perspective-correct) on the same mesh: triangle-id agreement and depth error,
  4. checks units/alignment against COLMAP: the z of each SfM point observed in
     the image vs the exact depth at its keypoint,
  5. compares with the rasterizer's own (centroid) median depth,
  6. saves RGB | centroid depth | exact depth | |difference| as PNG per view.

Usage:
  python physproxy/eval/check_depth.py -s SCENE -i images_8 -m MODEL [--views 0 5 10] [--scale 4]
"""
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physproxy.compat import use_cached_metric3d  # noqa: E402
use_cached_metric3d()
from arguments import ModelParams, PipelineParams  # noqa: E402
from scene import Scene  # noqa: E402
from scene.triangle_model import TriangleModel  # noqa: E402
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary  # noqa: E402
from triangle_renderer import render  # noqa: E402
from physproxy.depth import exact_depth, pixel_alpha, world_to_view  # noqa: E402
from physproxy.eval.zbuffer import p3d_zbuffer  # noqa: E402


def colorize(d, lo, hi):
    import matplotlib
    cm = matplotlib.colormaps["turbo"]
    x = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
    out = (cm(x)[..., :3] * 255).astype(np.uint8)
    out[d <= 0] = 0
    return out


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--views", type=int, nargs="+", default=[0, 6, 12, 18])
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--scale", type=int, default=4, help="render upsampling (training ends at 4)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--bary_tol", type=float, default=1e-3)
    args = parser.parse_args()
    dataset, pipe = lp.extract(args), pp.extract(args)
    dataset.eval = True
    out_dir = args.out or os.path.join(dataset.model_path, "physproxy_depth_check")
    os.makedirs(out_dir, exist_ok=True)

    from PIL import Image
    with torch.no_grad():
        tri = TriangleModel(dataset.sh_degree)
        scene = Scene(dataset, tri, None, None, load_iteration=args.iteration, shuffle=False)
        tri.scaling = args.scale
        bg = torch.zeros(3, device="cuda")
        cams = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()

        sparse = os.path.join(dataset.source_path, "sparse/0")
        extr = {os.path.basename(e.name).split(".")[0]: e for e in read_extrinsics_binary(os.path.join(sparse, "images.bin")).values()}
        intr = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
        from scene.colmap_loader import read_points3D_binary
        xyz, _, _ = read_points3D_binary(os.path.join(sparse, "points3D.bin"))
        # points3D.bin ids are not contiguous: rebuild id -> xyz.
        import struct
        ids = []
        with open(os.path.join(sparse, "points3D.bin"), "rb") as fid:
            n = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(n):
                props = struct.unpack("<QdddBBBd", fid.read(43))
                ids.append(props[0])
                tl = struct.unpack("<Q", fid.read(8))[0]
                fid.read(8 * tl)
        id2row = {pid: r for r, pid in enumerate(ids)}

        V, F = tri.get_vertices, tri.get_triangle_indices
        report = {"model": dataset.model_path, "scale": args.scale, "views": []}
        for vi in args.views:
            view = cams[vi % len(cams)]
            pkg = render(view, tri, pipe, bg)
            ex = exact_depth(view, V, F, pkg["rend_ids"], pixel_alpha(pkg))
            H, W = ex["depth"].shape
            valid = ex["valid"]
            z = ex["depth"]
            r = {"view": view.image_name, "render_hw": [H, W], "valid_frac": valid.float().mean().item()}

            b = ex["bary"][valid]
            r["on_triangle_frac"] = (b.min(dim=1).values >= -args.bary_tol).float().mean().item()
            r["bary_min_p01"] = torch.quantile(b.min(dim=1).values.float()[:1_000_000], 0.01).item()

            cen = pkg["depth_full"][0]
            rel_c = ((cen - z).abs() / z)[valid]
            r["centroid_vs_exact_rel"] = {"mean": rel_c.mean().item(), "p50": rel_c.median().item(),
                                          "p99": torch.quantile(rel_c[:1_000_000], 0.99).item()}

            zb, face = p3d_zbuffer(view, V, F, H, W)
            both = valid & (face >= 0)
            same = both & (face == ex["tri_id"])
            r["p3d_id_agree_frac"] = (same.sum() / both.sum().clamp_min(1)).item()
            rel = ((zb - z).abs() / z)[same]
            r["p3d_depth_rel_err_same_id"] = {"mean": rel.mean().item(), "max": rel.max().item()}
            relall = ((zb - z) / z)[both]
            r["p3d_depth_rel_all"] = {"p50_abs": relall.abs().median().item(),
                                      "frac_p3d_in_front_by_1pct": (relall < -0.01).float().mean().item()}
            # Pixels where the two rasterizers pick different triangles: same surface
            # (stacked/adjacent triangles) or a triangle our renderer culls?
            reld = ((zb - z) / z)[both & ~same]
            cull = torch.zeros(F.shape[0], dtype=torch.bool, device="cuda")
            cull[pkg["radii"] == 0] = True               # not drawn in this view by the TS+ rasterizer
            fd = face[both & ~same]
            r["p3d_diff_id"] = {"frac_of_pixels": (reld.numel() / max(both.sum().item(), 1)),
                                "abs_rel_p50": reld.abs().median().item() if reld.numel() else 0.0,
                                "abs_rel_p90": torch.quantile(reld.abs()[:1_000_000], 0.9).item() if reld.numel() else 0.0,
                                "frac_within_0.1pct": (reld.abs() < 1e-3).float().mean().item() if reld.numel() else 1.0,
                                "frac_p3d_face_culled_by_ts": cull[fd].float().mean().item() if reld.numel() else 0.0}

            e = extr[view.image_name]
            sc = W / intr[e.camera_id].width
            m = e.point3D_ids >= 0
            uv = e.xys[m] * sc
            P = torch.tensor(np.array([xyz[id2row[p]] for p in e.point3D_ids[m]]), device="cuda", dtype=torch.float32)
            zs = world_to_view(view, P)[:, 2]
            u = torch.tensor(uv[:, 0], device="cuda").floor().long().clamp(0, W - 1)
            v = torch.tensor(uv[:, 1], device="cuda").floor().long().clamp(0, H - 1)
            ok = valid[v, u]
            rs = (z[v, u][ok] - zs[ok]) / zs[ok]
            r["sfm"] = {"n": int(ok.sum()), "rel_err_p50": rs.median().item(), "abs_rel_p50": rs.abs().median().item(),
                        "abs_rel_p90": torch.quantile(rs.abs(), 0.9).item(),
                        "frac_within_5pct": (rs.abs() < 0.05).float().mean().item()}
            report["views"].append(r)
            print(json.dumps(r, indent=1))

            # Picture at native resolution (downsample the render-res maps by striding).
            st = args.scale
            rgb = (pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            zc, ze = cen[::st, ::st].cpu().numpy(), (z * valid)[::st, ::st].cpu().numpy()
            lo, hi = np.percentile(ze[ze > 0], [2, 98])
            diff = np.abs(zc - ze) / np.maximum(ze, 1e-6) * (ze > 0)
            dimg = colorize(diff + (ze > 0) * 1e-9, 0, 0.05)
            row = np.concatenate([rgb, colorize(zc, lo, hi), colorize(ze, lo, hi), dimg], axis=1)
            Image.fromarray(row).save(os.path.join(out_dir, f"{view.image_name}.png"))
            np.save(os.path.join(out_dir, f"{view.image_name}_depth.npy"), ze)

        with open(os.path.join(out_dir, "report.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        print("wrote", out_dir)


if __name__ == "__main__":
    main()
