#!/usr/bin/env python3
"""Geometry check (plan step 8.1 "without worse geometry"): depth of the opaque mesh vs COLMAP's 3D points.

Garden has no ground-truth depth, so the reference is the COLMAP sparse reconstruction: for every test view we take
the 3D points that COLMAP *observed in that photo* (so they are visible there, not occluded) and compare their
camera depth with the depth of the mesh at the same pixel, rendered as a game engine sees it (opaque triangles,
z-buffer = the collider surface; renderer from eval_engine_view.py).
Only reliable points are used: track length >= 3 and reprojection error <= 1.5 px.

Per run (mean over the 24 test views, all points pooled):
  median relative depth error |d_mesh - d_point| / d_point, share of points within 1 % / 5 %,
  share of points where the mesh is clearly in front (> 5 %, = a floater blocks the view), behind (> 5 %, = hole or
  surface too far) and with no triangle at all.
Caveat: TS+ starts from these same points, so this measures how well the final surface still agrees with SfM,
not an independent ground truth. Same reference for all methods, so the comparison between runs is fair.

Usage: python scripts/eval_geometry.py <model_dir> [<model_dir> ...] --colmap <dataset>/sparse/0 --images <dataset>/images_8
       -> prints one line per run, writes runs/geometry_eval/<run>.json
"""
import argparse, json, os, struct, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import eval_engine_view as E                      # noqa: E402
from scene.colmap_loader import read_extrinsics_binary  # noqa: E402


def read_points(path):
    """points3D.bin -> {id: (xyz, error, track_length)}"""
    pts = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid, x, y, z, _r, _g, _b, err = struct.unpack("<QdddBBBd", f.read(43))
            tl = struct.unpack("<Q", f.read(8))[0]
            f.seek(8 * tl, 1)
            pts[pid] = (np.array([x, y, z]), err, tl)
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--colmap", required=True); p.add_argument("--images", required=True)
    p.add_argument("--min_track", type=int, default=3); p.add_argument("--max_err", type=float, default=1.5)
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "../../runs/geometry_eval"))
    a = p.parse_args()
    dev = "cuda"
    pts = read_points(os.path.join(a.colmap, "points3D.bin"))
    imgs = {im.name.rsplit(".", 1)[0]: im for im in read_extrinsics_binary(os.path.join(a.colmap, "images.bin")).values()}
    os.makedirs(a.out_dir, exist_ok=True)
    for model in a.models:
        m = E.load_model(model, dev)
        cams = sorted(json.load(open(os.path.join(model, "cameras.json"))), key=lambda c: c["img_name"])
        rel_all, views = [], []
        for cam in cams[::8]:                                    # test views (llffhold 8, as TS+ --eval)
            W, H = Image.open(os.path.join(a.images, cam["img_name"] + ".JPG")).size
            ids = [i for i in imgs[cam["img_name"]].point3D_ids if i >= 0 and i in pts]
            P = np.array([pts[i][0] for i in ids if pts[i][2] >= a.min_track and pts[i][1] <= a.max_err])
            out = {}
            E.render(m, cam, W, H, "zbuf", False, out=out)
            depth = out["depth"]
            # project the points exactly like the renderer does
            s = W / cam["width"]
            R = torch.tensor(cam["rotation"], dtype=torch.float64, device=dev)
            t = torch.tensor(cam["position"], dtype=torch.float64, device=dev)
            pc = (torch.tensor(P, dtype=torch.float64, device=dev) - t) @ R
            Z = pc[:, 2]
            x = (pc[:, 0] / Z * cam["fx"] * s + W / 2).floor().long()
            y = (pc[:, 1] / Z * cam["fy"] * s + H / 2).floor().long()
            ok = (Z > 0.01) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
            d = depth[y[ok], x[ok]]
            rel = ((d - Z[ok]) / Z[ok]).cpu().numpy()            # signed: < 0 = mesh in front of the point
            rel_all.append(rel)
            views.append(dict(name=cam["img_name"], n=int(len(rel)), median_abs=float(np.median(np.abs(rel[np.isfinite(rel)])))))
        r = np.concatenate(rel_all); fin = np.isfinite(r); ar = np.abs(r[fin])
        res = dict(points=int(len(r)), median_abs_rel=float(np.median(ar)),
                   within_1pct=float((ar <= 0.01).sum() / len(r)), within_5pct=float((ar <= 0.05).sum() / len(r)),
                   front_5pct=float((r[fin] < -0.05).sum() / len(r)), behind_5pct=float((r[fin] > 0.05).sum() / len(r)),
                   no_surface=float((~fin).sum() / len(r)), triangles=int(m["faces"].shape[0]))
        run = os.path.basename(model.rstrip("/"))
        json.dump(dict(model=model, **res, views=views), open(os.path.join(a.out_dir, run + ".json"), "w"), indent=1)
        print(f"{run}: {res['triangles']} tris, {res['points']} pts | median |rel| {100*res['median_abs_rel']:.2f} % | "
              f"<1% {100*res['within_1pct']:.1f} % <5% {100*res['within_5pct']:.1f} % | in front {100*res['front_5pct']:.1f} % "
              f"behind {100*res['behind_5pct']:.1f} % none {100*res['no_surface']:.1f} %", flush=True)


if __name__ == "__main__":
    main()
