#!/usr/bin/env python3
"""Phase 7: export a trained physproxy model for the Unity benchmark (unity/Assets/PhysProxy).

Writes OUT/ (copy it to <UnityProject>/Assets/StreamingAssets/physproxy/<scene>/):
  render.ppm      visual mesh: the full TS+ render mesh with per-vertex colors (SH DC)
  collision.ppm   our collision mesh (physproxy/export_collision.py)
  naive.ppm       naive-cleanup collider (physproxy/eval/naive_cleanup.py)
  soup.ppm        the render triangles inside the proxy box, used as a collider ("render as-is")
  drops.csv       sphere drop start points + where a straight-down ray first meets the render mesh
  walks.csv       capsule walk start/goal points on the ground
  meta.json       frame, scale and provenance

Frame: Unity is left-handed and Y-up. Points go to the proxy box frame (local z = up), then
Unity (x, y, z) = s * (lx - cx, lz - floor, ly - cy), where s = --units_to_m, (cx, cy) is the box
centre and floor is the bottom of the box. Each triangle's index order is reversed on export: after
the axis swap, the cross product Unity uses for normals (RecalculateNormals, PhysX back-face
tests) would otherwise point into our closed meshes. Found in the first Unity run (2026-11-11):
spheres with continuous collision detection passed through the collision mesh's top surface from
outside and came to rest on the inside of the box floor.

Mesh format (.ppm, little endian): b"PPM1", int32 n_vertices, int32 n_faces, int32 has_color,
float32[n_vertices*3] positions, uint8[n_vertices*4] RGBA if has_color, int32[n_faces*3] indices.

  python unity/export_for_unity.py -s /data/manousis/360_v2/garden -i images_8 -m MODEL [--units_to_m 1.0]
"""
import csv
import json
import os
import struct
import sys
from argparse import ArgumentParser

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def write_ppm(path, v, f, rgba=None):
    v = np.ascontiguousarray(v, np.float32)
    f = np.ascontiguousarray(np.asarray(f)[:, ::-1], np.int32)     # see the frame note above
    with open(path, "wb") as fh:
        fh.write(b"PPM1")
        fh.write(struct.pack("<iii", v.shape[0], f.shape[0], 1 if rgba is not None else 0))
        fh.write(v.astype("<f4").tobytes())
        if rgba is not None:
            fh.write(np.ascontiguousarray(rgba, np.uint8).tobytes())
        fh.write(f.astype("<i4").tobytes())


def main():
    from physproxy.compat import use_cached_metric3d
    use_cached_metric3d()
    from arguments import ModelParams
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from physproxy.eval.mesh_stats import load_mesh
    from physproxy.eval.raycast import cast, free_space_origins, load_obj_mesh
    from physproxy.export_collision import latest

    parser = ArgumentParser()
    lp = ModelParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sdf", default=None)
    parser.add_argument("--units_to_m", type=float, default=1.0,
                        help="metres per scene unit (garden: cameras circle at ~3.7 units, so ~1 is plausible)")
    parser.add_argument("--n_drops", type=int, default=2000)
    parser.add_argument("--n_walks", type=int, default=300)
    parser.add_argument("--walk_len", type=float, nargs=2, default=(1.0, 3.0), help="straight-line length range (units)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    dataset = lp.extract(args)
    dataset.eval = True
    M = args.model_path
    out = args.out or os.path.join(M, "unity")
    os.makedirs(out, exist_ok=True)
    sdf_path = args.sdf or latest(os.path.join(M, "proxy", "sdf_*.pt"))
    sd = torch.load(sdf_path, map_location="cuda")["state_dict"]
    frame, lo, hi = sd["frame"].float(), sd["aabb_min"].float(), sd["aabb_max"].float()
    c = 0.5 * (lo + hi)
    s = args.units_to_m

    def to_unity(p):                      # world [N,3] torch -> Unity [N,3] numpy
        loc = p.float() @ frame.T
        u = torch.stack([loc[:, 0] - c[0], loc[:, 2] - lo[2], loc[:, 1] - c[1]], -1) * s
        return u.cpu().numpy()

    # Meshes.
    render_dir = latest(os.path.join(M, "point_cloud", "iteration_*"))
    ck = torch.load(os.path.join(render_dir, "point_cloud_state_dict.pt"), map_location="cuda")
    Vr = ck["triangles_points"].float()
    Fr = ck["_triangle_indices"].long()
    rgb = (0.28209479177387814 * ck["features_dc"][:, 0].float() + 0.5).clamp(0, 1)
    rgba = torch.cat([rgb, torch.ones_like(rgb[:, :1])], -1).mul(255).round().byte().cpu().numpy()
    write_ppm(os.path.join(out, "render.ppm"), to_unity(Vr), Fr.cpu().numpy(), rgba)
    loc = Vr @ frame.T
    inside = ((loc >= lo) & (loc <= hi)).all(-1)
    soup_f = Fr[inside[Fr].all(1)]
    write_ppm(os.path.join(out, "soup.ppm"), to_unity(Vr), soup_f.cpu().numpy())
    for name, path in [("collision", os.path.join(M, "collision", "collision.obj")),
                       ("naive", os.path.join(M, "collision", "naive.obj"))]:
        v, f = load_mesh(path)
        write_ppm(os.path.join(out, f"{name}.ppm"), to_unity(torch.tensor(v, device="cuda")), f)

    # Probe points: seen-as-empty origins (as in Phase 6) and where they land on the render mesh.
    tri = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, tri, None, None, load_iteration=args.iteration, shuffle=False)
    cams = scene.getTrainCameras()
    o = free_space_origins(cams, Vr, Fr, frame, lo, hi, 4 * args.n_drops, seed=args.seed)
    down = (-frame[2]).expand(o.shape[0], 3).contiguous()
    t = cast(o, down, Vr, Fr, float(hi[2] - lo[2]) + 0.1)
    to_floor = (o @ frame.T)[:, 2] - lo[2]
    ok = torch.isfinite(t) & (t <= to_floor + 0.05)
    land = o + down * torch.where(ok, t, torch.zeros_like(t))[:, None]
    ou, lu = to_unity(o), to_unity(land)
    okn = ok.cpu().numpy()
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(np.flatnonzero(okn))[:args.n_drops]
    with open(os.path.join(out, "drops.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "x", "y", "z", "ref_land_y"])
        for k, i in enumerate(idx):
            w.writerow([k, *np.round(ou[i], 5), round(float(lu[i, 1]), 5)])

    # Walks: start and goal on the ground (landing points within 0.3 units of the low ground level).
    gl = lu[okn]
    ground = np.percentile(gl[:, 1], 5)
    g = gl[gl[:, 1] < ground + 0.3 * s]
    walks, tries = [], 0
    while len(walks) < args.n_walks and tries < 100000:
        tries += 1
        a, b = g[rng.integers(len(g))], g[rng.integers(len(g))]
        d = np.linalg.norm((a - b)[[0, 2]])
        if args.walk_len[0] * s <= d <= args.walk_len[1] * s:
            walks.append((a, b))
    with open(os.path.join(out, "walks.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "sx", "sy", "sz", "gx", "gy", "gz"])
        for k, (a, b) in enumerate(walks):
            w.writerow([k, *np.round(a, 5), *np.round(b, 5)])

    meta = dict(model=M, sdf=sdf_path, render=render_dir, units_to_m=s, seed=args.seed,
                frame_rows=frame.cpu().tolist(), box_lo=lo.cpu().tolist(), box_hi=hi.cpu().tolist(),
                unity_from_local="(lx - cx, lz - lo_z, ly - cy) * units_to_m",
                box_size_unity=[float((hi - lo)[0]) * s, float((hi - lo)[2]) * s, float((hi - lo)[1]) * s],
                n_drops=int(len(idx)), n_walks=len(walks), ground_y=float(ground))
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[unity] wrote {out}: render {Fr.shape[0]} faces, soup {soup_f.shape[0]}, "
          f"{len(idx)} drops, {len(walks)} walks, box {meta['box_size_unity']} m")


if __name__ == "__main__":
    main()
