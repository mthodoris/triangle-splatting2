#!/usr/bin/env python3
"""Phase 6: engine-independent raycast agreement between a reference mesh and test meshes.

What a physics engine does with a collider is mostly ray and sweep queries, so: cast the same rays
against the reference (the render mesh: what you see) and against each test collider, and compare
hit / miss and hit distance within `--max_dist`.

Ray origins are points the cameras see as empty: a training camera ray, a random fraction of the
way to the first reference surface it hits (minus `--margin`), kept if inside the proxy box. Two
direction sets: uniform on the sphere ("random") and straight down along the box's up axis
("down": what a character or a dropped object queries), both within max_dist. Plus a drop test:
down from every origin to the box floor, run on the reference too; a miss = falls through the world.

Per ray (d = first-hit distance, inf if none within max_dist):
  agree       both miss, or both hit with |d_test - d_ref| <= tol
  early       test hits before the reference by more than tol, or hits where the reference misses
              (invisible wall)
  late        test hits after the reference by more than tol, or misses where the reference hits
              (passes through a visible surface: fall-through)

Ray casting is pure PyTorch on the GPU: Moller-Trumbore, brute force per cell of a coarse grid
over the ray origins (only faces whose bounds overlap the cell grown by max_dist are tested).

  python physproxy/eval/raycast.py -s SCENE -i images_8 -m MODEL --sdf SDF.pt \
      --ref MODEL/collision/render.obj --test ours=MODEL/collision/collision.obj naive=...obj
"""
import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ----------------------------------------------------------------------------- ray casting
def _mt(o, d, v0, e1, e2, t_max, eps=1e-9):
    """First hit of rays [R] against triangles [T] (brute force). Returns t [R] (inf if none)."""
    p = torch.cross(d[:, None, :].expand(-1, v0.shape[0], -1), e2[None].expand(o.shape[0], -1, -1), dim=-1)
    det = (e1[None] * p).sum(-1)
    ok = det.abs() > eps
    inv = 1.0 / torch.where(ok, det, torch.ones_like(det))
    s = o[:, None, :] - v0[None]
    u = (s * p).sum(-1) * inv
    del p
    q = torch.cross(s, e1[None].expand_as(s), dim=-1)
    del s
    v = (d[:, None, :] * q).sum(-1) * inv
    t = (e2[None] * q).sum(-1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (t <= t_max)
    t = torch.where(hit, t, torch.full_like(t, float("inf")))
    return t.min(1).values


@torch.no_grad()
def cast(o, d, V, F, max_dist, cell=1.0, budget=2 ** 24):
    """First-hit distance along unit directions d from origins o against mesh (V, F), up to max_dist.
    Returns t [R] (inf = no hit within max_dist)."""
    o, d = o.float(), d.float()
    tri = V[F.long()].float()                                   # [T,3,3]
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    fmin, fmax = tri.min(1).values, tri.max(1).values
    out = torch.full((o.shape[0],), float("inf"), device=o.device)
    key = torch.floor(o / cell).long()
    cells, inv = torch.unique(key, dim=0, return_inverse=True)
    for c in range(cells.shape[0]):
        rid = torch.nonzero(inv == c, as_tuple=True)[0]
        lo = cells[c].float() * cell - max_dist
        hi = lo + cell + 2 * max_dist
        fid = torch.nonzero(((fmax >= lo) & (fmin <= hi)).all(-1), as_tuple=True)[0]
        if fid.numel() == 0:
            continue
        rc = max(1, min(rid.numel(), budget // max(fid.numel(), 1)))
        fc = max(1, budget // rc)
        for i in range(0, rid.numel(), rc):
            r = rid[i:i + rc]
            best = torch.full((r.numel(),), float("inf"), device=o.device)
            for j in range(0, fid.numel(), fc):
                f = fid[j:j + fc]
                best = torch.minimum(best, _mt(o[r], d[r], v0[f], e1[f], e2[f], max_dist))
            out[r] = best
    return out


# ----------------------------------------------------------------------------- ray sets
def unit_sphere(n, gen, device):
    x = torch.randn(n, 3, generator=gen, device=device)
    return x / x.norm(dim=-1, keepdim=True)


@torch.no_grad()
def free_space_origins(cams, V, F, frame, lo, hi, n, margin=0.05, scale=4, seed=0):
    """`n` points seen as empty from the cameras (in front of the first reference surface), inside the box."""
    from physproxy.depth import view_rays, view_to_world
    from physproxy.eval.zbuffer import p3d_zbuffer
    gen = torch.Generator(device="cuda").manual_seed(seed)
    per = int(np.ceil(n / len(cams))) * 3
    pts = []
    for view in cams:
        H, W = view.image_height // scale, view.image_width // scale
        z, fid = p3d_zbuffer(view, V, F, H, W)
        dirs = view_rays(view, H, W, V.device)                  # view space, z = 1
        hit = torch.nonzero((fid >= 0).reshape(-1), as_tuple=True)[0]
        if hit.numel() == 0:
            continue
        pick = hit[torch.randint(hit.numel(), (per,), generator=gen, device="cuda")]
        zz = z.reshape(-1)[pick]
        dn = dirs.reshape(-1, 3)[pick]
        along = dn.norm(dim=-1)
        z_max = (zz - margin / along).clamp_min(0)              # margin is a distance along the ray
        u = torch.rand(per, generator=gen, device="cuda")
        p_view = dn * (z_max * u)[:, None]
        pw = view_to_world(view, p_view)
        pl = pw @ frame.T
        pts.append(pw[((pl >= lo) & (pl <= hi)).all(-1)])
    p = torch.cat(pts)
    return p[torch.randperm(p.shape[0], generator=gen, device="cuda")[:n]]


def compare(t_ref, t_test, tol):
    ref_hit, test_hit = torch.isfinite(t_ref), torch.isfinite(t_test)
    both = ref_hit & test_hit
    diff = torch.where(both, t_test - t_ref, torch.zeros_like(t_ref))
    agree = (~ref_hit & ~test_hit) | (both & (diff.abs() <= tol))
    early = (test_hit & ~ref_hit) | (both & (diff < -tol))
    late = (ref_hit & ~test_hit) | (both & (diff > tol))
    n = t_ref.numel()
    out = dict(n=n, agree=float(agree.sum()) / n, early_invisible_wall=float(early.sum()) / n,
               late_fall_through=float(late.sum()) / n, ref_hit=float(ref_hit.sum()) / n,
               test_hit=float(test_hit.sum()) / n,
               hit_agree=float((ref_hit == test_hit).sum()) / n)
    # Conditional rates: of the rays that hit the reference, how many does the collider stop within
    # tol (recall); of those that miss it, how many does the collider wrongly stop (invisible walls).
    out["recall_given_ref_hit"] = float((both & (diff.abs() <= tol)).sum()) / max(int(ref_hit.sum()), 1)
    out["passes_given_ref_hit"] = float((ref_hit & ~test_hit).sum()) / max(int(ref_hit.sum()), 1)
    out["wall_given_ref_miss"] = float((test_hit & ~ref_hit).sum()) / max(int((~ref_hit).sum()), 1)
    if both.any():
        ad = diff[both].abs()
        out.update(both_hit_absdiff_p50=float(ad.median()), both_hit_absdiff_p90=float(torch.quantile(ad[:10 ** 6], 0.9)))
    return out


def load_obj_mesh(path):
    from physproxy.eval.mesh_stats import load_mesh
    v, f = load_mesh(path)
    return (torch.tensor(v, dtype=torch.float32, device="cuda"), torch.tensor(f, dtype=torch.int64, device="cuda"))


def main():
    from physproxy.compat import use_cached_metric3d
    use_cached_metric3d()
    from arguments import ModelParams
    from scene import Scene
    from scene.triangle_model import TriangleModel

    parser = ArgumentParser()
    lp = ModelParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sdf", required=True, help="sdf_<it>.pt (box and frame)")
    parser.add_argument("--ref", required=True, help="reference mesh (the render mesh)")
    parser.add_argument("--test", nargs="+", required=True, help="name=path of each test collider")
    parser.add_argument("--n_rays", type=int, default=200000)
    parser.add_argument("--max_dist", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=0.05, help="absolute hit-distance tolerance (scene units)")
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="JSON output (default MODEL/raycast.json)")
    args = parser.parse_args()
    dataset = lp.extract(args)
    dataset.eval = True

    sd = torch.load(args.sdf, map_location="cuda")["state_dict"]
    frame, lo, hi = sd["frame"], sd["aabb_min"], sd["aabb_max"]
    Vr, Fr = load_obj_mesh(args.ref)
    tri = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, tri, None, None, load_iteration=args.iteration, shuffle=False)
    cams = scene.getTrainCameras()

    t0 = time.time()
    o = free_space_origins(cams, Vr, Fr, frame, lo, hi, args.n_rays, margin=args.margin, seed=args.seed)
    gen = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    dirsets = {"random": unit_sphere(o.shape[0], gen, "cuda"),
               "down": (-frame[2]).float().expand(o.shape[0], 3).contiguous()}
    res = dict(ref=args.ref, sdf=args.sdf, n_origins=int(o.shape[0]), max_dist=args.max_dist, tol=args.tol,
               margin=args.margin, seed=args.seed, results={})
    t_ref = {k: cast(o, d, Vr, Fr, args.max_dist) for k, d in dirsets.items()}
    print(f"[raycast] {o.shape[0]} origins; reference hit rate "
          + ", ".join(f"{k} {float(torch.isfinite(t).float().mean()):.3f}" for k, t in t_ref.items())
          + f" ({time.time() - t0:.0f} s)")
    for spec in args.test:
        name, path = spec.split("=", 1)
        Vt, Ft = load_obj_mesh(path)
        res["results"][name] = {"path": path, "faces": int(Ft.shape[0])}
        for k, d in dirsets.items():
            r = compare(t_ref[k], cast(o, d, Vt, Ft, args.max_dist), args.tol)
            res["results"][name][k] = r
            print(f"[raycast] {name:>10} {k:>6}: agree {r['agree']:.3f} | of ref hits: caught within tol "
                  f"{r['recall_given_ref_hit']:.3f}, passes {r['passes_given_ref_hit']:.3f} | of ref misses: "
                  f"wall {r['wall_given_ref_miss']:.3f}")
    # Drop test: straight down from every origin until the box floor. Every such ray should land
    # (the floor of the box is ground); a miss means a dropped object falls through the world.
    up = frame[2].float()
    to_floor = (o @ frame.T)[:, 2] - lo[2]
    h = float((hi - lo)[2])
    down = (-up).expand(o.shape[0], 3).contiguous()

    def drop(V, F):
        t = cast(o, down, V, F, h + 0.1)
        return torch.where(t <= to_floor + 0.05, t, torch.full_like(t, float("inf")))

    t_drop_ref = drop(Vr, Fr)
    res["drop"] = {"ref": {"falls_through": float(torch.isinf(t_drop_ref).float().mean())}}
    print(f"[raycast] drop test: reference falls through {res['drop']['ref']['falls_through']:.4f}")
    for spec in args.test:
        name, path = spec.split("=", 1)
        Vt, Ft = load_obj_mesh(path)
        t = drop(Vt, Ft)
        both = torch.isfinite(t) & torch.isfinite(t_drop_ref)
        diff = (t - t_drop_ref)[both]
        r = {"falls_through": float(torch.isinf(t).float().mean()),
             "lands_higher_than_ref_by_tol": float((diff < -args.tol).float().sum()) / o.shape[0],
             "lands_lower_than_ref_by_tol": float((diff > args.tol).float().sum()) / o.shape[0],
             "landing_absdiff_p50": float(diff.abs().median()) if diff.numel() else None,
             "landing_absdiff_p90": float(torch.quantile(diff.abs()[:10 ** 6], 0.9)) if diff.numel() else None}
        res["drop"][name] = r
        print(f"[raycast] drop test: {name:>10} falls through {r['falls_through']:.4f}, lands higher "
              f"{r['lands_higher_than_ref_by_tol']:.4f} / lower {r['lands_lower_than_ref_by_tol']:.4f} than ref "
              f"(|diff| p50 {r['landing_absdiff_p50']:.4f})")
    out = args.out or os.path.join(args.model_path, "raycast.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[raycast] wrote {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
