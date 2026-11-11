#!/usr/bin/env python3
"""Phase 5: collision mesh from the co-trained SDF, plus the render mesh, in the same frame and units.

Pipeline:
  1. Marching cubes on the SDF over the proxy box (`--res` cells on the longest axis; closed at the
     box walls, so it is watertight by construction).
  2. Drop connected components with area < `--min_comp_area` (scene units^2): proxy floaters.
  3. Quadric edge-collapse decimation (pymeshlab) to `--budget` faces, preserving topology. If that
     stalls above the budget (high-genus foliage), steps 1-3 are redone at a lower MC resolution.
  4. Repair: edge collapse leaves a few non-manifold edges. Remove the 1-ring of their vertices,
     close the holes (repeated until watertight), then orient every shell coherently and outward.
  5. Error report, both directions, max (Hausdorff) and percentiles, via pymeshlab sampling:
       - collision vs the full-detail proxy (MC at `--ref_res`): what resolution + decimation lose;
       - collision vs the render mesh (render triangles inside the proxy box).
     `--eps` is checked against these (max and p99); quadric decimation has no hard max-error stop,
     so eps is reported, not enforced.

Writes OUT/collision.obj, OUT/render.obj, OUT/collision_report.json (OUT defaults to MODEL/collision).

  python physproxy/export_collision.py -m output/physproxy/p4b_R0_oneway [--budget 50000 --eps 0.02]
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from physproxy.sdf import SDFField  # noqa: E402
from physproxy.extract import extract_mesh  # noqa: E402
from physproxy.eval.mesh_stats import load_mesh, mesh_stats  # noqa: E402


def latest(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(pattern)
    return max(files, key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[-1]))


def face_components(f):
    """Component label per face (faces sharing an edge)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    F = f.shape[0]
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(1)
    fid = np.tile(np.arange(F), 3)
    _, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    inv_s, fid_s = inv[order], fid[order]
    same = inv_s[1:] == inv_s[:-1]
    a, b = fid_s[:-1][same], fid_s[1:][same]
    g = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(F, F))
    return connected_components(g, directed=False)[1]


def drop_small_components(v, f, min_area):
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    lab = face_components(f)
    comp_area = np.bincount(lab, weights=area)
    keep = comp_area[lab] >= min_area
    f = f[keep]
    used = np.unique(f)
    remap = -np.ones(v.shape[0], np.int64)
    remap[used] = np.arange(used.size)
    info = dict(components_before=int(comp_area.size), components_kept=int((comp_area >= min_area).sum()),
                area_before=float(area.sum()), area_dropped=float(area[~keep].sum()))
    return v[used], remap[f], info


def ml_mesh(v, f):
    import pymeshlab as ml
    return ml.Mesh(vertex_matrix=np.ascontiguousarray(v, np.float64), face_matrix=np.ascontiguousarray(f, np.int32))


def _vf(ms):
    m = ms.current_mesh()
    return m.vertex_matrix().copy(), m.face_matrix().astype(np.int64)


def one_sided(vA, fA, vB, fB, n):
    """Distances from `n` area-uniform samples on mesh A to the closest point of mesh B."""
    import pymeshlab as ml
    ms = ml.MeshSet()
    ms.add_mesh(ml_mesh(vA, fA))
    ms.add_mesh(ml_mesh(vB, fB))
    diag = float(np.linalg.norm(np.ptp(np.concatenate([vA, vB]), 0)))
    r = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=1, savesample=True, samplevert=False,
                                  sampleface=True, samplenum=int(n), maxdist=ml.PureValue(diag))
    d = None
    for i in range(2, ms.mesh_number()):        # the saved "Hausdorff samples" mesh carries the distances
        m = ms.mesh(i)
        if m.vertex_number() > 0 and m.has_vertex_scalar():
            d = m.vertex_scalar_array().copy()
            break
    out = dict(max=float(r["max"]), mean=float(r["mean"]), n=int(r["n_samples"]))
    if d is not None and d.size:
        for q in (50, 90, 95, 99):
            out[f"p{q}"] = float(np.percentile(d, q))
    return out


def two_sided(vA, fA, vB, fB, n):
    ab, ba = one_sided(vA, fA, vB, fB, n), one_sided(vB, fB, vA, fA, n)
    return {"a_to_b": ab, "b_to_a": ba, "hausdorff": max(ab["max"], ba["max"])}


def decimate(v, f, budget):
    """Quadric edge collapse to `budget` faces, topology preserved (without it, high-genus foliage
    collapses into a broken mesh). May stall above the budget; the caller then lowers the MC res."""
    import pymeshlab as ml
    ms = ml.MeshSet()
    ms.add_mesh(ml_mesh(v, f))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(budget), qualitythr=0.3, preservenormal=True,
                                                preservetopology=True, optimalplacement=True,
                                                planarquadric=True, autoclean=True)
    return _vf(ms)


def orient_outward(v, f):
    """Coherent per-component orientation by BFS over edge-adjacent faces (parity propagation), then
    each closed shell is flipped to positive volume. Returns (f, number of parity conflicts);
    conflicts > 0 means some component is non-orientable."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import breadth_first_order, connected_components
    F = f.shape[0]
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    fwd = e[:, 0] < e[:, 1]
    key = np.sort(e, 1)
    fid = np.tile(np.arange(F), 3)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    inv_s = inv[order]
    pair = (inv_s[1:] == inv_s[:-1]) & (cnt[inv_s[1:]] == 2)
    i0, i1 = order[:-1][pair], order[1:][pair]
    fa, fb = fid[i0], fid[i1]
    rel = (fwd[i0] == fwd[i1]).astype(np.int8)        # same direction -> one of the two must flip
    n_comp, lab = connected_components(sp.coo_matrix((np.ones(len(fa)), (fa, fb)), shape=(F, F)), directed=False)
    # Faces can share two edges after decimation: keep one relation per face pair (coo would sum them).
    _, first = np.unique(np.stack([np.minimum(fa, fb), np.maximum(fa, fb)], 1), axis=0, return_index=True)
    ua, ub, ur = fa[first], fb[first], rel[first]
    w = np.r_[ur, ur].astype(np.float64) + 1          # stored as 1/2 so that 0-relations stay explicit
    nbr = sp.coo_matrix((w, (np.r_[ua, ub], np.r_[ub, ua])), shape=(F, F)).tocsr()
    flip = np.zeros(F, np.int8)
    _, roots = np.unique(lab, return_index=True)
    for root in roots:
        seq, pred = breadth_first_order(nbr, int(root), directed=False, return_predecessors=True)
        seq = seq[1:]
        if seq.size == 0:
            continue
        r =(np.asarray(nbr[seq, pred[seq]]).ravel() - 1).astype(np.int8)
        for node, p, ri in zip(seq.tolist(), pred[seq].tolist(), r.tolist()):   # BFS order: parent is done
            flip[node] = flip[p] ^ ri
    conflicts = int(((flip[fa] ^ flip[fb]) != rel).sum())
    f = np.where(flip[:, None] == 1, f[:, ::-1], f)
    tri = v[f]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    neg = np.bincount(lab, weights=vol, minlength=n_comp) < 0
    f = np.where(neg[lab][:, None], f[:, ::-1], f)
    return np.ascontiguousarray(f), conflicts


def repair(v, f, max_rounds=6):
    """Make the mesh watertight, manifold and outward-oriented. Returns (v, f, rounds used).

    Edge collapse leaves a few edges shared by 4 faces where two sheets touch. pymeshlab's
    "Remove Faces" keeps faces of both sheets around such an edge, and the hole fill then joins
    them with opposite orientations (non-orientable patch). Instead, remove the whole 1-ring of
    every vertex on a non-2-face edge, so each hole is bounded by one consistently oriented
    sheet, then close the holes; repeat until watertight."""
    import pymeshlab as ml
    rounds = 0
    for rounds in range(max_rounds + 1):
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
        u, c = np.unique(e, axis=0, return_counts=True)
        bad = np.unique(u[c != 2])
        if bad.size == 0 or rounds == max_rounds:
            break
        f = f[~np.isin(f, bad).any(1)]
        ms = ml.MeshSet()
        ms.add_mesh(ml_mesh(v, f))
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_repair_non_manifold_vertices()
        try:
            ms.meshing_close_holes(maxholesize=1000, selfintersection=False)
        except Exception:
            pass
        v, f = _vf(ms)
    f, _ = orient_outward(v, f)
    return v, f, rounds


def write_obj(path, v, f):
    with open(path, "w") as fh:
        fh.write(f"# {v.shape[0]} vertices, {f.shape[0]} faces\n")
        np.savetxt(fh, v, fmt="v %.6f %.6f %.6f")
        np.savetxt(fh, f + 1, fmt="f %d %d %d")


def mc_mesh(sdf, res, min_area, level=0.0):
    v, f = extract_mesh(sdf, res=res, level=level)
    v, f = v.double().cpu().numpy(), f.cpu().numpy()
    return drop_small_components(v, f, min_area)


def fmt_d(d):
    return "/".join(f"{d.get(k, float('nan')):.4f}" for k in ("p50", "p90", "p99", "max"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--sdf", default=None, help="sdf_<it>.pt (default: latest in MODEL/proxy)")
    ap.add_argument("--render", default=None, help="render checkpoint dir/.pt (default: latest point_cloud iteration)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--res", type=int, default=192, help="MC resolution for the collision mesh")
    ap.add_argument("--ref_res", type=int, default=384, help="MC resolution of the full-detail reference proxy")
    ap.add_argument("--iso", type=float, default=0.015,
                    help="extract f = iso: an outward skin of this thickness (scene units) that keeps thin parts "
                         "(table tops) from breaking apart in decimation")
    ap.add_argument("--min_comp_area", type=float, default=0.05)
    ap.add_argument("--budget", type=int, default=50000)
    ap.add_argument("--eps", type=float, default=0.02, help="tolerance (scene units), reported against max and p99")
    ap.add_argument("--n_samples", type=int, default=300000)
    a = ap.parse_args()

    out = a.out or os.path.join(a.model, "collision")
    os.makedirs(out, exist_ok=True)
    sdf_path = a.sdf or latest(os.path.join(a.model, "proxy", "sdf_*.pt"))
    render_path = a.render or latest(os.path.join(a.model, "point_cloud", "iteration_*"))
    t_all = time.time()
    rep = dict(model=a.model, sdf=sdf_path, render=render_path, args=vars(a))

    sdf = SDFField.from_checkpoint(sdf_path, "cuda")
    ext = float((sdf.aabb_max - sdf.aabb_min).max())
    # Lower the MC resolution until topology-preserving decimation reaches the budget (fewer handles).
    for res in sorted({a.res, int(a.res / 1.5), a.res // 2, a.res // 3}, reverse=True):
        v, f, comp = mc_mesh(sdf, res, a.min_comp_area, a.iso)
        print(f"[collision] MC {res}: components {comp['components_before']} -> {comp['components_kept']}, "
              f"{f.shape[0]} faces kept")
        vd, fd = decimate(v, f, a.budget)
        if fd.shape[0] <= 1.05 * a.budget:
            break
        print(f"[collision] decimation stalls at {fd.shape[0]} faces (> budget {a.budget}); lowering MC res")
    vc, fc, rounds = repair(vd, fd)
    cs = mesh_stats(vc, fc)
    rep["mc"] = dict(res=res, faces_kept=int(f.shape[0]), mc_cell=ext / res, sdf_voxel=float(sdf.voxel_size.max()),
                     **comp)
    rep.update(decimated_faces=int(fd.shape[0]), repair_rounds=rounds, collision_stats=cs)
    print(f"[collision] decimated to {fd.shape[0]} faces, {rounds} repair round(s)")

    # Errors against the full-detail proxy and against the render mesh (in-box part).
    vref, fref, _ = mc_mesh(sdf, a.ref_res, a.min_comp_area)
    e_ref = two_sided(vc, fc, vref, fref, a.n_samples)
    vr, fr = load_mesh(render_path)
    with torch.no_grad():
        loc = sdf.to_local(torch.as_tensor(vr, dtype=torch.float32, device="cuda"))
        inside = ((loc >= sdf.aabb_min) & (loc <= sdf.aabb_max)).all(-1).cpu().numpy()
    fr_in = fr[inside[fr].all(1)]
    e_ren = two_sided(vr, fr_in, vc, fc, a.n_samples)
    rep["vs_reference"] = dict(ref_res=a.ref_res, ref_faces=int(fref.shape[0]), collision_to_ref=e_ref["a_to_b"],
                               ref_to_collision=e_ref["b_to_a"], hausdorff=e_ref["hausdorff"])
    rep["render"] = dict(faces=int(fr.shape[0]), faces_in_box=int(fr_in.shape[0]))
    rep["vs_render"] = dict(render_to_collision=e_ren["a_to_b"], collision_to_render=e_ren["b_to_a"],
                            hausdorff=e_ren["hausdorff"])
    rep["checks"] = dict(
        watertight=bool(cs.get("watertight")), manifold=cs.get("nonmanifold_edges", 1) == 0,
        winding_consistent=bool(cs.get("winding_consistent")), within_budget=cs["faces"] <= a.budget,
        eps_max_vs_reference=e_ref["hausdorff"] <= a.eps,
        eps_p99_vs_reference=max(e_ref["a_to_b"].get("p99", 1e9), e_ref["b_to_a"].get("p99", 1e9)) <= a.eps,
        eps_p99_render_to_collision=e_ren["a_to_b"].get("p99", 1e9) <= a.eps)

    write_obj(os.path.join(out, "collision.obj"), vc, fc)
    write_obj(os.path.join(out, "render.obj"), vr, fr)
    rep["seconds"] = round(time.time() - t_all, 1)
    with open(os.path.join(out, "collision_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)

    print(f"[collision] collision.obj: {cs['faces']} faces, watertight {cs.get('watertight')}, non-manifold edges "
          f"{cs.get('nonmanifold_edges')}, winding ok {cs.get('winding_consistent')}, components {cs.get('components')}")
    print(f"[collision] distances p50/p90/p99/max (eps {a.eps}):")
    print(f"    collision -> proxy@{a.ref_res}  {fmt_d(e_ref['a_to_b'])}")
    print(f"    proxy@{a.ref_res} -> collision  {fmt_d(e_ref['b_to_a'])}")
    print(f"    render -> collision      {fmt_d(e_ren['a_to_b'])}")
    print(f"    collision -> render      {fmt_d(e_ren['b_to_a'])}")
    print(f"[collision] checks {rep['checks']}")
    print(f"[collision] wrote {out} in {rep['seconds']} s")


if __name__ == "__main__":
    main()
