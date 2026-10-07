#
# Compare trained meshes (<model_path>/mesh/iteration_N/mesh.ply) against the
# mesh they were initialized from (--mesh_path).
#
# For each trained mesh it reports:
#   - point-to-surface distance, both ways, from points sampled on each
#     surface: accuracy (trained -> init), completeness (init -> trained) and
#     F-score, in units of the init mesh's mean edge length
#   - normal agreement at those nearest points (|cos|, winding-independent)
#   - roughness: dihedral angle between adjacent faces over manifold edges
#   - topology: vertices, faces, manifold / boundary / non-manifold edges
#   - tangling: faces tilted more than --tilt degrees from the init surface normal
#     nearby, self-intersecting faces (sampled; pairs sharing a vertex excluded)
#     and the triangle shape-quality distribution
#   - bad triangles: zero-area and sliver faces, oversized faces (longest edge
#     well above the init mean edge) and the oversized faces that sit away from
#     the init surface; with the run's checkpoint next to the mesh, also how
#     opaque those faces are (i.e. whether they actually render)
#
# Usage:
#   python scripts/compare_to_init_mesh.py --init init_mesh.ply \
#       --meshes run_a/mesh/iteration_30000/mesh.ply run_b/mesh/iteration_30000/mesh.ply \
#       [--names baseline reg] [--save_dist_ply out_dir]
#
# Checkpoints are found automatically at <run>/point_cloud/iteration_N/ for a
# mesh at <run>/mesh/iteration_N/mesh.ply (needs torch; skipped otherwise).
#

import os
from argparse import ArgumentParser

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def sample(mesh, n, seed):
    points, face_ids = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return points, mesh.face_normals[face_ids]


class Surface:
    """Exact point-to-triangle distance; candidates are the k faces with the nearest centroids."""

    def __init__(self, mesh, k=16):
        self.triangles = mesh.triangles
        self.normals = mesh.face_normals
        self.tree = cKDTree(mesh.triangles_center)
        self.k = k

    def query(self, points, chunk=200_000):
        dist = np.empty(len(points))
        face = np.empty(len(points), dtype=np.int64)
        for s in range(0, len(points), chunk):
            p = points[s:s + chunk]
            _, cand = self.tree.query(p, k=self.k, workers=-1)  # [n,k]
            with np.errstate(divide="ignore", invalid="ignore"):
                closest = trimesh.triangles.closest_point(self.triangles[cand.ravel()], np.repeat(p, self.k, axis=0))
            d = np.linalg.norm(closest - np.repeat(p, self.k, axis=0), axis=1).reshape(-1, self.k)
            d[np.isnan(d)] = np.inf  # degenerate (zero-area) candidate faces
            best = d.argmin(1)
            dist[s:s + chunk] = d[np.arange(len(p)), best]
            face[s:s + chunk] = cand[np.arange(len(p)), best]
        return dist, face


def topology(mesh):
    edges = np.sort(mesh.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    angles = np.degrees(mesh.face_adjacency_angles)  # trimesh: only edges shared by exactly two faces
    return {
        "vertices": len(np.unique(mesh.faces)),
        "faces": len(mesh.faces),
        "manifold_%": 100.0 * (counts == 2).mean(),
        "boundary_%": 100.0 * (counts == 1).mean(),
        "nonmanifold_%": 100.0 * (counts > 2).mean(),
        "dihedral_mean_deg": angles.mean() if len(angles) else float("nan"),
        "dihedral_median_deg": np.median(angles) if len(angles) else float("nan"),
        "dihedral_>30deg_%": 100.0 * (angles > 30).mean() if len(angles) else float("nan"),
    }


def triangle_stats(mesh, unit, init_surface, opacity, sliver_q, big_edge, far):
    """Degenerate and oversized faces. Lengths in init mean edges; *_area_% is the share of total surface area."""
    tri = mesh.triangles
    e = np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2)  # [F,3] edge lengths
    area = mesh.area_faces
    with np.errstate(divide="ignore", invalid="ignore"):
        quality = np.nan_to_num(4.0 * np.sqrt(3.0) * area / (e ** 2).sum(1))  # 1 = equilateral, 0 = degenerate
    zero = area < 1e-6 * unit ** 2
    sliver = quality < sliver_q
    big = e.max(1) > big_edge * unit
    big_far = np.zeros(len(area), dtype=bool)
    if big.any():
        d, _ = init_surface.query(mesh.triangles_center[big])
        big_far[np.flatnonzero(big)[d > far * unit]] = True
    total = max(area.sum(), 1e-30)
    stats = {
        "zero_area_%": 100.0 * zero.mean(),
        "sliver_%": 100.0 * sliver.mean(),
        "big_%": 100.0 * big.mean(),
        "big_area_%": 100.0 * area[big].sum() / total,
        "big_far_%": 100.0 * big_far.mean(),
        "big_far_area_%": 100.0 * area[big_far].sum() / total,
    }
    if opacity is not None:
        opaque = opacity >= 0.5
        stats["opaque_%"] = 100.0 * opaque.mean()
        stats["sliver_opaque_%"] = 100.0 * opaque[sliver].mean() if sliver.any() else float("nan")
        stats["big_far_opaque_%"] = 100.0 * opaque[big_far].mean() if big_far.any() else float("nan")
    return stats


def segments_hit_triangles(p, q, a, b, c, eps=1e-9):
    """[N] bool: segment p->q crosses triangle abc (Moller-Trumbore, open segment)."""
    e1, e2, d = b - a, c - a, q - p
    h = np.cross(d, e2)
    det = (e1 * h).sum(1)
    ok = np.abs(det) > eps
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p - a
    u = (s * h).sum(1) * inv
    qv = np.cross(s, e1)
    v = (d * qv).sum(1) * inv
    t = (e2 * qv).sum(1) * inv
    return ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


def tangling_stats(mesh, init_surface, tilt_deg, n_sample, k=16, seed=0):
    """Tilted faces vs the init surface, sampled self-intersections, and triangle shape quality."""
    tri = mesh.triangles
    area = mesh.area_faces
    total = max(area.sum(), 1e-30)
    _, nearest = init_surface.query(mesh.triangles_center)
    cos = np.abs((mesh.face_normals * init_surface.normals[nearest]).sum(1))
    tilted = cos < np.cos(np.radians(tilt_deg))
    e = np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        quality = np.nan_to_num(4.0 * np.sqrt(3.0) * area / (e ** 2).sum(1))

    # self-intersections: sampled faces against their k nearest faces (by centroid) that share no vertex
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(area > 0)
    sample = rng.choice(valid, size=min(n_sample, len(valid)), replace=False)
    tree = cKDTree(mesh.triangles_center[valid])
    _, nb = tree.query(mesh.triangles_center[sample], k=k + 1, workers=-1)
    nb = valid[nb]
    F = mesh.faces
    hit = np.zeros(len(sample), dtype=bool)
    for j in range(1, k + 1):
        A, B = sample, nb[:, j]
        share = (F[A][:, :, None] == F[B][:, None, :]).any((1, 2))
        T1, T2 = tri[A], tri[B]
        x = np.zeros(len(A), dtype=bool)
        for i0, i1 in ((0, 1), (1, 2), (2, 0)):
            x |= segments_hit_triangles(T1[:, i0], T1[:, i1], T2[:, 0], T2[:, 1], T2[:, 2])
            x |= segments_hit_triangles(T2[:, i0], T2[:, i1], T1[:, 0], T1[:, 1], T1[:, 2])
        hit |= x & ~share
    return {
        "tilted_%": 100.0 * tilted.mean(),
        "tilted_area_%": 100.0 * area[tilted].sum() / total,
        "self_intersect_%": 100.0 * hit.mean(),
        "quality_median": float(np.median(quality)),
        "quality_<0.3_%": 100.0 * (quality < 0.3).mean(),
    }


def triangle_opacity(mesh_path, n_faces):
    """Mean vertex opacity per face from the run's checkpoint, or None if it cannot be found or does not match."""
    parts = os.path.normpath(os.path.abspath(mesh_path)).split(os.sep)
    if len(parts) < 4 or parts[-3] != "mesh":
        return None
    ckpt = os.path.join(os.sep.join(parts[:-3]), "point_cloud", parts[-2], "point_cloud_state_dict.pt")
    if not os.path.exists(ckpt):
        return None
    try:
        import torch
    except ImportError:
        return None
    state = torch.load(ckpt, map_location="cpu")
    faces = state["_triangle_indices"].long()
    floor = state.get("opacity_floor", 0.9999)  # same default as TriangleModel.load_parameters
    if "vertex_is_mesh" in state:  # --free_triangles: mesh.ply holds the mesh group only
        is_mesh = state["vertex_is_mesh"].bool().cpu()
        faces = faces[is_mesh[faces[:, 0]]]
        floor = torch.where(is_mesh, torch.tensor(floor), torch.tensor(state["free_opacity_floor"]))
    if len(faces) != n_faces:
        print("  {}: checkpoint has {} faces, mesh {}; skipping opacity".format(ckpt, len(faces), n_faces))
        return None
    vertex = floor + (1.0 - floor) * torch.sigmoid(state["vertex_weight"].detach().float().reshape(-1))
    return vertex[faces].mean(1).numpy()


def compare(init_surface, init_pts, init_nrm, mesh, n, unit, tau):
    pts, nrm = sample(mesh, n, seed=1)
    surface = Surface(mesh)
    acc, face_acc = init_surface.query(pts)  # trained -> init
    comp, face_comp = surface.query(init_pts)  # init -> trained
    cos_acc = np.abs((nrm * init_surface.normals[face_acc]).sum(1))
    cos_comp = np.abs((init_nrm * surface.normals[face_comp]).sum(1))
    precision = (acc < tau * unit).mean()
    recall = (comp < tau * unit).mean()
    return {
        "acc_mean": acc.mean() / unit,
        "acc_median": np.median(acc) / unit,
        "acc_p90": np.percentile(acc, 90) / unit,
        "comp_mean": comp.mean() / unit,
        "comp_median": np.median(comp) / unit,
        "comp_p90": np.percentile(comp, 90) / unit,
        "chamfer": (acc.mean() + comp.mean()) / 2.0 / unit,
        "precision_%": 100.0 * precision,
        "recall_%": 100.0 * recall,
        "fscore_%": 100.0 * 2 * precision * recall / max(precision + recall, 1e-12),
        "normal_cos": (cos_acc.mean() + cos_comp.mean()) / 2.0,
    }


def save_distance_ply(init_surface, mesh, unit, path):
    """Trained mesh with vertices colored by distance to the init surface (blue 0 -> red >= 5 edges)."""
    d, _ = init_surface.query(mesh.vertices)
    t = np.clip(d / (5.0 * unit), 0.0, 1.0)[:, None]
    colors = ((1 - t) * np.array([40, 90, 255]) + t * np.array([255, 40, 40])).astype(np.uint8)
    trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=colors, process=False).export(path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Compare trained meshes against the init mesh")
    parser.add_argument("--init", required=True, help="mesh passed to train.py --mesh_path")
    parser.add_argument("--meshes", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--tau", type=float, default=1.0, help="F-score threshold, in init mean edge lengths")
    parser.add_argument("--sliver_q", type=float, default=0.05, help="sliver: shape quality below this (1 = equilateral)")
    parser.add_argument("--big_edge", type=float, default=5.0, help="oversized: longest edge above this many init mean edges")
    parser.add_argument("--far", type=float, default=2.0, help="oversized face counts as away from the init surface beyond this many edges")
    parser.add_argument("--tilt", type=float, default=60.0, help="tilted: face normal more than this many degrees from the init surface normal")
    parser.add_argument("--intersect_samples", type=int, default=200_000, help="faces sampled for the self-intersection check")
    parser.add_argument("--save_dist_ply", default=None, help="directory for distance-colored copies of the trained meshes")
    args = parser.parse_args()

    names = args.names or [os.path.normpath(m).split(os.sep)[-4] if m.count(os.sep) >= 3 else m for m in args.meshes]
    assert len(names) == len(args.meshes), "--names must match --meshes"

    init = trimesh.load(args.init, process=False)
    unit = init.edges_unique_length.mean()
    init_pts, init_nrm = sample(init, args.samples, seed=0)
    init_surface = Surface(init)
    print("init: {}  (mean edge length {:.5f}; distances below are in these units)".format(args.init, unit))
    print("tangling: tilted > {} deg from the init surface normal; self-intersections from {} sampled faces; boundary_% = open edges (cracks, T-junctions)".format(args.tilt, args.intersect_samples))
    print("bad triangles: sliver quality < {}, oversized longest edge > {} edges, far > {} edges from init; opaque = mean opacity >= 0.5".format(args.sliver_q, args.big_edge, args.far))
    bad = lambda m, op: {**tangling_stats(m, init_surface, args.tilt, args.intersect_samples),
                         **triangle_stats(m, unit, init_surface, op, args.sliver_q, args.big_edge, args.far)}

    rows = {"init": {**topology(init), **bad(init, None)}}
    for name, path in zip(names, args.meshes):
        mesh = trimesh.load(path, process=False)
        opacity = triangle_opacity(path, len(mesh.faces))
        rows[name] = {**topology(mesh), **compare(init_surface, init_pts, init_nrm, mesh, args.samples, unit, args.tau), **bad(mesh, opacity)}
        if args.save_dist_ply:
            os.makedirs(args.save_dist_ply, exist_ok=True)
            save_distance_ply(init_surface, mesh, unit, os.path.join(args.save_dist_ply, name + "_dist_to_init.ply"))

    keys = list(dict.fromkeys(k for r in rows.values() for k in r))
    width = max(12, max(len(n) for n in rows) + 2)
    print("\n{:<22}".format("metric") + "".join("{:>{w}}".format(n, w=width) for n in rows))
    for k in keys:
        cells = []
        for r in rows.values():
            v = r.get(k)
            cells.append("{:>{w}}".format("-" if v is None else (str(v) if isinstance(v, (int, np.integer)) else "{:.4f}".format(v)), w=width))
        print("{:<22}".format(k) + "".join(cells))
