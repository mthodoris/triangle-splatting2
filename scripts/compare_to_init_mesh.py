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
#
# Usage:
#   python scripts/compare_to_init_mesh.py --init init_mesh.ply \
#       --meshes run_a/mesh/iteration_30000/mesh.ply run_b/mesh/iteration_30000/mesh.ply \
#       [--names baseline reg] [--save_dist_ply out_dir]
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
    parser.add_argument("--save_dist_ply", default=None, help="directory for distance-colored copies of the trained meshes")
    args = parser.parse_args()

    names = args.names or [os.path.normpath(m).split(os.sep)[-4] if m.count(os.sep) >= 3 else m for m in args.meshes]
    assert len(names) == len(args.meshes), "--names must match --meshes"

    init = trimesh.load(args.init, process=False)
    unit = init.edges_unique_length.mean()
    init_pts, init_nrm = sample(init, args.samples, seed=0)
    init_surface = Surface(init)
    print("init: {}  (mean edge length {:.5f}; distances below are in these units)".format(args.init, unit))

    rows = {"init": topology(init)}
    for name, path in zip(names, args.meshes):
        mesh = trimesh.load(path, process=False)
        rows[name] = {**topology(mesh), **compare(init_surface, init_pts, init_nrm, mesh, args.samples, unit, args.tau)}
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
