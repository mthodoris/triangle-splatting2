#!/usr/bin/env python3
"""Topology statistics of a triangle mesh: the "is it usable as a collider?" check.

Works on any mesh trimesh can read (.ply/.obj/...) and on a Triangle Splatting+
checkpoint (point_cloud_state_dict.pt, or the iteration folder containing it).

Reported (all counts are over faces/edges after merging nothing: the mesh is
measured exactly as stored, so a triangle soup stays a soup):
  faces, vertices, unreferenced vertices, degenerate faces (area <= eps),
  duplicate faces (same vertex set), edges, boundary edges (1 face),
  non-manifold edges (>2 faces), connected components (faces sharing an edge),
  area of the largest component (absolute and as a fraction), watertight
  (every edge shared by exactly 2 faces), winding consistent, volume (only if
  watertight and winding consistent).

Usage:
  python physproxy/eval/mesh_stats.py MESH [MESH ...] [--json out.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def load_mesh(path):
    """Return (vertices [V,3] float64, faces [F,3] int64)."""
    if os.path.isdir(path):
        path = os.path.join(path, "point_cloud_state_dict.pt")
    if path.endswith(".pt"):
        import torch
        sd = torch.load(path, map_location="cpu")
        v = sd["triangles_points"].detach().double().numpy()
        f = sd["_triangle_indices"].detach().long().numpy()
        return v, f
    import trimesh
    m = trimesh.load(path, process=False, force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def mesh_stats(v, f, area_eps=1e-12):
    F = f.shape[0]
    V = v.shape[0]
    out = {"faces": int(F), "vertices": int(V)}
    if F == 0:
        out.update(watertight=False)
        return out

    out["unreferenced_vertices"] = int(V - np.unique(f).size)

    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    out["total_area"] = float(areas.sum())
    out["degenerate_faces"] = int((areas <= area_eps).sum())
    out["duplicate_faces"] = int(F - np.unique(np.sort(f, axis=1), axis=0).shape[0])

    # Undirected edges: each face contributes 3 (face id kept for adjacency).
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e_dir = e.copy()
    e.sort(axis=1)
    face_of_edge = np.tile(np.arange(F), 3)
    uniq, inv, counts = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out["edges"] = int(uniq.shape[0])
    out["boundary_edges"] = int((counts == 1).sum())
    out["nonmanifold_edges"] = int((counts > 2).sum())
    out["boundary_edge_frac"] = out["boundary_edges"] / out["edges"]
    out["nonmanifold_edge_frac"] = out["nonmanifold_edges"] / out["edges"]
    out["watertight"] = bool(out["boundary_edges"] == 0 and out["nonmanifold_edges"] == 0)

    # Face adjacency through shared edges: connect every face to the first
    # face of each of its edges (a star per edge, enough for components).
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    first_idx = np.searchsorted(inv_sorted, np.arange(uniq.shape[0]))
    rep_face = face_of_edge[order][first_idx]          # one face per edge
    a = face_of_edge
    b = rep_face[inv]
    adj = sp.coo_matrix((np.ones_like(a), (a, b)), shape=(F, F))
    n_comp, labels = connected_components(adj, directed=False)
    comp_area = np.bincount(labels, weights=areas, minlength=n_comp)
    comp_faces = np.bincount(labels, minlength=n_comp)
    out["components"] = int(n_comp)
    out["largest_component_faces"] = int(comp_faces.max())
    out["largest_component_area"] = float(comp_area.max())
    out["largest_component_area_frac"] = float(comp_area.max() / max(areas.sum(), 1e-30))
    out["single_face_components"] = int((comp_faces == 1).sum())

    # Winding consistency: on a manifold edge the two faces must traverse it
    # in opposite directions.
    manifold = counts[inv] == 2
    fwd = e_dir[:, 0] < e_dir[:, 1]
    # For each manifold edge, sum of +1/-1 directions must be 0.
    sgn = np.where(fwd, 1, -1)
    dir_sum = np.bincount(inv[manifold], weights=sgn[manifold], minlength=uniq.shape[0])
    inconsistent = int((np.abs(dir_sum[counts == 2]) > 0).sum())
    out["winding_inconsistent_edges"] = inconsistent
    out["winding_consistent"] = bool(inconsistent == 0)

    if out["watertight"] and out["winding_consistent"]:
        # Divergence theorem.
        out["volume"] = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    return out


def fmt(stats):
    keys = ["faces", "vertices", "unreferenced_vertices", "degenerate_faces", "duplicate_faces",
            "edges", "boundary_edges", "boundary_edge_frac", "nonmanifold_edges", "nonmanifold_edge_frac",
            "components", "single_face_components", "largest_component_faces",
            "largest_component_area", "largest_component_area_frac", "total_area",
            "watertight", "winding_consistent", "winding_inconsistent_edges", "volume"]
    lines = []
    for k in keys:
        if k in stats:
            val = stats[k]
            if isinstance(val, float):
                val = f"{val:.4g}"
            lines.append(f"  {k:30s} {val}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("meshes", nargs="+")
    p.add_argument("--json", default=None, help="write all results to this JSON file")
    args = p.parse_args()
    results = {}
    for path in args.meshes:
        v, f = load_mesh(path)
        s = mesh_stats(v, f)
        results[path] = s
        print(path)
        print(fmt(s))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
