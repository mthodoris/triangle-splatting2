#
# Close small holes of a mesh: every boundary loop that is a clean simple cycle (each vertex on exactly two
# boundary edges, all edges consistently oriented) with at most --max_loop_edges edges is triangulated.
# Loops of 3 edges become one triangle, longer loops get a fan around a new vertex at their centroid
# (colour = mean of the loop's vertex colours). Boundary components that branch at pinch points, open chains
# and everything larger than the limit (the scene's outer border) are left alone.
#
# Usage:
#   python scripts/fill_mesh_holes.py --mesh init.ply --out init_filled.ply [--max_loop_edges 64]
#

import collections
from argparse import ArgumentParser

import numpy as np
import trimesh


def boundary_loops(faces, max_loop_edges):
    """Ordered vertex loops (in the faces' own edge direction) of clean boundary cycles up to max_loop_edges."""
    he = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    und = np.sort(he, axis=1)
    key = und[:, 0].astype(np.int64) * (int(faces.max()) + 1) + und[:, 1]
    _, inv, count = np.unique(key, return_inverse=True, return_counts=True)
    boundary = he[count[inv] == 1]  # directed as in the faces
    out = collections.defaultdict(list)
    inc = collections.Counter()
    for a, b in boundary:
        out[int(a)].append(int(b))
        inc[int(b)] += 1
    loops, skipped = [], collections.Counter()
    seen = set()
    for start in list(out):
        if start in seen:
            continue
        loop, cur, ok = [start], start, True
        while True:
            nxt = out[cur]
            if len(nxt) != 1 or inc[cur] != 1:
                ok = False  # pinch point or open chain
                break
            cur = nxt[0]
            if cur == start:
                break
            if cur in seen or len(loop) >= max_loop_edges:
                ok = False
                break
            loop.append(cur)
        seen.update(loop)
        if not ok:
            skipped["tangled or larger than the limit"] += 1
        elif len(loop) < 3:
            skipped["degenerate"] += 1
        else:
            loops.append(loop)
    return loops, skipped, len(boundary)


def main():
    parser = ArgumentParser(description="Close small clean boundary loops of a mesh")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_loop_edges", type=int, default=64)
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if getattr(mesh.visual, "vertex_colors", None) is not None:
        C = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64)
    else:
        C = np.full((len(V), 3), 128.0)

    loops, skipped, n_boundary = boundary_loops(F, args.max_loop_edges)
    new_v, new_c, new_f = [], [], []
    for loop in loops:
        loop = np.asarray(loop)
        if len(loop) == 3:
            new_f.append([loop[1], loop[0], loop[2]])  # opposite direction of the boundary edges
            continue
        c = len(V) + len(new_v)
        new_v.append(V[loop].mean(0))
        new_c.append(C[loop].mean(0))
        for i in range(len(loop)):
            new_f.append([loop[(i + 1) % len(loop)], loop[i], c])
    V2 = np.concatenate([V, np.array(new_v).reshape(-1, 3)])
    C2 = np.concatenate([C, np.array(new_c).reshape(-1, 3)]).clip(0, 255).astype(np.uint8)
    F2 = np.concatenate([F, np.array(new_f, dtype=np.int64).reshape(-1, 3)])
    trimesh.Trimesh(V2, F2, vertex_colors=C2, process=False).export(args.out)
    closed = sum(len(loop) for loop in loops)
    print("boundary edges {} -> {} | closed {} loops (<= {} edges): +{} vertices, +{} faces | the rest is tangled or larger than the limit".format(
        n_boundary, n_boundary - closed, len(loops), args.max_loop_edges, len(new_v), len(new_f)))


if __name__ == "__main__":
    main()
