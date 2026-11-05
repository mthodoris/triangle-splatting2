"""Midpoint subdivision of a mesh (every triangle -> 4, new vertices at edge midpoints, colors averaged).
The surface does not change, only the vertex count, so a run with --freeze_mesh on the result measures
what more color capacity on the same geometry buys. Vertices no face uses are dropped.

    python scripts/subdivide_mesh.py in.ply out.ply --levels 1
"""
import argparse

import numpy as np
import trimesh


def subdivide(vertices, faces, colors):
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    mid = len(vertices) + inverse.reshape(-1, 3)  # midpoint index of edges (v0,v1), (v1,v2), (v2,v0)
    vertices = np.concatenate([vertices, vertices[unique].mean(1)])
    colors = np.concatenate([colors, colors[unique].mean(1)])
    v0, v1, v2 = faces.T
    m0, m1, m2 = mid.T
    faces = np.concatenate([np.stack(f, 1) for f in ((v0, m0, m2), (m0, v1, m1), (m2, m1, v2), (m0, m1, m2))])
    return vertices, faces, colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--levels", type=int, default=1)
    args = parser.parse_args()

    mesh = trimesh.load(args.input, process=False, force="mesh")
    faces = np.asarray(mesh.faces, dtype=np.int64)
    used = np.unique(faces)
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)[used]
    if mesh.visual.kind == "vertex":
        colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float64)[used]
    else:
        colors = np.full((len(used), 4), 128.0)
    faces = remap[faces]
    print("{}: {} vertices, {} faces".format(args.input, len(vertices), len(faces)))

    for _ in range(args.levels):
        vertices, faces, colors = subdivide(vertices, faces, colors)

    trimesh.Trimesh(vertices, faces, vertex_colors=np.round(colors).astype(np.uint8), process=False).export(args.output)
    print("{}: {} vertices, {} faces ({} level(s))".format(args.output, len(vertices), len(faces), args.levels))


if __name__ == "__main__":
    main()
