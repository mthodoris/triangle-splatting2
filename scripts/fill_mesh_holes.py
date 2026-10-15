#
# Close small holes of a mesh. Boundary edges (used by one face) are grouped into a graph; in every connected
# component the shortest independent cycles (networkx minimum cycle basis) are hole candidates. This also works
# where holes touch non-manifold vertices or each other, which a plain boundary walk cannot follow.
# A cycle is filled when it has at most --max_loop_edges edges and its bounding box is at most --max_extent mean
# edges across (a hole is small; the scene's outer border is not). Loops of 3 edges become one triangle, longer
# loops get a fan around a new vertex at their centroid (colour = mean of the loop's vertex colours). Each
# boundary edge is used by at most one fill, shortest cycles first. Open chains (slits) have no cycle and are
# left alone. The fill is wound opposite to the majority of the loop's existing edge directions.
#
# Usage:
#   python scripts/fill_mesh_holes.py --mesh init.ply --out init_filled.ply \
#       [--max_loop_edges 64] [--max_extent 8] [--debug_ply filled_faces_red.ply]
#

from argparse import ArgumentParser

import networkx as nx
import numpy as np
import trimesh


def find_holes(V, F, max_loop_edges, max_extent):
    """Return (cycles, stats). cycles: lists of vertex ids, each a simple boundary cycle to fill."""
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    und = np.sort(he, axis=1)
    key = und[:, 0] * len(V) + und[:, 1]
    _, inv, count = np.unique(key, return_inverse=True, return_counts=True)
    boundary_directed = he[count[inv] == 1]
    directed = {(int(a), int(b)) for a, b in boundary_directed}
    edge_len = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1).mean()  # mean edge length (first edge of each face)
    G = nx.Graph()
    for a, b in boundary_directed:
        G.add_edge(int(a), int(b), weight=float(np.linalg.norm(V[a] - V[b])))

    candidates = []
    stats = {"components": 0, "with cycles": 0, "too big": 0}
    for nodes in nx.connected_components(G):
        H = G.subgraph(nodes)
        stats["components"] += 1
        if H.number_of_edges() < H.number_of_nodes():
            continue  # open chain (a tree): no cycle
        stats["with cycles"] += 1
        for cycle in nx.minimum_cycle_basis(H, weight="weight"):
            if len(cycle) > max_loop_edges:
                stats["too big"] += 1
                continue
            P = V[list(cycle)]
            if np.linalg.norm(P.max(0) - P.min(0)) > max_extent * edge_len:
                stats["too big"] += 1
                continue
            candidates.append(order_cycle(H, cycle))
    candidates.sort(key=len)
    used, cycles = set(), []
    for cyc in candidates:
        edges = {frozenset((cyc[i], cyc[(i + 1) % len(cyc)])) for i in range(len(cyc))}
        if used & edges:
            continue
        used |= edges
        cycles.append(cyc)
    return cycles, directed, stats, int(len(boundary_directed))


def order_cycle(H, cycle):
    """Vertices of a simple cycle (given as a node list from the cycle basis) in walking order."""
    nodes = set(cycle)
    order = [cycle[0]]
    prev, cur = None, cycle[0]
    while True:
        nxt = [n for n in H.neighbors(cur) if n in nodes and n != prev]
        if not nxt:
            break
        nxt = nxt[0]
        if nxt == order[0]:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(nodes):
            break
    return order


def main():
    parser = ArgumentParser(description="Close small holes of a mesh (shortest boundary cycles)")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_loop_edges", type=int, default=64)
    parser.add_argument("--max_extent", type=float, default=8.0, help="largest bounding-box diagonal of a hole, in mean edges")
    parser.add_argument("--debug_ply", default=None, help="also write the filled mesh with the new faces in red")
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if mesh.visual.kind == "vertex":
        C = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64)
    else:
        C = np.full((len(V), 3), 128.0)

    cycles, directed, stats, n_boundary = find_holes(V, F, args.max_loop_edges, args.max_extent)
    new_v, new_c, new_f = [], [], []
    for cyc in cycles:
        loop = np.asarray(cyc)
        n = len(loop)
        # existing faces use a->b along the loop for some edges: wind the fill the other way round
        forward = sum((int(loop[i]), int(loop[(i + 1) % n])) in directed for i in range(n))
        backward = sum((int(loop[(i + 1) % n]), int(loop[i])) in directed for i in range(n))
        if forward < backward:
            loop = loop[::-1]
        if n == 3:
            new_f.append([loop[1], loop[0], loop[2]])
            continue
        c = len(V) + len(new_v)
        new_v.append(V[loop].mean(0))
        new_c.append(C[loop].mean(0))
        for i in range(n):
            new_f.append([loop[(i + 1) % n], loop[i], c])
    V2 = np.concatenate([V, np.array(new_v).reshape(-1, 3)])
    C2 = np.concatenate([C, np.array(new_c).reshape(-1, 3)]).clip(0, 255).astype(np.uint8)
    F2 = np.concatenate([F, np.array(new_f, dtype=np.int64).reshape(-1, 3)])
    trimesh.Trimesh(V2, F2, vertex_colors=C2, process=False).export(args.out)
    if args.debug_ply:
        face_colors = np.tile(np.array([170, 170, 170, 255], dtype=np.uint8), (len(F2), 1))
        face_colors[len(F):] = [255, 0, 0, 255]
        trimesh.Trimesh(V2, F2, face_colors=face_colors, process=False).export(args.debug_ply)
    closed = sum(len(c) for c in cycles)
    print("boundary edges {} -> about {} | filled {} holes (<= {} edges, <= {} mean edges across): +{} vertices, +{} faces".format(
        n_boundary, n_boundary - closed, len(cycles), args.max_loop_edges, args.max_extent, len(new_v), len(new_f)))
    print("components: {components}, with cycles: {with cycles}, cycles skipped as too big: {too big}".format(**stats))


if __name__ == "__main__":
    main()
