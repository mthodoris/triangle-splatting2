#
# Close small holes of a mesh. Boundary edges (used by one face) are grouped into a graph; in every connected
# component the shortest independent cycles (networkx minimum cycle basis) are hole candidates. This also works
# where holes touch non-manifold vertices or each other, which a plain boundary walk cannot follow.
# A cycle is filled when it has at most --max_loop_edges edges and its bounding box is at most --max_extent mean
# edges across (a hole is small; the scene's outer border is not). Loops of 3 edges become one triangle, longer
# loops get a fan around a new vertex at their centroid (colour = mean of the loop's vertex colours). Each
# boundary edge is used by at most one fill, shortest cycles first. Open chains (slits) have no cycle and are
# left alone unless --through_nonmanifold lets rims run along edges shared by 3+ faces. The fill is wound opposite to the majority of the loop's existing edge directions.
#
# Usage:
#   python scripts/fill_mesh_holes.py --mesh init.ply --out init_filled.ply \
#       [--max_loop_edges 64] [--max_extent 8] [--debug_ply filled_faces_red.ply]
#

from argparse import ArgumentParser

import networkx as nx
import numpy as np
import trimesh


def find_holes(V, F, max_loop_edges, max_extent, through_nonmanifold=False, min_boundary_share=0.5):
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
        G.add_edge(int(a), int(b), weight=float(np.linalg.norm(V[a] - V[b])), nm=False)
    if through_nonmanifold:
        # Holes are often rimmed partly by edges shared by 3+ faces: every dead end of an open boundary chain
        # sits on one. Let the cycle search walk along those edges where they touch the boundary.
        on_boundary = set(G.nodes)
        for a, b in np.unique(und[count[inv] > 2], axis=0):
            if (int(a) in on_boundary or int(b) in on_boundary) and not G.has_edge(int(a), int(b)):
                G.add_edge(int(a), int(b), weight=float(np.linalg.norm(V[a] - V[b])), nm=True)

    candidates = []
    stats = {"components": 0, "with cycles": 0, "too big": 0, "mostly non-manifold": 0}
    if through_nonmanifold:
        # Fast local search (the minimum cycle basis is too slow on the merged components): for every real boundary
        # edge (a, b) the shortest path from a to b that avoids that edge closes the hole's rim. Non-manifold edges
        # cost twice their length so rims prefer real boundary edges.
        cutoff = 2.5 * max_extent * edge_len
        passes = []  # pass 1: real boundary edges only; pass 2 (leftovers): rims may also run along non-manifold edges
        for allow_nm in (False, True):
            found = []
            for a, b in boundary_directed:
                a, b = int(a), int(b)
                if not G.has_edge(a, b):
                    continue

                def weight(u, v, d, a=a, b=b, allow_nm=allow_nm):
                    if (u == a and v == b) or (u == b and v == a):
                        return None
                    if d["nm"] and not allow_nm:
                        return None
                    return d["weight"] * (2.0 if d["nm"] else 1.0)

                try:
                    _, path = nx.single_source_dijkstra(G, a, target=b, cutoff=cutoff, weight=weight)
                except nx.NetworkXNoPath:
                    continue
                if len(path) < 3 or len(path) > max_loop_edges:
                    stats["too big"] += 1
                    continue
                P = V[path]
                if np.linalg.norm(P.max(0) - P.min(0)) > max_extent * edge_len:
                    stats["too big"] += 1
                    continue
                ring = [(path[i], path[(i + 1) % len(path)]) for i in range(len(path))]
                n_real = sum(not G[u][v]["nm"] for u, v in ring)
                if n_real == 0 or n_real < min_boundary_share * len(ring):
                    stats["mostly non-manifold"] += 1
                    continue
                found.append(path)
            passes.append(sorted(found, key=len))
        candidates = passes[0] + passes[1]  # pure boundary rims win over rims through non-manifold edges
        stats["components"] = nx.number_connected_components(G)
    for nodes in ([] if through_nonmanifold else nx.connected_components(G)):
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
            ordered = order_cycle(H, cycle)
            ring = [(ordered[i], ordered[(i + 1) % len(ordered)]) for i in range(len(ordered))]
            if not all(H.has_edge(a, b) for a, b in ring):
                continue
            n_boundary_edges = sum(not H[a][b]["nm"] for a, b in ring)
            if n_boundary_edges == 0 or n_boundary_edges < min_boundary_share * len(ring):
                stats["mostly non-manifold"] += 1
                continue
            candidates.append(ordered)
    if not through_nonmanifold:
        candidates.sort(key=len)
    used, cycles = set(), []
    for cyc in candidates:
        edges = {frozenset((cyc[i], cyc[(i + 1) % len(cyc)])) for i in range(len(cyc))}
        if used & edges:
            continue
        used |= edges
        cycles.append(cyc)
    return cycles, directed, stats, int(len(boundary_directed))


def drop_nonmanifold_faces(V, F, max_rounds=20):
    """Make every edge belong to at most two faces: at an edge shared by 3+ faces keep the two whose normals are most
    parallel (sign-free, i.e. the pair that continues each other best) and drop the rest. Repeats because dropping
    a face lowers the count of its other two edges. Returns the kept faces."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = np.linalg.norm(n, axis=1)
    n = n / np.maximum(area, 1e-20)[:, None]
    alive = np.ones(len(F), dtype=bool)
    for _ in range(max_rounds):
        idx = np.nonzero(alive)[0]
        Fa = F[idx]
        ends = np.concatenate([Fa[:, [0, 1]], Fa[:, [1, 2]], Fa[:, [2, 0]]])
        key = np.sort(ends, axis=1)
        key = key[:, 0].astype(np.int64) * len(V) + key[:, 1]
        face_of = np.tile(idx, 3)
        order = np.argsort(key, kind="stable")
        key, face_of = key[order], face_of[order]
        starts = np.nonzero(np.r_[True, key[1:] != key[:-1]])[0]
        counts = np.diff(np.r_[starts, len(key)])
        bad = np.nonzero(counts > 2)[0]
        if len(bad) == 0:
            break
        for g in bad:
            faces = face_of[starts[g]:starts[g] + counts[g]]
            faces = faces[alive[faces]]
            if len(faces) <= 2:
                continue
            best, best_score = None, -1.0
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    score = abs(float(n[faces[i]] @ n[faces[j]])) + 1e-6 * (area[faces[i]] + area[faces[j]])
                    if score > best_score:
                        best, best_score = (faces[i], faces[j]), score
            for f in faces:
                if f not in best:
                    alive[f] = False
    return F[alive]


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
    parser.add_argument("--through_nonmanifold", action="store_true", help="let hole rims run along non-manifold edges (needed: every open boundary chain ends on one)")
    parser.add_argument("--min_boundary_share", type=float, default=0.5, help="with --through_nonmanifold: share of a rim that must be real boundary edges")
    parser.add_argument("--fix_nonmanifold", action="store_true", help="first drop the extra faces at edges shared by 3+ faces (keeps the two that continue each other best); removes real geometry, report says how much")
    parser.add_argument("--debug_ply", default=None, help="also write the filled mesh with the new faces in red")
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if mesh.visual.kind == "vertex":
        C = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64)
    else:
        C = np.full((len(V), 3), 128.0)

    if args.fix_nonmanifold:
        n_before = len(F)
        F = drop_nonmanifold_faces(V, F)
        print("fix_nonmanifold: dropped {} of {} faces ({:.1f}%)".format(n_before - len(F), n_before, 100.0 * (n_before - len(F)) / n_before))
    F0 = len(F)
    # --through_nonmanifold: first the fast search that may run along non-manifold edges, then the exact minimum
    # cycle basis on the real boundary edges of the result, which still finds a few simple loops the first one misses
    modes = [True, False] if args.through_nonmanifold else [False]
    n_boundary, n_holes, n_added_v = None, 0, 0
    for mode in modes:
        cycles, directed, stats, n_b = find_holes(V, F, args.max_loop_edges, args.max_extent, mode, args.min_boundary_share)
        n_boundary = n_b if n_boundary is None else n_boundary
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
        V = np.concatenate([V, np.array(new_v).reshape(-1, 3)])
        C = np.concatenate([C, np.array(new_c).reshape(-1, 3)])
        F = np.concatenate([F, np.array(new_f, dtype=np.int64).reshape(-1, 3)])
        n_holes += len(cycles)
        n_added_v += len(new_v)
        print("pass {}: filled {} holes (components {components}, skipped: too big {too big}, mostly non-manifold {mostly non-manifold})".format(
            "with non-manifold rims" if mode else "real boundary only", len(cycles), **stats))
    V2, F2, C2 = V, F, C.clip(0, 255).astype(np.uint8)
    trimesh.Trimesh(V2, F2, vertex_colors=C2, process=False).export(args.out)
    if args.debug_ply:
        face_colors = np.tile(np.array([170, 170, 170, 255], dtype=np.uint8), (len(F2), 1))
        face_colors[F0:] = [255, 0, 0, 255]
        trimesh.Trimesh(V2, F2, face_colors=face_colors, process=False).export(args.debug_ply)
    he = np.concatenate([F2[:, [0, 1]], F2[:, [1, 2]], F2[:, [2, 0]]])
    _, cnt = np.unique(np.sort(he, axis=1), axis=0, return_counts=True)
    print("filled {} holes in total: +{} vertices, +{} faces | boundary edges {} -> {}".format(
        n_holes, n_added_v, len(F2) - F0, n_boundary, int((cnt == 1).sum())))


if __name__ == "__main__":
    main()
