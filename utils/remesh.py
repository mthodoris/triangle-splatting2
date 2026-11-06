#
# Continuous remeshing of the mesh group (devel7 E2), after Palfinger, "Continuous Remeshing for Inverse Rendering" (2022).
#
# Every few hundred iterations the mesh gets one remeshing pass towards a per-vertex target edge length L:
#   split    edges longer than 4/3 L at their midpoint (conforming, neighbours re-triangulated);
#   collapse edges shorter than 4/5 L to their midpoint (interior, manifold, link condition, no normal flip,
#            no new edge longer than 4/3 L; an independent set per pass, so collapses never share a face);
#   flip     interior edges where it brings the four vertices' valences closer to 6 (flat places only);
#   relax    move every interior vertex part of the way to its neighbours' centroid, along the surface only.
# L shrinks on a schedule (coarse to fine) down to a screen-space floor: an edge never gets shorter than
# --remesh_px image pixels in the training view where that vertex looks largest.
# Between passes the mesh vertices move with RemeshStep: Adam with a scalar second moment per vertex and the step
# measured in local edge lengths, so coarse parts move far and fine parts move little.
#
# All operations work on the mesh group in its own (local) vertex numbering and report, for every output vertex,
# the two input vertices its attributes come from (src_a, src_b: equal for a vertex that is kept).
#

import torch


def unique_edges(F):
    """Unique edges [E,2] (sorted endpoints) and each face's edge ids [T,3]; edge k joins corners k and k+1."""
    ends = torch.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2)
    E, eid = torch.unique(torch.sort(ends, 1).values, dim=0, return_inverse=True)
    return E, eid.reshape(-1, 3)


def face_normals(V, F):
    """Unnormalized face normals (twice the area)."""
    return torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=1)


def tri_normals(P):
    """Unnormalized normals of triangles given by their corner positions [N,3,3]."""
    return torch.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], dim=1)


def unit(x):
    return torch.nn.functional.normalize(x, dim=-1)


def boundary_vertices(n, E, eid):
    """[n] bool: vertex on an edge that does not have exactly two faces (boundary or non-manifold)."""
    count = torch.bincount(eid.reshape(-1), minlength=E.shape[0])
    bad = torch.zeros(n, dtype=torch.bool, device=E.device)
    bad[E[count != 2].reshape(-1)] = True
    return bad, count


def vertex_edge_length(V, F):
    """[n] mean length of the edges at each vertex (0 for a vertex without edges)."""
    E, _ = unique_edges(F)
    le = (V[E[:, 0]] - V[E[:, 1]]).norm(dim=1)
    s = torch.zeros(V.shape[0], device=V.device).index_add_(0, E.reshape(-1), le.repeat_interleave(2))
    c = torch.bincount(E.reshape(-1), minlength=V.shape[0]).float()
    return s / c.clamp(min=1)


@torch.no_grad()
def pixel_footprint(V, cameras):
    """[n] smallest world size of one image pixel at each vertex over the cameras that see it (in the frustum, in
    front); inf for vertices no camera sees. Occlusion is ignored."""
    best = torch.full((V.shape[0],), float("inf"), device=V.device)
    Vh = torch.cat([V, torch.ones_like(V[:, :1])], 1)
    for cam in cameras:
        z = (Vh @ cam.world_view_transform)[:, 2]
        p = Vh @ cam.full_proj_transform
        w = p[:, 3].clamp(min=1e-6)
        inside = (z > 0.01) & ((p[:, 0] / w).abs() < 1.0) & ((p[:, 1] / w).abs() < 1.0)
        size = z * 2.0 * torch.tan(torch.tensor(cam.FoVx * 0.5)) / cam.image_width
        best = torch.where(inside, torch.minimum(best, size), best)
    return best


def _vertex_faces(F, n):
    """CSR vertex -> incident faces: (offsets [n], counts [n], faces [3T])."""
    v = F.reshape(-1)
    f = torch.arange(F.shape[0], device=F.device).repeat_interleave(3)
    order = torch.argsort(v)
    counts = torch.bincount(v, minlength=n)
    offsets = torch.cumsum(counts, 0) - counts
    return offsets, counts, f[order]


def _expand(verts, offsets, counts, vf):
    """For each query vertex i: its incident faces. Returns (query index [P], face [P])."""
    c = counts[verts]
    q = torch.arange(verts.numel(), device=verts.device).repeat_interleave(c)
    start = offsets[verts].repeat_interleave(c)
    k = torch.arange(q.numel(), device=verts.device) - (torch.cumsum(c, 0) - c).repeat_interleave(c)
    return q, vf[start + k]


def split_long(V, F, L, max_faces, templates):
    """Split every edge longer than 4/3 of its target (the mean L of its endpoints), longest relative to its
    target first while the face count stays under max_faces. Returns V, F, src_a, src_b, L, number split."""
    n, T = V.shape[0], F.shape[0]
    ident = torch.arange(n, device=V.device)
    E, eid = unique_edges(F)
    ratio = (V[E[:, 0]] - V[E[:, 1]]).norm(dim=1) / (0.5 * (L[E[:, 0]] + L[E[:, 1]]))
    cand = torch.nonzero(ratio > 4.0 / 3.0, as_tuple=True)[0]
    allowed = max(0, (max_faces - T) // 2)  # an interior edge split adds two faces
    if cand.numel() > allowed:
        cand = cand[torch.topk(ratio[cand], allowed).indices]
    if cand.numel() == 0:
        return V, F, ident, ident, L, 0
    split = torch.zeros(E.shape[0], dtype=torch.bool, device=V.device)
    split[cand] = True
    midpoint = torch.full((E.shape[0],), -1, dtype=torch.long, device=V.device)
    midpoint[cand] = n + torch.arange(cand.numel(), device=V.device)
    u, w = E[cand, 0], E[cand, 1]

    s = split[eid]
    pattern = s[:, 0].long() + 2 * s[:, 1].long() + 4 * s[:, 2].long()
    slots = torch.cat([F, midpoint[eid]], 1)  # v0, v1, v2, m0, m1, m2
    children = [F[pattern == 0]]
    for code in range(1, 8):
        idx = torch.nonzero(pattern == code, as_tuple=True)[0]
        if idx.numel():
            t = torch.tensor(templates[code], device=V.device)
            children.append(slots[idx][:, t].reshape(-1, 3))
    V = torch.cat([V, 0.5 * (V[u] + V[w])], 0)
    L = torch.cat([L, 0.5 * (L[u] + L[w])], 0)
    return V, torch.cat(children, 0), torch.cat([ident, u]), torch.cat([ident, w]), L, int(cand.numel())


def collapse_short(V, F, L, min_cos=0.5, max_share=6):
    """Collapse edges shorter than 4/5 of their target to their midpoint: interior edges between interior
    vertices, an independent set (no two collapses touch the same face), link condition (the endpoints share
    exactly the two opposite vertices), no surrounding face turning by more than acos(min_cos), no new edge
    longer than 4/3 of its target. Returns V, F, src_a, src_b, L, number collapsed."""
    n = V.shape[0]
    ident = torch.arange(n, device=V.device)
    E, eid = unique_edges(F)
    bnd, count = boundary_vertices(n, E, eid)
    target = 0.5 * (L[E[:, 0]] + L[E[:, 1]])
    le = (V[E[:, 0]] - V[E[:, 1]]).norm(dim=1)
    cand = torch.nonzero((le < 0.8 * target) & (count == 2) & ~bnd[E[:, 0]] & ~bnd[E[:, 1]], as_tuple=True)[0]
    if cand.numel() == 0:
        return V, F, ident, ident, L, 0
    if cand.numel() > n // max_share:
        # memory: the shortest relative to their target first (an independent set holds far fewer anyway)
        cand = cand[torch.topk(le[cand] / target[cand], n // max_share, largest=False).indices]
    a, b = E[cand, 0], E[cand, 1]
    C = cand.numel()
    offsets, counts, vf = _vertex_faces(F, n)
    qa, fa = _expand(a, offsets, counts, vf)
    qb, fb = _expand(b, offsets, counts, vf)
    q = torch.cat([qa, qb])
    f = torch.cat([fa, fb])

    # independent set: a candidate wins if it has the highest random priority at every vertex it touches
    prio = torch.randperm(C, device=V.device)
    tq, tv = q.repeat_interleave(3), F[f].reshape(-1)
    vmax = torch.full((n,), -1, dtype=torch.long, device=V.device).scatter_reduce(0, tv, prio[tq], "amax")
    ok = torch.zeros(C, device=V.device).index_add_(0, tq, (vmax[tv] != prio[tq]).float()) == 0

    # link condition: exactly two vertices adjacent to both endpoints
    na = torch.unique(qa.repeat_interleave(3) * n + F[fa].reshape(-1))
    nb = torch.unique(qb.repeat_interleave(3) * n + F[fb].reshape(-1))
    common = na[torch.isin(na, nb)]
    cq, cv = common // n, common % n
    keep = (cv != a[cq]) & (cv != b[cq])
    ncommon = torch.bincount(cq[keep], minlength=C)
    ok &= ncommon == 2

    # geometry: the faces that survive (not containing both endpoints) with a, b moved to the midpoint
    p = 0.5 * (V[a] + V[b])
    Ff = F[f]
    has_a, has_b = (Ff == a[q, None]), (Ff == b[q, None])
    survive = ~(has_a.any(1) & has_b.any(1))
    moved = has_a | has_b
    Vf = V[Ff]
    Vn = torch.where(moved[..., None], p[q][:, None, :], Vf)
    turned = (unit(tri_normals(Vf)) * unit(tri_normals(Vn))).sum(1) < min_cos
    # new edge lengths: from the midpoint to the face's other corners
    dmax = (Vn - p[q][:, None, :]).norm(dim=2).max(1).values
    too_long = dmax > (4.0 / 3.0) * target[cand][q]
    bad = survive & (turned | too_long)
    ok &= torch.zeros(C, device=V.device).index_add_(0, q, bad.float()) == 0

    sel = torch.nonzero(ok, as_tuple=True)[0]
    if sel.numel() == 0:
        return V, F, ident, ident, L, 0
    a, b = a[sel], b[sel]
    V = V.clone()
    V[a] = 0.5 * (V[a] + V[b])
    L = L.clone()
    L[a] = 0.5 * (L[a] + L[b])
    src_b = ident.clone()
    src_b[a] = b
    remap = ident.clone()
    remap[b] = a
    F = remap[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])]
    alive = torch.ones(n, dtype=torch.bool, device=V.device)
    alive[b] = False
    new_id = torch.full((n,), -1, dtype=torch.long, device=V.device)
    new_id[alive] = torch.arange(int(alive.sum()), device=V.device)
    return V[alive], new_id[F], ident[alive], src_b[alive], L[alive], int(sel.numel())


def flip_edges(V, F, max_angle_cos=0.8, min_cos=0.5):
    """Flip interior edges whose flip brings the valences of the four vertices closer to 6, only between
    consistently oriented faces that are nearly coplanar (normals within acos(max_angle_cos)), when the new
    edge does not exist yet and both new faces stay within acos(min_cos) of the old normals. Independent set:
    no two flips share a vertex. Returns F, number flipped."""
    n, T = V.shape[0], F.shape[0]
    E, eid = unique_edges(F)
    bnd, count = boundary_vertices(n, E, eid)
    e = eid.reshape(-1)
    f = torch.arange(T, device=V.device).repeat_interleave(3)
    k = torch.arange(3, device=V.device).repeat(T)
    f1 = torch.full((E.shape[0],), T, dtype=torch.long, device=V.device).scatter_reduce(0, e, f, "amin")
    f2 = torch.full((E.shape[0],), -1, dtype=torch.long, device=V.device).scatter_reduce(0, e, f, "amax")
    cand = torch.nonzero(count == 2, as_tuple=True)[0]
    k1 = torch.zeros(E.shape[0], dtype=torch.long, device=V.device)
    k2 = torch.zeros(E.shape[0], dtype=torch.long, device=V.device)
    first = f == f1[e]
    k1[e[first]] = k[first]
    k2[e[~first]] = k[~first]
    f1, f2, k1, k2 = f1[cand], f2[cand], k1[cand], k2[cand]
    a, b, c = F[f1, k1], F[f1, (k1 + 1) % 3], F[f1, (k1 + 2) % 3]
    consistent = (F[f2, k2] == b) & (F[f2, (k2 + 1) % 3] == a)
    d = F[f2, (k2 + 2) % 3]

    val = torch.bincount(E.reshape(-1), minlength=n)
    dev = lambda x: (x - 6).abs()
    before = dev(val[a]) + dev(val[b]) + dev(val[c]) + dev(val[d])
    after = dev(val[a] - 1) + dev(val[b] - 1) + dev(val[c] + 1) + dev(val[d] + 1)
    ok = consistent & (after < before) & (c != d) & (val[a] > 3) & (val[b] > 3)
    ok &= ~(bnd[a] | bnd[b] | bnd[c] | bnd[d])
    # the new edge (c, d) must not exist yet
    keys = torch.sort(E[:, 0] * n + E[:, 1]).values
    new_key = torch.minimum(c, d) * n + torch.maximum(c, d)
    pos = torch.searchsorted(keys, new_key).clamp(max=keys.numel() - 1)
    ok &= keys[pos] != new_key
    # geometry
    n1, n2 = unit(face_normals(V, F[f1])), unit(face_normals(V, F[f2]))
    g1 = unit(face_normals(V, torch.stack([c, a, d], 1)))
    g2 = unit(face_normals(V, torch.stack([d, b, c], 1)))
    navg = unit(n1 + n2)
    ok &= ((n1 * n2).sum(1) > max_angle_cos) & ((g1 * navg).sum(1) > min_cos) & ((g2 * navg).sum(1) > min_cos)
    idx = torch.nonzero(ok, as_tuple=True)[0]
    if idx.numel() == 0:
        return F, 0
    # independent set over vertices (no two flips share any of their four vertices, so the valence change each
    # one assumed is exact and every flip improves; flips sharing a vertex would undo each other pass after pass),
    # and no two flips creating the same edge
    prio = torch.randperm(idx.numel(), device=V.device)
    quad = torch.stack([a[idx], b[idx], c[idx], d[idx]], 1)
    vmax = torch.full((n,), -1, dtype=torch.long, device=V.device).scatter_reduce(0, quad.reshape(-1), prio.repeat_interleave(4), "amax")
    win = (vmax[quad] == prio[:, None]).all(1)
    idx = idx[win]
    _, inv, cnt = torch.unique(new_key[idx], return_inverse=True, return_counts=True)
    idx = idx[cnt[inv] == 1]
    F = F.clone()
    F[f1[idx]] = torch.stack([c[idx], a[idx], d[idx]], 1)
    F[f2[idx]] = torch.stack([d[idx], b[idx], c[idx]], 1)
    return F, int(idx.numel())


def relax(V, F, lam):
    """Tangential smoothing: interior vertices move lam of the way to their neighbours' centroid, with the
    component along the vertex normal removed (the surface does not shrink)."""
    if lam <= 0:
        return V
    n = V.shape[0]
    E, eid = unique_edges(F)
    bnd, _ = boundary_vertices(n, E, eid)
    s = torch.zeros_like(V).index_add_(0, E[:, 0], V[E[:, 1]]).index_add_(0, E[:, 1], V[E[:, 0]])
    deg = torch.bincount(E.reshape(-1), minlength=n).float()
    nrm = unit(torch.zeros_like(V).index_add_(0, F.reshape(-1), face_normals(V, F).repeat_interleave(3, dim=0)))
    d = s / deg.clamp(min=1)[:, None] - V
    d = d - (d * nrm).sum(1, keepdim=True) * nrm
    move = (~bnd & (deg > 0))[:, None]
    return torch.where(move, V + lam * d, V)


@torch.no_grad()
def remesh(V, F, L, max_faces, relax_lambda, templates):
    """One pass: split, collapse, flip, relax. Returns V, F, src_a, src_b (into the input vertices), stats."""
    V1, F1, a1, b1, L1, ns = split_long(V, F, L, max_faces, templates)
    V2, F2, a2, b2, _, nc = collapse_short(V1, F1, L1)
    F3, nf = flip_edges(V2, F2)
    V3 = relax(V2, F3, relax_lambda)
    # compose the sources: every vertex comes from at most two input vertices. A kept vertex keeps its split
    # sources; a collapse joins the first source of each side (close enough for attributes that keep training).
    kept = a2 == b2
    src_a = a1[a2]
    src_b = torch.where(kept, b1[a2], a1[b2])
    return V3, F3, src_a, src_b, {"split": ns, "collapsed": nc, "flipped": nf}


class RemeshStep:
    """Adam for the mesh vertices with one second moment per vertex (isotropic steps) and the step measured in
    local edge lengths: x -= lr * h * m / (sqrt(v) + eps), h = mean edge length at the vertex. Same interface as
    utils/voronoi_handles.py: step() writes the mesh rows and zeroes their gradient. A new one is made after
    every remeshing pass (the state starts over)."""

    def __init__(self, h, b1=0.9, b2=0.999, eps=1e-15):
        self.h = h[:, None]
        self.m1 = torch.zeros(h.shape[0], 3, device=h.device)
        self.m2 = torch.zeros(h.shape[0], 1, device=h.device)
        self.b1, self.b2, self.eps = b1, b2, eps
        self.step_count = 0

    @property
    def num_vertices(self):
        return self.h.shape[0]

    @torch.no_grad()
    def step(self, vertices, lr, idx):
        assert idx.numel() == self.num_vertices
        if vertices.grad is None:
            return
        g = vertices.grad[idx]
        if lr > 0:
            self.step_count += 1
            self.m1.mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.m2.mul_(self.b2).add_((g * g).mean(1, keepdim=True), alpha=1 - self.b2)
            m1 = self.m1 / (1 - self.b1 ** self.step_count)
            m2 = self.m2 / (1 - self.b2 ** self.step_count)
            vertices.data[idx] -= lr * self.h * m1 / (m2.sqrt() + self.eps)
        vertices.grad[idx] = 0
