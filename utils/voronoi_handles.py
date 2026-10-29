#
# Coarse-to-fine mesh deformation with Voronoi seeds on the surface (devel6, see DESIGN_voronoi_seeds.md).
#
# A few mesh vertices are seeds; every mesh vertex belongs to the Voronoi cell of its nearest seed, measured
# along the mesh (multi-source Dijkstra over the edge graph, so cells do not jump across thin gaps). Each seed
# has one displacement d_s and the vertices follow
#     x_v = x0_v + sum_s w_vs d_s
# hard: w_vs = 1 for the vertex's own seed (each cell moves rigidly);
# soft: Gaussian falloff over the own seed and the seeds of neighbouring cells (cells touching along a mesh
#       edge), width = the cell's mean vertex-to-seed distance, normalized to sum 1.
# The seed gradient is W^T g (g: vertex gradients) and d is updated with Adam. A round (new_seeds, then a new
# VoronoiHandles from the current positions) bakes the deformation into x0, so the mesh does not jump.
# Same interface as utils/large_steps.py: step() writes the mesh rows and zeroes their gradient.
#

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, connected_components
import torch


def edge_graph(x, faces):
    """Unique edges [E,2] of the faces and the symmetric scipy graph weighted by edge length."""
    f = faces.long()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    e = torch.unique(torch.sort(e, dim=1).values, dim=0)
    length = (x[e[:, 0]] - x[e[:, 1]]).norm(dim=1).clamp(min=1e-12)  # 0 would mean "no edge" to scipy
    i, j, w = e[:, 0].cpu().numpy(), e[:, 1].cpu().numpy(), length.cpu().numpy().astype(np.float64)
    n = x.shape[0]
    graph = sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(n, n)).tocsr()
    return e, graph


def surface_voronoi(graph, seeds):
    """Multi-source Dijkstra: per vertex its distance to the nearest seed and that seed's index into seeds."""
    dist, _, source = dijkstra(graph, directed=False, indices=seeds, min_only=True, return_predecessors=True)
    pos = np.full(graph.shape[0], -1, dtype=np.int64)
    pos[seeds] = np.arange(len(seeds))
    return dist, pos[source]


def farthest_point_seeds(x, num):
    """num vertex indices by farthest-point sampling (3D distance), from the vertex nearest the centroid."""
    num = min(num, x.shape[0])
    seeds = torch.empty(num, dtype=torch.long, device=x.device)
    seeds[0] = (x - x.mean(0)).norm(dim=1).argmin()
    d = (x - x[seeds[0]]).norm(dim=1)
    for k in range(1, num):
        seeds[k] = d.argmax()
        d = torch.minimum(d, (x - x[seeds[k]]).norm(dim=1))
    return seeds


def cover_components(graph, seeds):
    """Add one seed to every connected component of the mesh that has none (Dijkstra cannot reach it)."""
    _, comp = connected_components(graph, directed=False)
    has = np.zeros(comp.max() + 1, dtype=bool)
    has[comp[seeds]] = True
    first = np.full(comp.max() + 1, -1, dtype=np.int64)
    first[comp[::-1]] = np.arange(len(comp))[::-1]  # lowest vertex index of each component
    return np.concatenate([seeds, first[~has]])


class VoronoiHandles:
    def __init__(self, vertices, faces, seeds, blend="soft", betas=(0.9, 0.999), eps=1e-15):
        """vertices: [V,3] current mesh positions (they become x0); faces: [F,3] mesh-local indices;
        seeds: vertex indices (tensor or array). Components without a seed get one."""
        dev = vertices.device
        self.x0 = vertices.detach().clone()
        self.faces = faces.long()
        self.num_vertices = self.x0.shape[0]
        self.blend = blend
        self.edges, self.graph = edge_graph(self.x0, self.faces)
        seeds = np.unique(torch.as_tensor(seeds).cpu().numpy().astype(np.int64))
        seeds = cover_components(self.graph, seeds)
        dist, label = surface_voronoi(self.graph, seeds)
        self.seeds = torch.from_numpy(seeds).to(dev)
        self.label = torch.from_numpy(label).to(dev)
        self.dist = torch.from_numpy(dist).float().to(dev)
        S, V = self.seeds.numel(), self.num_vertices

        if blend == "hard":
            rows, cols, w = torch.arange(V, device=dev), self.label, torch.ones(V, device=dev)
        elif blend == "soft":
            # neighbouring cells: a mesh edge joins them
            la, lb = self.label[self.edges[:, 0]], self.label[self.edges[:, 1]]
            cross = la != lb
            pairs = torch.unique(torch.cat([torch.stack([la[cross], lb[cross]], 1), torch.stack([lb[cross], la[cross]], 1)], 0), dim=0)
            deg = torch.bincount(pairs[:, 0], minlength=S)
            off = torch.cumsum(deg, 0) - deg  # pairs are sorted by their first column
            cnt = deg[self.label]
            r_nb = torch.repeat_interleave(torch.arange(V, device=dev), cnt)
            k = torch.arange(r_nb.numel(), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
            c_nb = pairs[off[self.label[r_nb]] + k, 1]
            rows = torch.cat([torch.arange(V, device=dev), r_nb])
            cols = torch.cat([self.label, c_nb])
            d = torch.cat([self.dist, (self.x0[r_nb] - self.x0[self.seeds[c_nb]]).norm(dim=1)])
            # width: mean vertex-to-seed distance of the cell, at least half the mean edge
            n_cell = torch.bincount(self.label, minlength=S).float()
            h = torch.zeros(S, device=dev).index_add_(0, self.label, self.dist) / n_cell.clamp(min=1)
            mean_edge = (self.x0[self.edges[:, 0]] - self.x0[self.edges[:, 1]]).norm(dim=1).mean()
            h = h.clamp(min=0.5 * float(mean_edge))
            w = torch.exp(-0.5 * (d / h[cols]) ** 2)
            w = w / torch.zeros(V, device=dev).index_add_(0, rows, w)[rows]
        else:
            raise ValueError("blend must be hard or soft, got {}".format(blend))
        self.rows, self.cols, self.w = rows, cols, w[:, None]

        self.delta = torch.zeros(S, 3, device=dev)
        self.m1 = torch.zeros_like(self.delta)
        self.m2 = torch.zeros_like(self.delta)
        self.b1, self.b2 = betas
        self.eps = eps
        self.step_count = 0

    @property
    def num_seeds(self):
        return self.seeds.numel()

    def positions(self):
        return self.x0 + torch.zeros_like(self.x0).index_add_(0, self.rows, self.w * self.delta[self.cols])

    @torch.no_grad()
    def step(self, vertices, lr, idx):
        """Adam step on the seed displacements from the gradient of the mesh rows idx of vertices; writes the
        new positions there and zeroes their gradient (the regular optimizer must not move them too)."""
        assert idx.numel() == self.num_vertices
        if vertices.grad is None:
            return  # densification replaced the parameter after backward: no gradient this step
        g = vertices.grad[idx]
        g_seed = torch.zeros_like(self.delta).index_add_(0, self.cols, self.w * g[self.rows])
        self.step_count += 1
        self.m1.mul_(self.b1).add_(g_seed, alpha=1 - self.b1)
        self.m2.mul_(self.b2).addcmul_(g_seed, g_seed, value=1 - self.b2)
        m1 = self.m1 / (1 - self.b1 ** self.step_count)
        m2 = self.m2 / (1 - self.b2 ** self.step_count)
        self.delta -= lr * m1 / (m2.sqrt() + self.eps)
        vertices.data[idx] = self.positions()
        vertices.grad[idx] = 0

    @torch.no_grad()
    def new_seeds(self, num_new, vertex_error, max_per_cell=16):
        """The current seeds plus about num_new new ones, shared out over the cells in proportion to their summed
        vertex_error [V]; inside a cell each new seed goes to the vertex farthest (along the mesh, at the current
        positions) from all seeds so far. Returns vertex indices (numpy)."""
        S = self.num_seeds
        cell_err = torch.zeros(S, device=self.x0.device).index_add_(0, self.label, vertex_error.float().clamp(min=0))
        n_cell = torch.bincount(self.label, minlength=S)
        if cell_err.sum() <= 0:
            cell_err = n_cell.float()
        share = cell_err / cell_err.sum() * num_new
        want = torch.floor(share).long()
        rest = num_new - int(want.sum())
        if rest > 0:  # largest remainders
            want[torch.topk(share - want, min(rest, S)).indices] += 1
        want = torch.minimum(want, (n_cell - 1).clamp(min=0)).clamp(max=max_per_cell).cpu().numpy()

        _, graph = edge_graph(self.positions(), self.faces)
        label = self.label.cpu().numpy()  # the cells the error was summed over
        seeds = self.seeds.cpu().numpy()
        while want.max() > 0:
            dist, _ = surface_voronoi(graph, seeds)
            dist[seeds] = -1.0
            active = want[label] > 0
            v = np.nonzero(active)[0]
            order = np.lexsort((-dist[v], label[v]))  # by cell, farthest first
            v = v[order]
            first = np.ones(len(v), dtype=bool)
            first[1:] = label[v[1:]] != label[v[:-1]]
            far = v[first]
            far = far[dist[far] > 0]
            if len(far) == 0:
                break
            seeds = np.concatenate([seeds, far])
            want[label[far]] -= 1
            want[want < 0] = 0
        return seeds

    def cell_colors(self):
        """A random color per cell for each vertex (uint8 [V,3]), for looking at the cells."""
        g = torch.Generator().manual_seed(0)
        palette = torch.randint(40, 256, (self.num_seeds, 3), generator=g, dtype=torch.uint8)
        return palette[self.label.cpu()].numpy()
