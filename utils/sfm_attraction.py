#
# Attraction of the mesh to the SfM points (devel6, --lambda_sfm).
#
# The photometric loss does not move the coarse mesh onto thin structures (leaves): measured against the garden SfM
# points, no run gets closer to the foliage points than the init mesh. The SfM points sit on the real surfaces, so
# each one pulls its nearest mesh vertex along that vertex's normal (point-to-plane):
#     r = ((p - x_v) . n_v) / init_edge,   loss = mean_p charbonnier(r)
# Only points with reprojection error below --sfm_max_error and within --sfm_max_dist init edges of the mesh count;
# farther ones belong to things the mesh does not model (the free background group covers those).
# Moving only along the normal avoids sliding; several points on one vertex pull it to their mean plane offset.
# The nearest vertices and the normals are recomputed every 500 iterations and whenever the vertex set changes.
#

import numpy as np
import torch
from scipy.spatial import cKDTree

from scene.colmap_loader import read_points3D_binary


class SfmAttraction:
    def __init__(self, points3d_bin, max_error=1.5, max_dist=2.0, eps=0.1):
        xyz, _, err = read_points3D_binary(points3d_bin)
        keep = err[:, 0] < max_error
        self.points = torch.from_numpy(xyz[keep]).float().cuda()
        self.num_loaded = int(keep.size)
        self.max_dist = max_dist
        self.eps = eps
        self.num_vertices = -1  # size of the vertex tensor at the last assignment

    @torch.no_grad()
    def assign(self, vertices, mesh_idx, mesh_faces, init_edge):
        """vertices: all model vertices [N,3]; mesh_idx: indices of the mesh vertices; mesh_faces: mesh triangles in
        the numbering of vertices. Pairs each kept point with its nearest mesh vertex and stores that vertex's normal."""
        x = vertices.detach()
        f = mesh_faces.long()
        fn = torch.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]], dim=1)  # area-weighted
        vn = torch.zeros_like(x)
        for k in range(3):
            vn.index_add_(0, f[:, k], fn)
        xm = x[mesh_idx].cpu().numpy()
        d, j = cKDTree(xm).query(self.points.cpu().numpy(), workers=-1)
        near = d < self.max_dist * init_edge
        near_t = torch.from_numpy(near).cuda()
        self.point_idx = torch.nonzero(near_t, as_tuple=True)[0]
        self.vertex_idx = mesh_idx[torch.from_numpy(j[near]).cuda()]
        self.normal = torch.nn.functional.normalize(vn[self.vertex_idx], dim=1)
        self.init_edge = float(init_edge)
        self.num_vertices = x.shape[0]
        return int(near.sum())

    def residual(self, vertices):
        """Signed point-to-plane offsets in init edges, one per assigned point."""
        p = self.points[self.point_idx]
        return ((p - vertices[self.vertex_idx]) * self.normal).sum(1) / self.init_edge

    def loss(self, vertices):
        r = self.residual(vertices)
        return (torch.sqrt(r * r + self.eps * self.eps) - self.eps).mean()
