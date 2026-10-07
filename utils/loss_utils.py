#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def triangle_probability_loss(vertices, _triangle_indices, _closest_vertices, theta=10.0):
    """
    Computes the probability loss for triangles to maximize their separation from nearby points.
    
    For each triangle:
    1. Computes circumcenter and circumradius
    2. Finds closest point among K neighbors
    3. Calculates signed distance (closest_distance - circumradius)
    4. Computes probability = sigmoid(signed_distance * theta)
    
    Returns the negative mean probability (loss to minimize)
    """
    # Get triangle vertices [T, 3, 3]
    tri_pts = vertices[_triangle_indices]
    
    # Compute circumcenters and circumradii
    A, B, C = tri_pts.unbind(dim=1)
    AB = B - A
    AC = C - A
    n = torch.cross(AB, AC, dim=1)
    
    # Vector magnitudes
    AB2 = (AB * AB).sum(dim=1, keepdim=True)
    AC2 = (AC * AC).sum(dim=1, keepdim=True)
    
    # Denominator (2 * |n|^2)
    den = 2.0 * (n * n).sum(dim=1, keepdim=True)
    
    # Identify degenerate triangles (den near zero)
    eps = 1e-12
    degenerate = (den.abs() < eps).squeeze(1)
    
    # Compute circumcenters (use centroid for degenerate triangles)
    term1 = AB2 * torch.cross(AC, n, dim=1)
    term2 = AC2 * torch.cross(n, AB, dim=1)
    circumcenters = A + (term1 + term2) / den
    
    # For degenerate triangles, use centroid instead
    centroid = tri_pts.mean(dim=1)
    circumcenters = torch.where(degenerate.unsqueeze(1), centroid, circumcenters)
    
    # Compute circumradii (distance from circumcenter to vertices)
    dists_to_vertices = torch.norm(tri_pts - circumcenters.unsqueeze(1), dim=2)
    circumradii = dists_to_vertices.max(dim=1).values
    
    # Get closest points for each triangle [T, K, 3]
    closest_points = vertices[_closest_vertices]
    
    # Compute distances to closest points
    dists_to_neighbors = torch.norm(
        closest_points - circumcenters.unsqueeze(1),
        dim=2
    )
    
    # Find minimum distance for each triangle
    min_dists, _ = torch.min(dists_to_neighbors, dim=1)
    
    # Signed distance = (closest distance) - circumradius
    signed_dist = min_dists - circumradii
    
    # For degenerate triangles, set signed_dist to large negative value
    #signed_dist = torch.where(degenerate, -1e6 * torch.ones_like(signed_dist), signed_dist)
    
    # Compute probability using sigmoid
    probability = torch.sigmoid(-theta * signed_dist)
    
    # Loss is negative mean probability (to maximize probability)
    return torch.mean(probability)


def u_shaped_opacity_loss(x, center=0.1, width=0.03):
    # Normalized distance to center (e.g., 0.1)
    penalty = torch.exp(-((x - center) ** 2) / (2 * width ** 2))
    return penalty.mean()

def binarization_loss(x, eps=1e-6):
    x = torch.clamp(x, eps, 1 - eps)  # avoid log(0)
    return -x * torch.log(x) - (1 - x) * torch.log(1 - x)

def equilateral_regularizer(triangles):

    nan_mask = torch.isnan(triangles).any(dim=(1, 2))
    if nan_mask.any():
        print("NaN detected in triangle(s):")

    v0 = triangles[:, 1, :] - triangles[:, 0, :]
    v1 = triangles[:, 2, :] - triangles[:, 0, :]
    cross = torch.cross(v0, v1, dim=1)
    area = 0.5 * torch.norm(cross, dim=1)

    return area


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def lp_loss(pred, target, p=0.7, eps=1e-6):
    """
    Computes Lp loss with 0 < p < 1.
    Args:
        pred: (N, C, H, W) predicted image
        target: (N, C, H, W) groundtruth image
        p: norm degree < 1
        eps: small constant for numerical stability
    """
    diff = torch.abs(pred - target) + eps
    loss = torch.pow(diff, p).mean()
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def mesh_topology(faces):
    """
    Edge connectivity of an indexed triangle mesh.

    Returns:
        edges: [E,2] unique undirected edges (sorted vertex ids)
        manifold: [M] indices into edges of the edges shared by exactly two faces
        f0, f1: [M] the two faces adjacent to each manifold edge
        boundary_vertices: [B] vertex ids on boundary edges (edges with a single face)
    """
    f = faces.long()
    T = f.shape[0]
    e = torch.sort(torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0), dim=1).values
    fid = torch.arange(T, device=f.device).repeat(3)
    edges, inv, counts = torch.unique(e, dim=0, return_inverse=True, return_counts=True)
    order = torch.argsort(inv, stable=True)
    start = torch.cumsum(counts, 0) - counts
    manifold = torch.nonzero(counts == 2, as_tuple=True)[0]
    f0 = fid[order[start[manifold]]]
    f1 = fid[order[start[manifold] + 1]]
    boundary_vertices = torch.unique(edges[counts == 1])
    return edges, manifold, f0, f1, boundary_vertices


def face_normals(vertices, faces):
    v = vertices[faces.long()]
    return F.normalize(torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=1), dim=1)


def crease_weights(vertices, faces, topology, sigma):
    """
    Per-edge weights in [0,1]: ~1 on flat regions, ~0 across sharp creases (detached).
    Boundary and non-manifold edges get weight 1.
    """
    edges, manifold, f0, f1, _ = topology
    with torch.no_grad():
        n = face_normals(vertices, faces)
        cos = (n[f0] * n[f1]).sum(1).abs()  # abs: face winding is not guaranteed consistent
        w = torch.ones(edges.shape[0], device=vertices.device)
        w[manifold] = torch.exp(-(1.0 - cos) / sigma)
    return w


def laplacian_loss(vertices, edges, weights=None, fixed_vertices=None):
    """
    Uniform (optionally edge-weighted) Laplacian smoothing, normalized by the mean edge length
    so that it does not depend on the scene scale. fixed_vertices (e.g. boundary vertices) are
    left out, otherwise the uniform Laplacian pulls open boundaries inwards and shrinks the mesh.
    Each vertex is also scaled by its smallest incident edge weight: a vertex on a crease has
    weight-1 neighbours on both sides, whose average lies off the crease, so edge weights alone
    would still round it off.
    """
    i, j = edges[:, 0], edges[:, 1]
    w = torch.ones(i.shape[0], device=vertices.device) if weights is None else weights
    nb = torch.zeros_like(vertices).index_add_(0, i, w[:, None] * vertices[j]).index_add_(0, j, w[:, None] * vertices[i])
    deg = torch.zeros(vertices.shape[0], device=vertices.device).index_add_(0, i, w).index_add_(0, j, w)
    used = deg > 1e-8
    if fixed_vertices is not None:
        used[fixed_vertices] = False
    if not used.any():
        return torch.zeros((), device=vertices.device)
    vertex_w = torch.ones(vertices.shape[0], device=vertices.device)
    vertex_w = vertex_w.scatter_reduce(0, i, w, "amin").scatter_reduce(0, j, w, "amin")
    lap = vertices[used] - nb[used] / deg[used, None]
    scale = (vertices[i] - vertices[j]).norm(dim=1).mean().detach() + 1e-12
    return (vertex_w[used] * lap.norm(dim=1) / scale).mean()


def edge_aware_normal_consistency(vertices, faces, topology, sigma):
    """
    1 - cos between adjacent face normals, down-weighted across sharp creases so they stay sharp.
    """
    _, _, f0, f1, _ = topology
    if f0.numel() == 0:
        return torch.zeros((), device=vertices.device)
    n = face_normals(vertices, faces)
    cos = (n[f0] * n[f1]).sum(1).abs()
    w = torch.exp(-(1.0 - cos.detach()) / sigma)
    return (w * (1.0 - cos)).mean()


def edge_aware_smoothness(normal, image, alpha):
    """
    Image-space smoothness of a [3,H,W] normal map, relaxed where the [3,H,W] image has edges.
    """
    dn_x = (normal[:, :, 1:] - normal[:, :, :-1]).abs().sum(0)
    dn_y = (normal[:, 1:, :] - normal[:, :-1, :]).abs().sum(0)
    w_x = torch.exp(-alpha * (image[:, :, 1:] - image[:, :, :-1]).abs().mean(0))
    w_y = torch.exp(-alpha * (image[:, 1:, :] - image[:, :-1, :]).abs().mean(0))
    return (dn_x * w_x).mean() + (dn_y * w_y).mean()


def tangential_laplacian_loss(vertices, edges, normals, fixed_vertices=None):
    """
    Uniform Laplacian with the component along the vertex normal removed: evens out the
    triangles along the surface (against stretching and slivers) without pulling the surface
    itself. normals: [V,3] unit vertex normals (e.g. the init-mesh anchor normals).
    """
    i, j = edges[:, 0], edges[:, 1]
    ones = torch.ones(i.shape[0], device=vertices.device)
    nb = torch.zeros_like(vertices).index_add_(0, i, vertices[j]).index_add_(0, j, vertices[i])
    deg = torch.zeros(vertices.shape[0], device=vertices.device).index_add_(0, i, ones).index_add_(0, j, ones)
    used = deg > 0
    if fixed_vertices is not None:
        used[fixed_vertices] = False
    if not used.any():
        return torch.zeros((), device=vertices.device)
    lap = vertices[used] - nb[used] / deg[used, None]
    n = normals[used]
    lap_t = lap - (lap * n).sum(1, keepdim=True) * n
    scale = (vertices[i] - vertices[j]).norm(dim=1).mean().detach() + 1e-12
    return (lap_t.norm(dim=1) / scale).mean()
