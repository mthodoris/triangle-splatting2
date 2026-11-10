"""Exact per-pixel depth of the opaque triangles (Phase 1).

The rasterizer's depth channels hold the view-space z of each triangle's
*centroid*, constant over the triangle. Here the visible triangle per pixel
(allmap channel 6, `rend_ids`, at render resolution) is intersected with that
pixel's camera ray, which gives the exact z-depth of the surface actually drawn.

Conventions (checked in the CUDA code, see NOTES.md):
  * pixel i of an S-wide render sits at NDC (2i+1)/S - 1 (pixel centres),
  * NDC x = x_view / (z_view * tanfovx), y likewise, so the view ray through
    pixel (i, j) is d = (ndc_x * tanfovx, ndc_y * tanfovy, 1) (+y down, +z fwd),
  * depth is view-space z (not ray length), in scene (COLMAP) units.
"""
import math

import torch


def view_rays(view, H, W, device="cuda"):
    """Ray directions with z = 1 in view space, [H, W, 3], for an H x W render."""
    tx = math.tan(view.FoVx * 0.5)
    ty = math.tan(view.FoVy * 0.5)
    i = torch.arange(W, device=device, dtype=torch.float32)
    j = torch.arange(H, device=device, dtype=torch.float32)
    x = ((2 * i + 1) / W - 1) * tx
    y = ((2 * j + 1) / H - 1) * ty
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)


def world_to_view(view, pts):
    """[N,3] world -> [N,3] view space (row-vector convention of world_view_transform)."""
    M = view.world_view_transform  # = W2C^T
    return pts @ M[:3, :3] + M[3, :3]


def view_to_world(view, pts):
    C2W = torch.inverse(view.world_view_transform.T)
    return pts @ C2W[:3, :3].T + C2W[:3, 3]


def exact_depth(view, vertices, faces, rend_ids, rend_alpha, alpha_min=0.5, eps=1e-8):
    """Exact z-depth of the visible triangle per pixel.

    Args:
      vertices [V,3], faces [T,3] (world space, as in TriangleModel)
      rend_ids   [1,H,W] or [H,W] (allmap channel 6, render resolution)
      rend_alpha [1,H,W] or [H,W] (allmap channel 1, render resolution)
    Returns dict with
      depth [H,W]   (0 where invalid), valid [H,W] bool, tri_id [H,W] long (-1 invalid),
      bary  [H,W,3] barycentric coordinates of the hit on its triangle (for checks),
      points_world [H,W,3].
    Differentiable w.r.t. `vertices` (only through the visible triangles).
    """
    ids = rend_ids.reshape(rend_ids.shape[-2:])
    alpha = rend_alpha.reshape(rend_alpha.shape[-2:])
    H, W = ids.shape
    T = faces.shape[0]
    tri = torch.round(ids).long()
    valid = (alpha >= alpha_min) & (tri >= 0) & (tri < T) & torch.isfinite(ids)
    tri = torch.where(valid, tri, torch.full_like(tri, -1))

    d = view_rays(view, H, W, device=vertices.device)            # [H,W,3]
    vi = valid.nonzero(as_tuple=True)
    f = faces[tri[vi]].long()                                    # [N,3]
    p = world_to_view(view, vertices[f.reshape(-1)]).reshape(-1, 3, 3)  # [N,3,3]
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    n = torch.cross(e1, e2, dim=-1)
    dv = d[vi]                                                   # [N,3]
    denom = (n * dv).sum(-1)
    ok = denom.abs() > eps
    z = (n * p[:, 0]).sum(-1) / torch.where(ok, denom, torch.ones_like(denom))
    ok = ok & (z > 0)

    # Barycentrics of the hit point (for the "is the hit on the triangle" check).
    q = dv * z[:, None]
    nn2 = (n * n).sum(-1).clamp_min(eps)
    b1 = (torch.cross(q - p[:, 0], e2, dim=-1) * n).sum(-1) / nn2
    b2 = (torch.cross(e1, q - p[:, 0], dim=-1) * n).sum(-1) / nn2
    bary_v = torch.stack([1 - b1 - b2, b1, b2], dim=-1)

    depth = torch.zeros(H, W, device=vertices.device, dtype=z.dtype)
    bary = torch.zeros(H, W, 3, device=vertices.device, dtype=z.dtype)
    depth = depth.index_put(vi, torch.where(ok, z, torch.zeros_like(z)))
    bary = bary.index_put(vi, bary_v)
    valid = valid.clone()
    valid[vi] = ok
    tri[~valid] = -1
    pts_view = d * depth[..., None]
    pts_world = view_to_world(view, pts_view.reshape(-1, 3)).reshape(H, W, 3)
    return {"depth": depth, "valid": valid, "tri_id": tri, "bary": bary, "points_world": pts_world}


def render_exact_depth(view, triangles, render_pkg):
    """Convenience wrapper around a `triangle_renderer.render` output."""
    return exact_depth(view, triangles.get_vertices, triangles.get_triangle_indices,
                       render_pkg["rend_ids"], render_pkg["alpha_full"])
