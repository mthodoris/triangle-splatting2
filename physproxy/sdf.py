"""SDF field for the collision proxy (Phase 2). Pure PyTorch.

f(x) = prior(x) + scale * MLP([grid features(x), x_n])

* x is in world (scene) units; x_n = x normalized to [-1, 1] inside the AABB.
* Grid features: a dense multi-resolution pyramid of [C, R, R, R] grids
  (R = base_res * growth^l), sampled trilinearly with F.grid_sample
  (align_corners=True, padding "border"), concatenated over the levels.
* prior: the geometric init (sphere / plane / none) as an analytic SDF in scene
  units. The last MLP layer is zero-initialized, so at init f == prior exactly
  (stable start, |grad f| = 1). The network learns a residual on top of it.
* Sign: negative inside, positive outside, in scene units (COLMAP units here).
* grad f: central finite differences with step h (default: half the finest
  voxel), because double backward through 3D grid_sample is not implemented in
  PyTorch (checked on 2.7: "derivative for aten::grid_sampler_3d_backward is not
  implemented"). The FD gradient is differentiable w.r.t. the parameters, so
  the eikonal / normal losses can train the field.
* Outside the AABB the grid is clamped (border padding) and only the prior keeps
  varying, so queries are defined everywhere but only meaningful inside.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDFField(nn.Module):
    def __init__(self, aabb_min, aabb_max, base_res=16, n_levels=4, growth=2.0, feat_dim=4,
                 hidden=64, n_hidden=2, init="sphere", init_radius=0.5, init_center=None,
                 init_normal=(0.0, 0.0, 1.0), init_offset=0.0, grid_init_std=1e-4, out_scale=None):
        """
        aabb_min/aabb_max: [3] scene-unit box (the proxy domain).
        init: "sphere" (radius = init_radius * smallest half-extent, centered at
              init_center or the box center), "plane" (n.x - offset, n points
              to the outside/free side), or "none" (prior = 0).
        out_scale: scene units per MLP output unit (default: smallest half-extent / 4).
        """
        super().__init__()
        aabb_min = torch.as_tensor(aabb_min, dtype=torch.float32)
        aabb_max = torch.as_tensor(aabb_max, dtype=torch.float32)
        assert (aabb_max > aabb_min).all()
        self.register_buffer("aabb_min", aabb_min)
        self.register_buffer("aabb_max", aabb_max)
        half = 0.5 * (aabb_max - aabb_min)
        self.resolutions = [int(round(base_res * growth ** l)) for l in range(n_levels)]
        self.grids = nn.ParameterList([
            nn.Parameter(grid_init_std * torch.randn(1, feat_dim, r, r, r)) for r in self.resolutions])
        layers, d_in = [], feat_dim * n_levels + 3
        for _ in range(n_hidden):
            layers += [nn.Linear(d_in, hidden), nn.Softplus(beta=100)]
            d_in = hidden
        last = nn.Linear(d_in, 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.mlp = nn.Sequential(*layers, last)
        self.out_scale = float(out_scale) if out_scale is not None else float(half.min()) / 4

        self.init = init
        center = 0.5 * (aabb_min + aabb_max) if init_center is None else torch.as_tensor(init_center, dtype=torch.float32)
        self.register_buffer("init_center", center)
        self.init_radius = float(init_radius * half.min())
        n = torch.as_tensor(init_normal, dtype=torch.float32)
        self.register_buffer("init_normal", n / n.norm())
        self.init_offset = float(init_offset)

    # ---- geometry helpers -------------------------------------------------
    @property
    def voxel_size(self):
        """Finest voxel edge per axis, [3] scene units."""
        return (self.aabb_max - self.aabb_min) / (self.resolutions[-1] - 1)

    def normalize(self, x):
        return 2 * (x - self.aabb_min) / (self.aabb_max - self.aabb_min) - 1

    def inside_aabb(self, x):
        return ((x >= self.aabb_min) & (x <= self.aabb_max)).all(dim=-1)

    def prior(self, x):
        if self.init == "sphere":
            return (x - self.init_center).norm(dim=-1) - self.init_radius
        if self.init == "plane":
            return x @ self.init_normal - self.init_offset
        return torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)

    # ---- evaluation -------------------------------------------------------
    def features(self, xn):
        g = xn.reshape(1, -1, 1, 1, 3)
        feats = [F.grid_sample(grid, g, mode="bilinear", padding_mode="border", align_corners=True)
                 .reshape(grid.shape[1], -1).T for grid in self.grids]
        return torch.cat(feats, dim=-1)

    def forward(self, x):
        """x [..., 3] scene units -> f [...] scene units."""
        shp = x.shape[:-1]
        x = x.reshape(-1, 3)
        xn = self.normalize(x)
        h = torch.cat([self.features(xn), xn.clamp(-1, 1)], dim=-1)
        f = self.prior(x) + self.out_scale * self.mlp(h).squeeze(-1)
        return f.reshape(shp)

    def default_fd_step(self):
        return 0.5 * float(self.voxel_size.min())

    def gradient(self, x, h=None, return_sdf=False):
        """Central-difference grad f, [..., 3]. One batched call of 6 offsets."""
        h = self.default_fd_step() if h is None else h
        shp = x.shape[:-1]
        x = x.reshape(-1, 1, 3)
        off = h * torch.cat([torch.eye(3, device=x.device), -torch.eye(3, device=x.device)], 0)  # [6,3]
        f6 = self.forward((x + off[None]).reshape(-1, 3)).reshape(-1, 6)
        g = (f6[:, :3] - f6[:, 3:]) / (2 * h)
        g = g.reshape(*shp, 3)
        if return_sdf:
            # Mean of the 6 samples equals f(x) up to O(h^2) curvature; evaluate exactly instead.
            return self.forward(x.reshape(*shp, 3)), g
        return g

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def param_groups(self, lr_grid=1e-2, lr_mlp=1e-3):
        return [{"params": list(self.grids.parameters()), "lr": lr_grid, "name": "sdf_grid"},
                {"params": list(self.mlp.parameters()), "lr": lr_mlp, "name": "sdf_mlp"}]


def aabb_from_points(points, cam_centers, pct=(2.0, 98.0), radius_mult=1.5, pad=0.05):
    """Region of interest for the proxy.

    Keeps SfM points within radius_mult * (max camera distance) of the camera
    centroid (drops sky / far background), takes robust per-axis percentiles,
    and pads the box by `pad` of its extent. Returns (min [3], max [3]) tensors.
    """
    points = torch.as_tensor(points, dtype=torch.float32)
    cams = torch.as_tensor(cam_centers, dtype=torch.float32)
    c = cams.mean(0)
    r = (cams - c).norm(dim=-1).max() * radius_mult
    p = points[(points - c).norm(dim=-1) <= r]
    lo = torch.quantile(p, pct[0] / 100, dim=0)
    hi = torch.quantile(p, pct[1] / 100, dim=0)
    ext = hi - lo
    return lo - pad * ext, hi + pad * ext
