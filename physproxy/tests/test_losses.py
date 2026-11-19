"""Phase 3 known-answer checks for physproxy/losses.py on planes and spheres."""
import math
import os
import sys

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physproxy.sdf import SDFField, fd_gradient  # noqa: E402
from physproxy.depth import exact_depth  # noqa: E402
from physproxy import losses as L  # noqa: E402
from physproxy.tests.test_depth_and_log import FakeView  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Analytic:
    """Analytic SDF wrapper with the same interface as SDFField."""
    def __init__(self, fn, h=1e-4):
        self.fn, self.h = fn, h

    def __call__(self, x):
        return self.fn(x)

    def gradient(self, x):
        return fd_gradient(self.fn, x, self.h)


def sphere(r, c=(0.0, 0.0, 0.0), scale=1.0):
    c = torch.tensor(c, device=DEV)
    return Analytic(lambda x: scale * ((x - c).norm(dim=-1) - r))


def plane(n, off):
    n = torch.nn.functional.normalize(torch.tensor(n, device=DEV, dtype=torch.float32), dim=0)
    return Analytic(lambda x: x @ n - off)


def icosphere(r, sub=5):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    return (torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=DEV),
            torch.tensor(np.asarray(m.faces), dtype=torch.int32, device=DEV))


# ------------------------------------------------------------ sampling
def test_sample_surface_area_weighted_and_on_triangles():
    torch.manual_seed(0)
    # Two triangles in z=0 with areas 1 and 3.
    v = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 2, 0], [2, 0, 0], [4, 0, 0], [2, 3, 0]], dtype=torch.float32, device=DEV)
    f = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=DEV)
    p, n, fid = L.sample_surface(v, f, 200000)
    assert abs((fid == 1).float().mean().item() - 0.75) < 0.005
    assert torch.allclose(n.abs(), torch.tensor([0.0, 0.0, 1.0], device=DEV).expand_as(n))
    # Uniform on a triangle -> sample mean = centroid.
    assert torch.allclose(p[fid == 1].mean(0), v[3:6].mean(0), atol=0.01)
    # Masking excludes faces.
    _, _, fid2 = L.sample_surface(v, f, 1000, face_mask=torch.tensor([True, False], device=DEV))
    assert (fid2 == 0).all()


def test_ray_points_intervals():
    o = torch.zeros(1, 3, device=DEV)
    d = torch.tensor([[0.6, 0.0, 1.0]], device=DEV)      # |d| = 1.166
    D = torch.tensor([3.0], device=DEV)
    x = L.ray_points(o, d, D, "free", 1000, margin=0.1, near=0.2)
    s = x[:, 2]
    assert s.min() >= 0.2 and s.max() <= 3 - 0.1 / d.norm()
    x = L.ray_points(o, d, D, "occ", 1000, margin=0.1, thickness=0.3)
    dist_behind = (x[:, 2] - 3.0) * d.norm()             # true distance along the ray
    assert dist_behind.min() >= 0.1 - 1e-5 and dist_behind.max() <= 0.3 + 1e-5


# ------------------------------------------------------------ terms
def test_L_surf_sphere():
    v, f = icosphere(0.5)
    p, _, _ = L.sample_surface(v, f, 50000)
    assert L.loss_surf(sphere(0.5), p) < 1e-3                       # icosphere chord error only
    assert abs(L.loss_surf(sphere(0.4), p).item() - 0.1) < 1e-3     # field 0.1 too small


def test_L_free_and_L_occ_exact_values_on_one_ray():
    # Ray along +z, surface (rendered depth) at z = 3; field = wall at z = 3 (free side z < 3).
    o = torch.zeros(1, 3, device=DEV)
    d = torch.tensor([[0.0, 0.0, 1.0]], device=DEV)
    D = torch.tensor([3.0], device=DEV)
    m, t, delta = 0.05, 0.2, 0.01
    free = L.ray_points(o, d, D, "free", 100000, margin=m, near=0.0)
    occ = L.ray_points(o, d, D, "occ", 100000, margin=m, thickness=t)
    wall = lambda off: plane((0, 0, -1), -off)                     # noqa: E731  f = off - z
    assert L.loss_free(wall(3.0), free, delta) == 0
    assert L.loss_occ(wall(3.0), occ, delta) == 0
    # Wall too close (z = 2): E[ReLU(delta - (2 - s))], s ~ U(0, 3 - m).
    expect = (3 - m - (2 - delta)) ** 2 / 2 / (3 - m)
    assert abs(L.loss_free(wall(2.0), free, delta).item() - expect) < 1e-4
    # Wall too far (z = 4): f = 4 - z on z in (3.05, 3.2) -> mean = delta + 4 - 3.125.
    assert abs(L.loss_occ(wall(4.0), occ, delta).item() - (delta + 4 - 3.125)) < 1e-4


def test_L_eik():
    x = torch.rand(10000, 3, device=DEV) * 2 - 1
    x = x[x.norm(dim=-1) > 0.2]
    assert L.loss_eik(sphere(0.5), x) < 1e-5
    assert abs(L.loss_eik(sphere(0.5, scale=2.0), x).item() - 1.0) < 1e-3


def test_L_normal():
    v, f = icosphere(0.5)
    p, n, _ = L.sample_surface(v, f, 20000)
    assert L.loss_normal(sphere(0.5), p, n) < 1e-3
    assert L.loss_normal(sphere(0.5), p, -n) < 1e-3                 # unsigned: winding is arbitrary
    x = torch.rand(1000, 3, device=DEV)
    ones = torch.ones(1000, 1, device=DEV)
    assert abs(L.loss_normal(plane((0, 0, 1), 0), x, ones * torch.tensor([1.0, 0, 0], device=DEV)).item() - 1.0) < 1e-4
    n60 = ones * torch.tensor([math.sin(math.pi / 3), 0, math.cos(math.pi / 3)], device=DEV)
    assert abs(L.loss_normal(plane((0, 0, 1), 0), x, n60).item() - 0.5) < 1e-4


def test_L_simple_area_and_curvature():
    torch.manual_seed(0)
    x = torch.rand(4_000_000, 3, device=DEV) * 2 - 1                 # box [-1,1]^3, volume 8
    a_sphere = L.surface_density(sphere(0.5), x, beta=0.01).item() * 8
    a_plane = L.surface_density(plane((0, 0, 1), 0.0), x, beta=0.01).item() * 8
    assert abs(a_sphere - 4 * math.pi * 0.25) / (4 * math.pi * 0.25) < 0.03
    assert abs(a_plane - 4.0) / 4.0 < 0.03
    # Curvature: zero on a plane, larger on the more curved sphere.
    pts = torch.nn.functional.normalize(torch.randn(5000, 3, device=DEV), dim=-1)
    assert L.loss_simple_curvature(plane((0, 0, 1), 0), pts, 0.05) < 1e-6
    c_small = L.loss_simple_curvature(sphere(0.25), 0.25 * pts, 0.05)
    c_big = L.loss_simple_curvature(sphere(1.0), 1.0 * pts, 0.05)
    assert c_small > 4 * c_big > 0


# ------------------------------------------------------------ pipeline
def _wall_scene():
    """Camera at the origin looking +z at a wall z = 3 (one big triangle), exact depth for it."""
    view = FakeView(np.eye(3), np.zeros(3), 1.0, 0.8)
    verts = torch.tensor([[-30.0, -30.0, 3.0], [30.0, -30.0, 3.0], [0.0, 40.0, 3.0]], device=DEV)
    faces = torch.tensor([[0, 1, 2]], device=DEV, dtype=torch.int32)
    H, W = 60, 80
    dep = exact_depth(view, verts, faces, torch.zeros(1, H, W, device=DEV), torch.ones(1, H, W, device=DEV))
    return view, verts, faces, dep


def test_rays_from_depth_hit_the_wall():
    view, verts, faces, dep = _wall_scene()
    o, d, D = L.rays_from_depth(view, dep, 500)
    hit = o + D[:, None] * d
    assert torch.allclose(hit[:, 2], torch.full_like(hit[:, 2], 3.0), atol=1e-4)


def test_training_from_rays_puts_zero_level_on_the_wall():
    """Integration: an SDFField starting as a wrong plane (z = 1.5) is trained with
    L_surf + L_free + L_occ + L_eik + L_normal on the wall scene; its zero
    crossing along the optical axis must move to z = 3."""
    torch.manual_seed(0)
    view, verts, faces, dep = _wall_scene()
    lo, hi = torch.tensor([-2.0, -1.5, 0.5]), torch.tensor([2.0, 1.5, 4.5])
    sdf = SDFField(lo, hi, base_res=16, n_levels=3, init="plane", init_normal=(0, 0, -1), init_offset=-1.5).to(DEV)
    cfg = L.ProxyLossConfig(n_surf=8192, n_rays=2048, n_eik_uniform=2048, n_eik_near=2048, margin=0.05, thickness=0.3)
    opt = torch.optim.Adam(sdf.param_groups(lr_grid=1e-2, lr_mlp=1e-3))
    for it in range(800):
        s = L.build_samples(cfg, sdf, verts, faces, view, dep)
        total, terms = L.proxy_losses(cfg, sdf, s)
        opt.zero_grad()
        total.backward()
        opt.step()
    z = torch.linspace(0.6, 4.4, 3801, device=DEV)
    f = sdf(torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], -1)).detach()
    sign_change = z[1:][(f[:-1] > 0) & (f[1:] <= 0)]
    print("zero crossings on axis:", sign_change.tolist(), {k: round(v.item(), 5) for k, v in terms.items()})
    assert sign_change.numel() >= 1 and abs(sign_change[0].item() - 3.0) < 0.03
    assert set(terms) == {"L_surf", "L_free", "L_occ", "L_eik", "L_normal"}


def test_weights_switch_terms_off():
    view, verts, faces, dep = _wall_scene()
    sdf = SDFField(torch.tensor([-2.0, -1.5, 0.5]), torch.tensor([2.0, 1.5, 4.5]), init="plane").to(DEV)
    cfg = L.ProxyLossConfig(w_free=0, w_occ=0, w_normal=0, w_simple=1.0, simple_mode="area", n_surf=1024)
    s = L.build_samples(cfg, sdf, verts, faces, view, dep)
    assert "free" not in s and "occ" not in s
    _, terms = L.proxy_losses(cfg, sdf, s)
    assert set(terms) == {"L_surf", "L_eik", "L_simple"}


def test_free_band_samples_lie_in_the_band():
    view, verts, faces, dep = _wall_scene()
    sdf = SDFField(torch.tensor([-2.0, -1.5, 0.5]), torch.tensor([2.0, 1.5, 4.5]), init="plane").to(DEV)
    cfg = L.ProxyLossConfig(w_occ=0, w_eik=0, w_normal=0, w_surf=0, n_rays=500, n_per_ray=8,
                            free_band_frac=1.0, free_band=0.3, margin=0.05)
    x = L.build_samples(cfg, sdf, verts, faces, view, dep)["free"]
    # Wall at z = 3, so the true distance before the wall along each ray is (3 - z) * |d| / d_z = (3 - z) * |d|.
    dist = (3.0 - x[:, 2]) * (x / x[:, 2:3]).norm(dim=-1)
    assert dist.min() >= 0.05 - 1e-4 and dist.max() <= 0.3 + 1e-4
    cfg.free_band_frac = 0.5
    x = L.build_samples(cfg, sdf, verts, faces, view, dep)["free"]
    dist = (3.0 - x[:, 2]) * (x / x[:, 2:3]).norm(dim=-1)
    assert 0.4 < (dist <= 0.3).float().mean() < 0.65


def test_no_triangle_in_box_is_handled():
    sdf = SDFField(torch.tensor([-1.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0]), init="sphere").to(DEV)
    v = torch.tensor([[5.0, 5, 5], [6, 5, 5], [5, 6, 5]], device=DEV)          # far outside the box
    f = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEV)
    p, n, fid = L.sample_surface(v, f, 100, face_mask=torch.tensor([False], device=DEV))
    assert p.shape == (0, 3) and fid.numel() == 0
    cfg = L.ProxyLossConfig(n_surf=256, n_eik_uniform=256)
    s = L.build_samples(cfg, sdf, v, f)
    assert s["surf"].shape[0] == 0
    total, terms = L.proxy_losses(cfg, sdf, s)
    assert torch.isfinite(total) and terms["L_surf"] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
