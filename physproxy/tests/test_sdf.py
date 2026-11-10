"""Phase 2 checks for physproxy/sdf.py."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physproxy.sdf import SDFField  # noqa: E402
from physproxy.aabb import aabb_from_points, up_frame, oriented_proxy_box  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LO, HI = torch.tensor([-1.0, -0.5, -1.5]), torch.tensor([1.0, 1.5, 0.5])   # non-cubic, off-origin


def _rand_in_box(n, lo=LO, hi=HI, margin=0.0):
    lo, hi = lo.to(DEV) + margin, hi.to(DEV) - margin
    return lo + (hi - lo) * torch.rand(n, 3, device=DEV)


def test_shapes_and_init_is_prior():
    sdf = SDFField(LO, HI, init="sphere").to(DEV)
    x = _rand_in_box(1000).reshape(10, 100, 3)
    f = sdf(x)
    g = sdf.gradient(x)
    assert f.shape == (10, 100) and g.shape == (10, 100, 3)
    assert torch.allclose(f, sdf.prior(x), atol=1e-6)   # zero-initialized last layer
    assert sdf.resolutions == [16, 32, 64, 128]


def test_fd_gradient_on_analytic_sphere():
    sdf = SDFField(LO, HI, init="sphere", init_radius=0.5).to(DEV)
    x = _rand_in_box(5000)
    d = x - sdf.init_center
    x = x[d.norm(dim=-1) > 0.25]             # FD error ~ (h/r)^2: stay away from the centre
    g = sdf.gradient(x)
    g_true = torch.nn.functional.normalize(x - sdf.init_center, dim=-1)
    assert (g - g_true).norm(dim=-1).max() < 1e-3
    assert ((g.norm(dim=-1) - 1).abs()).max() < 1e-3


def test_fd_matches_autograd_first_derivative():
    # Make the residual non-trivial, then compare FD with autograd's first
    # derivative (single backward through grid_sample does work).
    torch.manual_seed(0)
    sdf = SDFField(LO, HI, init="plane", grid_init_std=0.3).to(DEV)
    with torch.no_grad():
        sdf.mlp[-1].weight.normal_(0, 0.5)
    x = _rand_in_box(4000, margin=0.05).requires_grad_(True)
    g_ad, = torch.autograd.grad(sdf(x).sum(), x)
    g_fd = sdf.gradient(x.detach(), h=1e-4)   # random grid is very rough: converge as h -> 0
    rel = (g_fd - g_ad).norm(dim=-1) / g_ad.norm(dim=-1).clamp_min(1e-3)
    # Trilinear features have kinks at voxel faces, so a few samples straddle one.
    assert rel.median() < 1e-2 and (rel < 0.05).float().mean() > 0.95


def test_fd_gradient_trains_parameters():
    sdf = SDFField(LO, HI, init="plane").to(DEV)
    with torch.no_grad():
        sdf.mlp[-1].weight.normal_(0, 0.1)
    g = sdf.gradient(_rand_in_box(256))
    ((g.norm(dim=-1) - 1) ** 2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in sdf.grids)


def test_fit_sphere_from_samples():
    torch.manual_seed(0)
    c, r = torch.tensor([0.2, 0.6, -0.4], device=DEV), 0.45
    target = lambda x: (x - c).norm(dim=-1) - r     # noqa: E731
    sdf = SDFField(LO, HI, init="plane", init_normal=(0, 1, 0), base_res=16, n_levels=3).to(DEV)
    opt = torch.optim.Adam(sdf.param_groups(lr_grid=1e-2, lr_mlp=1e-3))
    t0 = time.time()
    for it in range(1500):
        x = _rand_in_box(8192)
        s = torch.randn(4096, 3, device=DEV)
        xs = c + r * torch.nn.functional.normalize(s, dim=-1) + 0.02 * torch.randn(4096, 1, device=DEV) * torch.nn.functional.normalize(s, dim=-1)
        xa = torch.cat([x, xs])
        loss = (sdf(xa) - target(xa)).abs().mean()
        g = sdf.gradient(x[:2048])
        loss = loss + 0.1 * ((g.norm(dim=-1) - 1) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        xe = _rand_in_box(20000)
        mae = (sdf(xe) - target(xe)).abs().mean().item()
        surf = c + r * torch.nn.functional.normalize(torch.randn(5000, 3, device=DEV), dim=-1)
        surf_err = sdf(surf).abs().mean().item()
        eik = (sdf.gradient(xe).norm(dim=-1) - 1).abs().mean().item()
    print(f"fit sphere: {time.time() - t0:.1f}s mae={mae:.4f} |f| on surface={surf_err:.4f} ||grad|-1|={eik:.4f}")
    # The box is 2 units wide; 1 % of that is 0.02.
    assert mae < 0.02 and surf_err < 0.01 and eik < 0.1


def test_aabb_from_points_drops_far_background():
    torch.manual_seed(0)
    cams = torch.randn(50, 3) * 0.1 + torch.tensor([0.0, 0.0, 4.0])
    cams = torch.nn.functional.normalize(cams - torch.tensor([0, 0, 4.0]), dim=-1) * 3   # ring of radius 3 around 0
    near = torch.rand(10000, 3) * 2 - 1                                                  # scene in [-1,1]^3
    far = torch.randn(500, 3) * 50                                                       # background / sky
    lo, hi = aabb_from_points(torch.cat([near, far]), cams, pct=(1, 99), radius_mult=1.5, pad=0.0)
    assert (lo > -1.3).all() and (hi < 1.3).all() and (lo < -0.9).all() and (hi > 0.9).all()


def _rotation(seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    return q * torch.sign(torch.det(q))            # proper rotation


def test_up_frame_recovers_tilted_up_and_box():
    # Synthetic scene in a "gravity" frame: a flat 6x3 slab, cameras on a ring
    # above it; then everything is rotated by a random R into the world.
    torch.manual_seed(0)
    Rw = _rotation(3)                               # local -> world
    pts_l = torch.rand(20000, 3) * torch.tensor([6.0, 3.0, 1.0]) - torch.tensor([3.0, 1.5, 0.0])
    ang = torch.linspace(0, 6.28, 40)
    cams_l = torch.stack([4 * ang.cos(), 4 * ang.sin(), torch.full_like(ang, 1.5)], -1)
    up_l = torch.tensor([0.0, 0.0, 1.0]).expand(40, 3) + 0.05 * torch.randn(40, 3)
    to_w = lambda a: a @ Rw.T                      # noqa: E731
    frame, lo, hi = oriented_proxy_box(to_w(pts_l), to_w(cams_l), to_w(up_l), pct=(0, 100), pad=0.0,
                                       top_to_cameras=False)
    assert torch.allclose(frame @ frame.T, torch.eye(3), atol=1e-5) and torch.det(frame) > 0
    assert (frame[2] @ Rw[:, 2]) > 0.99                         # z = up
    assert abs(frame[0] @ Rw[:, 0]) > 0.99                      # x = long horizontal axis
    ext = hi - lo
    assert torch.allclose(ext, torch.tensor([6.0, 3.0, 1.0]), atol=0.1)   # noisy up: ~0.5 deg tilt
    # Default: the top is raised to the highest camera (z = 1.5 above the slab bottom at 0).
    _, lo2, hi2 = oriented_proxy_box(to_w(pts_l), to_w(cams_l), to_w(up_l), pct=(0, 100), pad=0.0)
    assert abs((hi2 - lo2)[2].item() - 1.5) < 0.1
    # Without up vectors: the ring normal of the camera centres gives the same z (up to sign).
    f2 = up_frame(None, None, to_w(cams_l))
    assert abs(f2[2] @ Rw[:, 2]) > 0.99


def test_oriented_field_box_mapping_and_isotropic_grid():
    R = _rotation(1)
    lo, hi = torch.tensor([-2.0, -1.0, -0.5]), torch.tensor([2.0, 1.0, 0.5])
    sdf = SDFField(lo, hi, frame=R, base_res=16, n_levels=3, init="plane").to(DEV)
    assert sdf.grid_shapes == [(16, 8, 4), (32, 16, 8), (64, 32, 16)]
    assert sdf.grids[-1].shape[2:] == (16, 32, 64)              # [Rz, Ry, Rx]
    vs = sdf.voxel_size.cpu()
    assert vs.max() / vs.min() < 1.1                            # about cubic voxels
    corners = torch.tensor([[a, b, c] for a in (-2.0, 2.0) for b in (-1.0, 1.0) for c in (-0.5, 0.5)])
    xn = sdf.normalize(sdf.to_world(corners.to(DEV)))
    assert torch.allclose(xn.abs(), torch.ones_like(xn), atol=1e-5)
    x = sdf.sample_uniform(5000)
    assert sdf.inside_aabb(x).all()
    assert torch.allclose(sdf.init_normal.cpu(), R[2], atol=1e-6)   # plane prior normal = local up
    # Prior is defined in world units: f(x) = n.x for a world point.
    assert torch.allclose(sdf(x), x @ sdf.init_normal, atol=1e-5)


def test_fit_sphere_in_rotated_box():
    torch.manual_seed(0)
    R = _rotation(7).to(DEV)
    lo, hi = torch.tensor([-1.0, -0.6, -0.4]), torch.tensor([1.0, 0.6, 0.4])
    sdf = SDFField(lo, hi, frame=R.cpu(), base_res=32, n_levels=3, init="plane").to(DEV)
    c, r = sdf.to_world(torch.tensor([0.2, 0.0, 0.0], device=DEV)), 0.3
    target = lambda x: (x - c).norm(dim=-1) - r      # noqa: E731
    opt = torch.optim.Adam(sdf.param_groups(lr_grid=1e-2, lr_mlp=1e-3))
    for it in range(1500):
        x = sdf.sample_uniform(8192)
        loss = (sdf(x) - target(x)).abs().mean() + 0.1 * ((sdf.gradient(x[:2048]).norm(dim=-1) - 1) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        surf = c + r * torch.nn.functional.normalize(torch.randn(5000, 3, device=DEV), dim=-1)
        err = sdf(surf).abs().mean().item()
    print(f"rotated-box sphere fit: |f| on surface={err:.4f}")
    assert err < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
