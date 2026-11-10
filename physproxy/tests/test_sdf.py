"""Phase 2 checks for physproxy/sdf.py."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physproxy.sdf import SDFField, aabb_from_points  # noqa: E402

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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
