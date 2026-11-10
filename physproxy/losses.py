"""Losses tying the SDF proxy to the rendered triangles (Phase 3).

All distances are in scene units. `sdf` is any object with `sdf(x) -> f` and
`sdf.gradient(x) -> grad f` (SDFField, or an analytic field in the tests).

Samples
  sample_surface      area-weighted points on (opaque) render triangles + their unit normals
  rays_from_depth     random valid pixels of an exact-depth map -> world rays (z-depth param)
  ray_points          stratified points along rays: "free" (camera .. D - margin) or
                      "occ" (D + margin .. D + thickness)
  near_surface        surface samples jittered by N(0, sigma) (for the eikonal term)

Terms (each switchable by a weight of 0 in ProxyLossConfig)
  L_surf   = mean |f(x_surf)|
  L_free   = mean ReLU(delta - f(x_free))         free space must be outside
  L_occ    = mean ReLU(delta + f(x_occ))          just behind the surface is inside
  L_eik    = mean (|grad f| - 1)^2                uniform box points + near-surface points
  L_normal = mean 1 - |cos(n_tri, grad f/|grad f|)|   (unsigned: triangle winding is arbitrary)
  L_simple = "curvature": mean ||n(x) - n(x + eps*u)||^2 at near-surface points (smoothness), or
             "area": mean_{x ~ U(box)} exp(-|f|/beta) / (2 beta)  (= surface area / box volume
             as beta -> 0; penalizes extra surface, keeps the proxy coarse)

Ray parametrization: a ray is o + s * d with d = R_c2w (x/z, y/z, 1), so s is the
view z-depth, matching physproxy.depth. Margins are true distances along the
ray, converted with |d| (s_margin = margin / |d|).

Caveat (why delta is small): a free-space point is only guaranteed f > 0. Its
distance to the nearest surface can be much smaller than its distance along the
ray (a ray grazing an object), so delta must not be large.
"""
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

from physproxy.depth import view_rays


# ---------------------------------------------------------------- sampling
def sample_surface(vertices, faces, n, face_mask=None):
    """Area-weighted uniform points on triangles.

    Returns (points [n,3], unit normals [n,3], face ids [n] into `faces`).
    Points are differentiable w.r.t. vertices (fixed barycentrics); detach for
    one-way coupling.
    """
    fidx = torch.arange(faces.shape[0], device=faces.device)
    if face_mask is not None:
        fidx = fidx[face_mask]
    if fidx.numel() == 0 or n == 0:
        z = torch.zeros(0, 3, device=vertices.device)
        return z, z, fidx[:0]
    tri = vertices[faces[fidx].long()]                       # [T,3,3]
    cr = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    area = 0.5 * cr.norm(dim=-1)
    pick = torch.multinomial(area.detach().clamp_min(0) + 1e-20, n, replacement=True)
    r1 = torch.rand(n, 1, device=vertices.device).sqrt()
    r2 = torch.rand(n, 1, device=vertices.device)
    t = tri[pick]
    pts = (1 - r1) * t[:, 0] + r1 * (1 - r2) * t[:, 1] + r1 * r2 * t[:, 2]
    nrm = F.normalize(cr[pick], dim=-1)
    return pts, nrm, fidx[pick]


def rays_from_depth(view, depth_out, n_rays):
    """Random valid pixels of `exact_depth` output -> (origins [N,3], dirs [N,3], D [N])."""
    valid = depth_out["valid"]
    H, W = valid.shape
    idx = valid.reshape(-1).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        z = torch.zeros(0, 3, device=valid.device)
        return z, z, torch.zeros(0, device=valid.device)
    pick = idx[torch.randint(idx.numel(), (min(n_rays, idx.numel()),), device=idx.device)]
    j, i = pick // W, pick % W
    d_view = view_rays(view, H, W, valid.device)[j, i]
    c2w = torch.inverse(view.world_view_transform.T)
    d = d_view @ c2w[:3, :3].T
    o = c2w[:3, 3].expand_as(d)
    return o, d, depth_out["depth"][j, i].detach()


def ray_points(o, d, D, mode, n_per_ray, margin, thickness=None, near=0.0, stratified=True):
    """Points along rays (s = view z-depth). mode "free": s in (near, D - margin/|d|);
    "occ": s in (D + margin/|d|, D + thickness/|d|). Rays whose interval is empty are dropped.
    Returns [M,3]."""
    dn = d.norm(dim=-1)
    if mode == "free":
        lo = torch.full_like(D, float(near))
        hi = D - margin / dn
    elif mode == "occ":
        assert thickness is not None and thickness > margin
        lo = D + margin / dn
        hi = D + thickness / dn
    else:
        raise ValueError(mode)
    keep = hi > lo
    o, d, lo, hi = o[keep], d[keep], lo[keep], hi[keep]
    if o.shape[0] == 0:
        return torch.zeros(0, 3, device=d.device)
    k = torch.arange(n_per_ray, device=d.device, dtype=d.dtype)
    u = (k + (torch.rand(o.shape[0], n_per_ray, device=d.device) if stratified else 0.5)) / n_per_ray
    s = lo[:, None] + (hi - lo)[:, None] * u
    return (o[:, None] + s[..., None] * d[:, None]).reshape(-1, 3)


def near_surface(points, sigma):
    return points + sigma * torch.randn_like(points)


# ---------------------------------------------------------------- terms
def loss_surf(sdf, x):
    return sdf(x).abs().mean() if x.numel() else x.sum() * 0


def loss_free(sdf, x, delta):
    return F.relu(delta - sdf(x)).mean() if x.numel() else x.sum() * 0


def loss_occ(sdf, x, delta):
    return F.relu(delta + sdf(x)).mean() if x.numel() else x.sum() * 0


def loss_eik(sdf, x):
    if not x.numel():
        return x.sum() * 0
    return ((sdf.gradient(x).norm(dim=-1) - 1) ** 2).mean()


def loss_normal(sdf, x, normals, eps=1e-8):
    if not x.numel():
        return x.sum() * 0
    g = sdf.gradient(x)
    cos = (g * normals).sum(-1) / (g.norm(dim=-1) * normals.norm(dim=-1)).clamp_min(eps)
    return (1 - cos.abs()).mean()


def loss_simple_curvature(sdf, x, eps):
    """Mean squared change of the unit normal over a random step of length eps."""
    if not x.numel():
        return x.sum() * 0
    u = F.normalize(torch.randn_like(x), dim=-1)
    n0 = F.normalize(sdf.gradient(x), dim=-1)
    n1 = F.normalize(sdf.gradient(x + eps * u), dim=-1)
    return ((n0 - n1) ** 2).sum(-1).mean()


def surface_density(sdf, x, beta):
    """Laplace density of the zero level set: E_{x~U(box)}[.] = area / volume as beta -> 0."""
    return (torch.exp(-sdf(x).abs() / beta) / (2 * beta)).mean()


# ---------------------------------------------------------------- config + combination
@dataclass
class ProxyLossConfig:
    w_surf: float = 1.0
    w_free: float = 1.0
    w_occ: float = 0.5
    w_eik: float = 0.1
    w_normal: float = 0.05
    w_simple: float = 0.0
    simple_mode: str = "curvature"   # or "area"
    margin: float = 0.05             # scene units, gap left free around the rendered surface
    thickness: float = 0.2           # scene units, inside band behind the surface (L_occ)
    delta: float = 0.02              # hinge margin on f for L_free / L_occ
    near: float = 0.2                # start of free space along a ray (z), = rasterizer near cull
    n_surf: int = 32768
    n_rays: int = 4096
    n_per_ray: int = 16
    n_eik_uniform: int = 8192
    n_eik_near: int = 8192
    eik_sigma: float = 0.05          # near-surface jitter, scene units
    curv_eps: float = 0.05           # curvature step, scene units
    area_beta: float = 0.02          # area density width, scene units
    n_area: int = 32768

    def as_dict(self):
        return asdict(self)


def build_samples(cfg, sdf, vertices, faces, view=None, depth_out=None, face_mask=None):
    """All sample sets one step needs. Ray-based sets need (view, depth_out).
    Every set is restricted to the proxy box (outside it the grid is clamped)."""
    s = {}
    need_surf = cfg.w_surf > 0 or cfg.w_normal > 0 or cfg.w_eik > 0 or (cfg.w_simple > 0 and cfg.simple_mode == "curvature")
    if need_surf:
        # Only triangles that overlap the box (their local-frame bounds intersect
        # it): big background triangles (sky, far trees) otherwise take most of
        # the area-weighted samples. Samples are filtered to the box below.
        tl = sdf.to_local(vertices.detach())[faces.long()]          # [T,3,3]
        in_box = ((tl.max(1).values >= sdf.aabb_min) & (tl.min(1).values <= sdf.aabb_max)).all(-1)
        fm = in_box if face_mask is None else (in_box & face_mask)
        p, n, _ = sample_surface(vertices, faces, cfg.n_surf, fm)
        m = sdf.inside_aabb(p)
        s["surf"], s["surf_n"] = p[m], n[m]
    if view is not None and depth_out is not None and (cfg.w_free > 0 or cfg.w_occ > 0):
        o, d, D = rays_from_depth(view, depth_out, cfg.n_rays)
        if cfg.w_free > 0:
            x = ray_points(o, d, D, "free", cfg.n_per_ray, cfg.margin, near=cfg.near)
            s["free"] = x[sdf.inside_aabb(x)]
        if cfg.w_occ > 0:
            x = ray_points(o, d, D, "occ", max(2, cfg.n_per_ray // 4), cfg.margin, cfg.thickness)
            s["occ"] = x[sdf.inside_aabb(x)]
    if cfg.w_eik > 0 or cfg.w_simple > 0:
        near = near_surface(s["surf"][: cfg.n_eik_near], cfg.eik_sigma) if "surf" in s else None
        if near is not None:
            near = near[sdf.inside_aabb(near)]
        s["eik"] = torch.cat([sdf.sample_uniform(cfg.n_eik_uniform)] + ([near] if near is not None else []))
        s["near"] = near
    if cfg.w_simple > 0 and cfg.simple_mode == "area":
        s["area"] = sdf.sample_uniform(cfg.n_area)
    return s


def proxy_losses(cfg, sdf, samples):
    """Weighted terms and their sum. Returns (total, dict of unweighted terms)."""
    terms = {}
    if cfg.w_surf > 0 and "surf" in samples:
        terms["L_surf"] = loss_surf(sdf, samples["surf"])
    if cfg.w_free > 0 and "free" in samples:
        terms["L_free"] = loss_free(sdf, samples["free"], cfg.delta)
    if cfg.w_occ > 0 and "occ" in samples:
        terms["L_occ"] = loss_occ(sdf, samples["occ"], cfg.delta)
    if cfg.w_eik > 0 and "eik" in samples:
        terms["L_eik"] = loss_eik(sdf, samples["eik"])
    if cfg.w_normal > 0 and "surf" in samples:
        terms["L_normal"] = loss_normal(sdf, samples["surf"], samples["surf_n"])
    if cfg.w_simple > 0:
        if cfg.simple_mode == "curvature" and samples.get("near") is not None:
            terms["L_simple"] = loss_simple_curvature(sdf, samples["near"], cfg.curv_eps)
        elif cfg.simple_mode == "area" and "area" in samples:
            terms["L_simple"] = surface_density(sdf, samples["area"], cfg.area_beta)
    w = {"L_surf": cfg.w_surf, "L_free": cfg.w_free, "L_occ": cfg.w_occ, "L_eik": cfg.w_eik,
         "L_normal": cfg.w_normal, "L_simple": cfg.w_simple}
    total = sum(w[k] * v for k, v in terms.items()) if terms else torch.zeros((), device=sdf.aabb_min.device)
    return total, terms
