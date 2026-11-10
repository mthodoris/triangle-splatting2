"""Region of interest for the collision proxy: a gravity-aligned (oriented) box.

COLMAP frames are arbitrary: on garden the mean camera up is about
(0, -0.88, -0.47), so a world-axis box fits badly. Here
  * z_local = mean camera up (cameras' -y axis in world, COLMAP/OpenCV view
    convention), or the normal of a plane fitted to the camera centres when the
    up vectors are not given (sign: towards the cameras' up if known),
  * x_local / y_local = principal axes of the region's points in the plane
    perpendicular to z (x = major axis), right-handed,
  * box = robust per-axis percentiles of the SfM points within radius_mult x
    the largest camera distance from the camera centroid, padded.
All in scene units. `frame` rows are the local axes in world coordinates:
x_local = frame @ x_world.
"""
import torch


def camera_up_vectors(views):
    """World-space up (= -y of the view frame) for upstream Camera objects, [N,3]."""
    ups = []
    for v in views:
        c2w = torch.inverse(v.world_view_transform.T)
        ups.append(-c2w[:3, 1])
    return torch.stack(ups).float().cpu()


def camera_centers(views):
    return torch.stack([v.camera_center for v in views]).float().cpu()


def up_frame(up_vectors=None, points=None, cam_centers=None):
    """Rotation [3,3] (rows = local x, y, z in world) with z = up."""
    if up_vectors is not None:
        z = torch.as_tensor(up_vectors, dtype=torch.float32).mean(0)
    else:
        c = torch.as_tensor(cam_centers, dtype=torch.float32)
        c = c - c.mean(0)
        z = torch.linalg.svd(c, full_matrices=False).Vh[-1]   # normal of the camera "ring"
    z = z / z.norm()
    x = None
    if points is not None and len(points) >= 3:
        p = torch.as_tensor(points, dtype=torch.float32)
        p = p - p.mean(0)
        p = p - (p @ z)[:, None] * z[None]
        cov = p.T @ p
        evals, evecs = torch.linalg.eigh(cov)
        x = evecs[:, -1]
        x = x - (x @ z) * z
        if x.norm() < 1e-6:
            x = None
    if x is None:  # any vector perpendicular to z
        a = torch.tensor([1.0, 0.0, 0.0]) if abs(float(z[0])) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
        x = a - (a @ z) * z
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    return torch.stack([x, y, z])


def proxy_box(points, cam_centers, frame=None, pct=(2.0, 98.0), radius_mult=1.5, pad=0.05):
    """Box for the proxy in the given frame (default identity).

    Returns (lo [3], hi [3]) in LOCAL coordinates. Points farther than
    radius_mult * (max camera distance) from the camera centroid are dropped
    (sky / far background) before the percentiles.
    """
    points = torch.as_tensor(points, dtype=torch.float32)
    cams = torch.as_tensor(cam_centers, dtype=torch.float32)
    frame = torch.eye(3) if frame is None else torch.as_tensor(frame, dtype=torch.float32)
    c = cams.mean(0)
    r = (cams - c).norm(dim=-1).max() * radius_mult
    p = points[(points - c).norm(dim=-1) <= r] @ frame.T
    lo = torch.quantile(p, pct[0] / 100, dim=0)
    hi = torch.quantile(p, pct[1] / 100, dim=0)
    ext = hi - lo
    return lo - pad * ext, hi + pad * ext


def oriented_proxy_box(points, cam_centers, up_vectors=None, pct=(2.0, 98.0), radius_mult=1.5, pad=0.05,
                       top_to_cameras=True):
    """(frame, lo, hi): a gravity-aligned box; see the module docstring.

    top_to_cameras: raise the box top to at least the highest camera, so
    vertical structures (trunks, walls, the top of objects) are not cut by the
    upper z percentile, which SfM's sparse sampling of tall things pulls down.
    """
    points = torch.as_tensor(points, dtype=torch.float32)
    cams = torch.as_tensor(cam_centers, dtype=torch.float32)
    c = cams.mean(0)
    r = (cams - c).norm(dim=-1).max() * radius_mult
    near = points[(points - c).norm(dim=-1) <= r]
    frame = up_frame(up_vectors, near, cams)
    lo, hi = proxy_box(points, cams, frame, pct, radius_mult, pad)
    if top_to_cameras:
        hi = hi.clone()
        hi[2] = torch.maximum(hi[2], (cams @ frame.T)[:, 2].max())
    return frame, lo, hi


def aabb_from_points(points, cam_centers, pct=(2.0, 98.0), radius_mult=1.5, pad=0.05):
    """World axis-aligned box (identity frame)."""
    return proxy_box(points, cam_centers, None, pct, radius_mult, pad)
