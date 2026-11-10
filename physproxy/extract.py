"""Zero level set of an SDFField -> triangle mesh (pytorch3d marching cubes).

The field is sampled on a lattice over the (oriented) proxy box with about
cubic cells (`res` cells along the longest axis). The lattice is padded by one
layer of "outside" values, so the surface is closed where it meets the box
walls (e.g. the ground plane becomes a capped slab): the result is watertight
by construction. Vertices are returned in world coordinates, scene units.
"""
import torch
from pytorch3d.ops.marching_cubes import marching_cubes


@torch.no_grad()
def sdf_lattice(sdf, res=256, chunk=2 ** 20):
    lo, hi = sdf.aabb_min, sdf.aabb_max
    ext = hi - lo
    n = [max(2, int(round(res * float(e / ext.max())))) for e in ext]          # (nx, ny, nz)
    axes = [torch.linspace(float(lo[i]), float(hi[i]), n[i], device=lo.device) for i in range(3)]
    zz, yy, xx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")    # [nz, ny, nx]
    pts_local = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
    vals = torch.cat([sdf(sdf.to_world(p)) for p in pts_local.split(chunk)])
    return vals.reshape(n[2], n[1], n[0]), axes


@torch.no_grad()
def extract_mesh(sdf, res=256, pad_value=None):
    """Returns (verts [V,3] world, faces [F,3] long) of the f = 0 surface, closed at the box."""
    vol, axes = sdf_lattice(sdf, res)
    if pad_value is None:
        pad_value = float(sdf.voxel_size.max())
    vol = torch.nn.functional.pad(vol, (1, 1, 1, 1, 1, 1), value=pad_value)
    verts, faces = marching_cubes(vol[None], 0.0, return_local_coords=False)
    v, f = verts[0], faces[0].long()
    if v.numel() == 0:
        return v, f
    # Lattice index (x,y,z order, shifted by the padding) -> local coordinates -> world.
    lo = torch.stack([a[0] for a in axes])
    step = torch.stack([a[1] - a[0] for a in axes])
    v_local = lo + (v - 1.0) * step
    # Clamp the padding layer back onto the box walls (caps lie on the walls).
    v_local = torch.max(torch.min(v_local, sdf.aabb_max), sdf.aabb_min)
    # pytorch3d orients faces outward (towards f > 0) for negative-inside fields
    # (checked: a sphere gives positive signed volume).
    return sdf.to_world(v_local), f
