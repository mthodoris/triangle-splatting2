"""Independent exact z-buffer of a triangle mesh in an upstream camera (pytorch3d)."""
import math

import torch

from physproxy.depth import world_to_view


def p3d_zbuffer(view, vertices, faces, H, W, znear=0.2):
    """Independent exact z-buffer with pytorch3d. Returns (zbuf [H,W], face_id [H,W])."""
    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    from pytorch3d.structures import Meshes
    pv = world_to_view(view, vertices)
    keep = (pv[faces.long()][..., 2] > znear).all(dim=1)
    fid = torch.nonzero(keep, as_tuple=True)[0]
    f = faces[fid].long()
    tx, ty = math.tan(view.FoVx / 2), math.tan(view.FoVy / 2)
    z = pv[:, 2].clamp_min(1e-6)
    # pytorch3d NDC: +X left, +Y up, the shorter image side spans [-1, 1].
    s_x = W / min(H, W)
    s_y = H / min(H, W)
    v_ndc = torch.stack([-(pv[:, 0] / z / tx) * s_x, -(pv[:, 1] / z / ty) * s_y, pv[:, 2]], dim=-1)
    mesh = Meshes(verts=[v_ndc], faces=[f])
    p2f, zbuf, _, _ = rasterize_meshes(mesh, image_size=(H, W), blur_radius=0.0, faces_per_pixel=1,
                                       perspective_correct=True, bin_size=1 << max(6, math.ceil(math.log2(max(H, W) / 22))),
                                       max_faces_per_bin=400000,
                                       cull_backfaces=False)
    p2f = p2f[0, ..., 0]
    zb = zbuf[0, ..., 0]
    face = torch.where(p2f >= 0, fid[p2f.clamp_min(0)], torch.full_like(p2f, -1))
    return zb, face
