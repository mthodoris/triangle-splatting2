#
# Multi-view photometric consistency (devel6, --lambda_mv).
#
# The image loss lets per-vertex color absorb geometric error (a smooth sheet over the leaves, painted with leaf
# detail). This loss can only be lowered by moving the surface: patches of the reference GT image are lifted to 3D
# with the rendered mesh depth and projected into neighboring training views, and their GT patches must match
# (normalized cross-correlation). A wrong depth makes the patches disagree whatever color the mesh has.
#   - Each patch pixel is lifted with its own rendered depth, so the gradient reaches the depth of every pixel.
#   - Per patch the best neighbor counts (min of 1 - NCC over the neighbors that see it), which tolerates occlusion.
#   - Only patches on the mesh (alpha > 0.9) with some texture (gray std above min_std) are used.
# Pixel convention as in the rasterizer: pixel i has ndc (2i + 1) / W - 1, so grid_sample (align_corners=False)
# takes the ndc directly.
#

import math
import torch
import torch.nn.functional as F


def _gray(img):
    return (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2])[None, None]  # [1,1,H,W]


class MultiViewConsistency:
    def __init__(self, cameras, num_neighbors=4, max_angle=45.0, num_patches=2048, patch_size=7, dilation=2, min_std=0.02):
        self.cameras = cameras
        centers = torch.stack([c.camera_center for c in cameras])
        dirs = torch.stack([c.world_view_transform[:3, 2] for c in cameras])  # view z axis in world (row of W^T)
        dirs = F.normalize(dirs, dim=1)
        dist = torch.cdist(centers, centers)
        cos = dirs @ dirs.T
        dist[cos < math.cos(math.radians(max_angle))] = float("inf")
        dist.fill_diagonal_(float("inf"))
        k = min(num_neighbors, len(cameras) - 1)
        d, idx = dist.topk(k, dim=1, largest=False)
        self.neighbors = {c.image_name: [cameras[j] for j, dj in zip(idx[i].tolist(), d[i].tolist()) if math.isfinite(dj)]
                          for i, c in enumerate(cameras)}
        self.num_patches = num_patches
        r = (patch_size // 2) * dilation
        o = torch.arange(-r, r + 1, dilation, device="cuda")
        oy, ox = torch.meshgrid(o, o, indexing="ij")
        self.offsets = torch.stack([ox.reshape(-1), oy.reshape(-1)], 1)  # [P,2] (x, y)
        self.radius = r
        self.min_std = min_std
        self.gray = {}

    def gray_of(self, cam):
        g = self.gray.get(cam.image_name)
        if g is None:
            g = self.gray[cam.image_name] = _gray(cam.original_image.cuda())
        return g

    @staticmethod
    def _ncc(a, b, eps=1e-4):
        a = a - a.mean(-1, keepdim=True)
        b = b - b.mean(-1, keepdim=True)
        return (a * b).sum(-1) / torch.sqrt((a * a).sum(-1) * (b * b).sum(-1) + eps)

    def loss(self, cam, depth, alpha):
        """cam: reference camera; depth, alpha: [1,H,W] rendered mesh depth (view z) and alpha. Returns
        (loss, number of patches used)."""
        nbrs = self.neighbors.get(cam.image_name, [])
        zero = depth.sum() * 0
        if not nbrs:
            return zero, 0
        H, W = depth.shape[-2:]
        r = self.radius
        ref = self.gray_of(cam)
        # patch centers on the mesh, away from the border
        ok = alpha[0, r:H - r, r:W - r] > 0.9
        cand = torch.nonzero(ok, as_tuple=False)
        if cand.shape[0] == 0:
            return zero, 0
        cand = cand[torch.randint(0, cand.shape[0], (self.num_patches,), device="cuda")] + r  # [N,2] (y, x)
        px = cand[:, None, 1] + self.offsets[None, :, 0]  # [N,P]
        py = cand[:, None, 0] + self.offsets[None, :, 1]
        ref_patch = ref[0, 0, py, px]  # [N,P]
        textured = ref_patch.std(-1) > self.min_std
        z = depth[0, py, px]  # [N,P], differentiable
        valid_ref = textured & (z > 0).all(-1) & (alpha[0, py, px] > 0.5).all(-1)
        if not valid_ref.any():
            return zero, 0
        ref_patch, z, px, py = ref_patch[valid_ref], z[valid_ref], px[valid_ref], py[valid_ref]

        # lift to world: view x = ndc_x * tan(fovx / 2) * z (the projection is symmetric)
        ndc_x = (2 * px.float() + 1) / W - 1
        ndc_y = (2 * py.float() + 1) / H - 1
        xv = torch.stack([ndc_x * math.tan(cam.FoVx * 0.5) * z, ndc_y * math.tan(cam.FoVy * 0.5) * z, z, torch.ones_like(z)], -1)
        xw = xv @ torch.inverse(cam.world_view_transform)  # row vectors: [x_w, 1] @ W = [x_v, 1]

        costs = []
        for nb in nbrs:
            xn = xw @ nb.world_view_transform  # [n,P,4]
            zn = xn[..., 2]
            gx = xn[..., 0] / (zn * math.tan(nb.FoVx * 0.5))
            gy = xn[..., 1] / (zn * math.tan(nb.FoVy * 0.5))
            inside = (zn > 0.01).all(-1) & (gx.abs() < 1).all(-1) & (gy.abs() < 1).all(-1)
            grid = torch.stack([gx, gy], -1)[None]  # [1,n,P,2]
            nb_patch = F.grid_sample(self.gray_of(nb), grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
            cost = (1 - self._ncc(ref_patch, nb_patch)).clamp(0, 2)
            costs.append(torch.where(inside, cost, torch.full_like(cost, float("inf"))))
        best = torch.stack(costs, 0).min(0).values
        seen = torch.isfinite(best)
        if not seen.any():
            return zero, 0
        return best[seen].mean(), int(seen.sum())
