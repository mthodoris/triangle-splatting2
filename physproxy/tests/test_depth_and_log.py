"""Known-answer checks for physproxy/depth.py and physproxy/log.py."""
import math
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physproxy.depth import exact_depth, view_rays, world_to_view, view_to_world  # noqa: E402
from physproxy.log import JsonLogger, read_log  # noqa: E402
from utils.graphics_utils import getWorld2View2  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class FakeView:
    def __init__(self, R, T, fovx, fovy):
        self.FoVx, self.FoVy = fovx, fovy
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).to(DEV)


def _rot(axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_roundtrip_world_view():
    R = _rot("y", 0.3) @ _rot("x", -0.2)
    v = FakeView(R, np.array([0.1, -0.4, 2.0]), 1.0, 0.7)
    p = torch.randn(100, 3, device=DEV)
    assert torch.allclose(view_to_world(v, world_to_view(v, p)), p, atol=1e-5)


def test_tilted_plane_depth_exact():
    # Camera at the origin looking down +z (R = I, T = 0). A big tilted triangle
    # z = 3 + 0.5 x covers the whole image; every pixel's exact depth is known.
    v = FakeView(np.eye(3), np.zeros(3), 1.2, 0.9)
    H, W = 30, 40
    verts = torch.tensor([[-50.0, -50.0, 3 - 25.0], [50.0, -50.0, 3 + 25.0], [0.0, 60.0, 3.0]], device=DEV)
    faces = torch.tensor([[0, 1, 2]], device=DEV, dtype=torch.int32)
    ids = torch.zeros(1, H, W, device=DEV)
    alpha = torch.ones(1, H, W, device=DEV)
    alpha[0, 0, 0] = 0.2  # one pixel transparent -> invalid
    out = exact_depth(v, verts, faces, ids, alpha)
    d = view_rays(v, H, W, DEV)
    # z = 3 + 0.5 * x with x = z * dx  ->  z = 3 / (1 - 0.5 dx)
    z_true = 3.0 / (1 - 0.5 * d[..., 0])
    m = out["valid"]
    assert not m[0, 0] and m.sum() == H * W - 1
    assert torch.allclose(out["depth"][m], z_true[m], rtol=1e-5)
    assert (out["bary"][m].min(dim=-1).values >= 0).all()
    # The back-projected point lies on the plane.
    P = out["points_world"][m]
    assert torch.allclose(P[:, 2], 3 + 0.5 * P[:, 0], atol=1e-4)


def test_depth_with_rotated_camera_matches_point():
    # A known world point: put a small triangle around it, project, and compare.
    R = _rot("y", 0.4)
    v = FakeView(R, np.array([0.3, 0.1, 4.0]), 1.0, 0.8)
    target = torch.tensor([[0.2, -0.1, 0.5]], device=DEV)
    pv = world_to_view(v, target)[0]
    H, W = 50, 60
    tx, ty = math.tan(0.5), math.tan(0.4)
    i = int(((pv[0] / pv[2] / tx + 1) * W - 1) / 2 + 0.5)
    j = int(((pv[1] / pv[2] / ty + 1) * H - 1) / 2 + 0.5)
    tri = target + torch.tensor([[0.3, 0, 0], [-0.2, 0.3, 0.1], [-0.2, -0.3, -0.1]], device=DEV)
    ids = torch.full((1, H, W), -1.0, device=DEV)
    ids[0, j, i] = 0
    alpha = torch.ones(1, H, W, device=DEV)
    out = exact_depth(v, tri, torch.tensor([[0, 1, 2]], device=DEV), ids, alpha)
    assert out["valid"].sum() == 1
    # Pixel centre is within half a pixel of the projection: depth within the plane's slope.
    assert abs(out["depth"][j, i].item() - pv[2].item()) < 0.05


def test_json_logger():
    with tempfile.TemporaryDirectory() as d:
        lg = JsonLogger(d, config={"lambda_surf": 1.0})
        lg.log(10, "loss", L_surf=torch.tensor(0.5), n=3)
        lg.log(20, "loss", L_surf=0.25, n=4)
        lg.close()
        recs = read_log(os.path.join(d, "physproxy_log.jsonl"), tag="loss")
        assert [r["iter"] for r in recs] == [10, 20]
        assert recs[0]["L_surf"] == 0.5 and recs[1]["n"] == 4
        assert read_log(os.path.join(d, "physproxy_log.jsonl"), tag="config")[0]["lambda_surf"] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
