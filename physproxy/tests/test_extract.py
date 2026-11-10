"""Zero-level extraction: watertight, right volume, right place, world frame."""
import math
import os
import sys

import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physproxy.sdf import SDFField  # noqa: E402
from physproxy.extract import extract_mesh  # noqa: E402
from physproxy.eval.mesh_stats import mesh_stats  # noqa: E402
from physproxy.tests.test_sdf import _rotation  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _tm(v, f):
    return trimesh.Trimesh(v.cpu().numpy(), f.cpu().numpy(), process=False)


def test_sphere_in_rotated_box():
    R = _rotation(5)
    lo, hi = torch.tensor([-1.0, -0.8, -0.6]), torch.tensor([1.0, 0.8, 0.6])
    sdf = SDFField(lo, hi, frame=R, init="sphere", init_radius=0.8).to(DEV)     # r = 0.48
    v, f = extract_mesh(sdf, res=128)
    m = _tm(v, f)
    s = mesh_stats(v.double().cpu().numpy(), f.cpu().numpy())
    assert s["watertight"] and s["winding_consistent"] and s["components"] == 1
    r = sdf.init_radius
    assert abs(m.volume - 4 / 3 * math.pi * r ** 3) / (4 / 3 * math.pi * r ** 3) < 0.01   # positive = outward
    assert ((v - sdf.init_center).norm(dim=-1) - r).abs().max() < 0.01                   # world frame


def test_ground_plane_becomes_closed_slab():
    R = _rotation(2)
    lo, hi = torch.tensor([-2.0, -1.0, -1.0]), torch.tensor([2.0, 1.0, 1.0])
    # Plane at local z = -0.4 with normal = local up: inside below it.
    sdf = SDFField(lo, hi, frame=R, init="plane", init_offset=-0.4).to(DEV)
    v, f = extract_mesh(sdf, res=64)
    s = mesh_stats(v.double().cpu().numpy(), f.cpu().numpy())
    assert s["watertight"] and s["winding_consistent"]
    vol = _tm(v, f).volume
    assert abs(vol - 4 * 2 * 0.6) / (4 * 2 * 0.6) < 0.01
    zl = sdf.to_local(v)[:, 2]
    assert zl.min() >= -1.0 - 1e-5 and zl.max() <= -0.4 + 1e-3


def test_checkpoint_roundtrip(tmp_path=None):
    import tempfile
    R = _rotation(4)
    sdf = SDFField(torch.tensor([-1.0, -1, -1]), torch.tensor([1.0, 1, 1]), frame=R, init="plane", init_offset=-0.3).to(DEV)
    with torch.no_grad():
        sdf.mlp[-1].weight.normal_(0, 0.1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sdf.pt")
        torch.save({"state_dict": sdf.state_dict(), "ctor_kwargs": sdf.ctor_kwargs}, path)
        sdf2 = SDFField.from_checkpoint(path, device=DEV)
    x = sdf.sample_uniform(1000)
    assert torch.allclose(sdf(x), sdf2(x), atol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
