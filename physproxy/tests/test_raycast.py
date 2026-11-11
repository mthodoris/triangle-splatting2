"""Phase 6 ray caster: analytic distances, max_dist cutoff, grid culling == brute force, comparison classes."""
import os
import sys

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physproxy.eval.raycast import cast, compare, unit_sphere  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _mesh(m):
    return (torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=DEV),
            torch.tensor(np.asarray(m.faces), dtype=torch.int64, device=DEV))


def test_sphere_from_center():
    V, F = _mesh(trimesh.creation.icosphere(5, 1.0))
    g = torch.Generator(device=DEV).manual_seed(0)
    d = unit_sphere(5000, g, DEV)
    t = cast(torch.zeros_like(d), d, V, F, max_dist=2.0)
    assert torch.isfinite(t).all()
    assert (t - 1.0).abs().max() < 2e-3               # icosphere chords: within 0.2% of the radius


def test_plane_distance_and_cutoff():
    V, F = _mesh(trimesh.creation.box((10, 10, 0.01)))   # thin slab around z = 0, top at 0.005
    o = torch.tensor([[0.0, 0.0, 0.5], [1.0, 2.0, 3.0], [0.0, 0.0, 0.5]], device=DEV)
    d = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], device=DEV)
    t = cast(o, d, V, F, max_dist=1.0)
    assert abs(float(t[0]) - 0.495) < 1e-4
    assert torch.isinf(t[1]) and torch.isinf(t[2])     # beyond max_dist / pointing away


def test_grid_culling_matches_brute_force():
    m = trimesh.creation.icosphere(3, 1.0)
    pieces = [m.copy().apply_translation(c) for c in [(0, 0, 0), (2.5, 0.3, 0), (-1, 3, 1)]]
    V, F = _mesh(trimesh.util.concatenate(pieces))
    g = torch.Generator(device=DEV).manual_seed(1)
    o = (torch.rand(4000, 3, generator=g, device=DEV) - 0.5) * 8
    d = unit_sphere(4000, g, DEV)
    t_grid = cast(o, d, V, F, max_dist=1.5, cell=0.7)
    t_bf = cast(o, d, V, F, max_dist=1.5, cell=1000.0)
    assert torch.equal(torch.isfinite(t_grid), torch.isfinite(t_bf))
    fin = torch.isfinite(t_bf)
    assert fin.any() and (t_grid[fin] - t_bf[fin]).abs().max() < 1e-5


def test_compare_classes():
    inf = float("inf")
    t_ref = torch.tensor([inf, 1.0, 1.0, 1.0, inf, 1.0])
    t_tst = torch.tensor([inf, 1.01, 0.5, 2.0, 0.3, inf])
    r = compare(t_ref, t_tst, tol=0.05)
    assert abs(r["agree"] - 2 / 6) < 1e-6
    assert abs(r["early_invisible_wall"] - 2 / 6) < 1e-6
    assert abs(r["late_fall_through"] - 2 / 6) < 1e-6
