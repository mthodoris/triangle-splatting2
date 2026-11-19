"""Phase 5 pieces: component filter, orientation, non-manifold repair, decimation, distances."""
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physproxy.export_collision import (decimate, drop_small_components, one_sided, orient_outward,  # noqa: E402
                                        repair)
from physproxy.eval.mesh_stats import mesh_stats  # noqa: E402


def _sphere(r=1.0, sub=3, shift=(0, 0, 0)):
    m = trimesh.creation.icosphere(sub, r)
    return np.asarray(m.vertices) + np.asarray(shift), np.asarray(m.faces, np.int64)


def _merge(*meshes):
    vs, fs, n = [], [], 0
    for v, f in meshes:
        vs.append(v)
        fs.append(f + n)
        n += v.shape[0]
    return np.concatenate(vs), np.concatenate(fs)


def test_drop_small_components():
    v, f = _merge(_sphere(1.0), _sphere(0.05, 2, (3, 0, 0)))
    v2, f2, info = drop_small_components(v, f, min_area=0.5)
    assert info["components_before"] == 2 and info["components_kept"] == 1
    assert f2.shape[0] == _sphere(1.0)[1].shape[0] and f2.max() < v2.shape[0]


def test_orient_outward_fixes_random_flips_and_inside_out_shells():
    v, f = _merge(_sphere(1.0), _sphere(0.3, 2, (3, 0, 0)))
    rng = np.random.default_rng(0)
    m = rng.random(f.shape[0]) < 0.4
    f[m] = f[m][:, ::-1]
    assert not mesh_stats(v, f)["winding_consistent"]
    g, conflicts = orient_outward(v, f)
    st = mesh_stats(v, g)
    assert conflicts == 0 and st["winding_consistent"]
    exact = 4 / 3 * np.pi * (1 + 0.3 ** 3)
    assert abs(st["volume"] - exact) / exact < 0.03          # positive: every shell faces outward


def test_repair_nonmanifold_edge():
    # Two closed boxes glued along one edge: that edge has 4 faces.
    a = trimesh.creation.box((1, 1, 1))
    b = trimesh.creation.box((1, 1, 1))
    b.apply_translation((1, 1, 0))
    v, f = _merge((np.asarray(a.vertices), np.asarray(a.faces)), (np.asarray(b.vertices), np.asarray(b.faces)))
    tm = trimesh.Trimesh(v, f, process=True)                 # merge the shared edge's vertices
    v, f = np.asarray(tm.vertices), np.asarray(tm.faces, np.int64)
    assert mesh_stats(v, f)["nonmanifold_edges"] > 0
    v2, f2, _ = repair(v, f)
    st = mesh_stats(v2, f2)
    assert st["watertight"] and st["nonmanifold_edges"] == 0 and st["winding_consistent"]
    assert st["volume"] > 0


def test_decimate_then_repair_sphere():
    v, f = _sphere(1.0, 5)                                   # 20480 faces
    vd, fd = decimate(v, f, 2000)
    assert fd.shape[0] <= 2100
    vc, fc, _ = repair(vd, fd)
    st = mesh_stats(vc, fc)
    assert st["watertight"] and st["winding_consistent"] and st["volume"] > 0
    assert np.abs(np.linalg.norm(vc, axis=1) - 1).max() < 0.02


def test_one_sided_distance_concentric_spheres():
    a, b = _sphere(1.0, 4), _sphere(1.1, 4)
    d = one_sided(*a, *b, n=20000)
    assert abs(d["p50"] - 0.1) < 0.005 and abs(d["max"] - 0.1) < 0.005
