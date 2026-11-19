"""Known-answer checks for physproxy/eval/mesh_stats.py.

Run:  python -m pytest physproxy/tests -q   (or python physproxy/tests/test_mesh_stats.py)
"""
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physproxy.eval.mesh_stats import mesh_stats  # noqa: E402


def _stats(m):
    return mesh_stats(np.asarray(m.vertices, float), np.asarray(m.faces, np.int64))


def test_closed_sphere():
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    s = _stats(m)
    assert s["watertight"] and s["winding_consistent"]
    assert s["boundary_edges"] == 0 and s["nonmanifold_edges"] == 0
    assert s["components"] == 1
    assert abs(s["volume"] - m.volume) < 1e-9
    assert abs(s["volume"] - 4 / 3 * np.pi) < 0.05


def test_open_box():
    m = trimesh.creation.box()
    faces = m.faces[1:]                    # remove one triangle -> a 3-edge hole
    s = mesh_stats(np.asarray(m.vertices, float), faces)
    assert not s["watertight"]
    assert s["boundary_edges"] == 3
    assert "volume" not in s


def test_two_components_and_largest_area():
    a = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    b = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    b.apply_translation([5, 0, 0])
    s = _stats(trimesh.util.concatenate([a, b]))
    assert s["components"] == 2
    assert abs(s["largest_component_area_frac"] - a.area / (a.area + b.area)) < 1e-9


def test_nonmanifold_fin():
    m = trimesh.creation.box()
    v = np.vstack([m.vertices, [[0.0, 0.0, 3.0]]])
    e = m.edges_unique[0]
    f = np.vstack([m.faces, [[e[0], e[1], len(v) - 1]]])    # third face on an edge
    s = mesh_stats(v, f)
    assert s["nonmanifold_edges"] == 1 and not s["watertight"]
    assert s["boundary_edges"] == 2


def test_flipped_face_and_duplicates():
    m = trimesh.creation.icosphere(subdivisions=1)
    f = m.faces.copy()
    f[0] = f[0][::-1]
    s = mesh_stats(np.asarray(m.vertices, float), f)
    assert s["watertight"] and not s["winding_consistent"]
    assert s["winding_inconsistent_edges"] == 3
    f2 = np.vstack([m.faces, m.faces[:1]])
    s2 = mesh_stats(np.asarray(m.vertices, float), f2)
    assert s2["duplicate_faces"] == 1 and s2["nonmanifold_edges"] == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
