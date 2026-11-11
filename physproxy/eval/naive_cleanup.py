#!/usr/bin/env python3
"""Phase 6 baseline: "naive cleanup" collider made directly from the render mesh.

What one would do without a proxy: take the render triangles inside the proxy box, weld vertices
closer than `--weld`, drop duplicate/degenerate faces, drop connected components with area below
`--min_comp_area` (crumbs only: a single render triangle has area ~0.003, so the
proxy's 0.05 threshold would delete ~95% of the soup), then quadric-decimate to `--budget`
faces (same budget). No hole filling: the result is generally not watertight.

  python physproxy/eval/naive_cleanup.py -m MODEL [--sdf SDF.pt] [--out MODEL/collision/naive.obj]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physproxy.eval.mesh_stats import load_mesh, mesh_stats  # noqa: E402
from physproxy.export_collision import _vf, drop_small_components, latest, ml_mesh, write_obj  # noqa: E402


def naive_cleanup(v, f, frame, lo, hi, weld=0.02, min_comp_area=1e-3, budget=50000):
    import pymeshlab as ml
    loc = v @ frame.T
    inside = ((loc >= lo) & (loc <= hi)).all(-1)
    f = f[inside[f].all(1)]
    ms = ml.MeshSet()
    ms.add_mesh(ml_mesh(v, f))
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_merge_close_vertices(threshold=ml.PureValue(weld))
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    vw, fw = _vf(ms)
    welded = mesh_stats(vw, fw)
    vk, fk, comp = drop_small_components(vw, fw, min_comp_area)
    ms = ml.MeshSet()
    ms.add_mesh(ml_mesh(vk, fk))
    if fk.shape[0] > budget:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(budget), qualitythr=0.3, preservenormal=True,
                                                    optimalplacement=True, planarquadric=True, autoclean=True)
    vd, fd = _vf(ms)
    info = dict(faces_in_box=int(f.shape[0]), welded_components=welded.get("components"),
                welded_faces=int(fw.shape[0]), after_component_filter=int(fk.shape[0]), **comp)
    return vd, fd, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--sdf", default=None, help="sdf_<it>.pt for the box (default: latest in MODEL/proxy)")
    ap.add_argument("--render", default=None, help="render checkpoint/mesh (default: latest point_cloud iteration)")
    ap.add_argument("--weld", type=float, default=0.02)
    ap.add_argument("--min_comp_area", type=float, default=1e-3,
                    help="area of one render triangle is ~0.003, so larger thresholds delete most of the soup")
    ap.add_argument("--budget", type=int, default=50000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sdf_path = a.sdf or latest(os.path.join(a.model, "proxy", "sdf_*.pt"))
    render_path = a.render or latest(os.path.join(a.model, "point_cloud", "iteration_*"))
    out = a.out or os.path.join(a.model, "collision", "naive.obj")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    sd = torch.load(sdf_path, map_location="cpu")["state_dict"]
    frame, lo, hi = (sd[k].double().numpy() for k in ("frame", "aabb_min", "aabb_max"))
    v, f = load_mesh(render_path)
    vd, fd, info = naive_cleanup(v, f, frame, lo, hi, a.weld, a.min_comp_area, a.budget)
    st = mesh_stats(vd, fd)
    write_obj(out, vd, fd)
    rep = dict(render=render_path, sdf=sdf_path, args=vars(a), steps=info, stats=st)
    with open(os.path.splitext(out)[0] + "_report.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"[naive] in box {info['faces_in_box']} faces -> welded {info['welded_faces']} faces / "
          f"{info['welded_components']} components -> kept {info['components_kept']} components, "
          f"{info['after_component_filter']} faces -> {st['faces']} faces")
    print(f"[naive] watertight {st.get('watertight')}, boundary edges {st.get('boundary_edges')} "
          f"({st.get('boundary_edge_frac', 0):.1%}), non-manifold edges {st.get('nonmanifold_edges')}, "
          f"components {st.get('components')}; wrote {out}")


if __name__ == "__main__":
    main()
