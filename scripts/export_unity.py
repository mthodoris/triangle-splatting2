#!/usr/bin/env python3
"""Export a (textured or plain) TS+ checkpoint for Unity with view-dependent colour (plan step 9.2).

Writes OUT/ (copy it to <UnityProject>/Assets/StreamingAssets/trisplat/<name>/):
  render.tsm   binary mesh (format below): triangle soup in Unity's frame, atlas UVs, per-corner source vertex id,
               and the SH coefficients of every source vertex (float16)
  atlas.png    per-triangle tiles = interpolated view-INdependent colour (SH degree 0, clamped >= 0) + texture
               (export_textured_mesh.bake); plain TS+ models get the same with a zero texture
  meta.json    frame, scale, sizes, provenance
  cameras.json the test-view cameras in Unity's frame (position, forward, up, vertical FOV, aspect)

Colour in Unity (shader TriSplat/SHAtlas), same formula as eval_engine_view.py --sh (TS+):
  per vertex v:  c0_v = max(C0*dc_v + 0.5, 0),  c_v = max(eval_sh(deg, sh_v, dir_v) + 0.5, 0),  r_v = c_v - c0_v
                 dir_v = normalize(vertex - camera) in the SCENE frame
  per pixel:     colour = saturate(atlas(uv) + interp(r))   (atlas already = interp(c0) + texture)
  The only approximation: the atlas is 8-bit and clamped to [0, 1] before r is added.

Frame (as physproxy/unity/export_for_unity.py): gravity-aligned box from the SfM points (input.ply) and the training
cameras (physproxy.aabb.oriented_proxy_box); Unity (x, y, z) = s * (lx - cx, lz - lo_z, ly - cy) with local = frame @
world, s = --units_to_m (garden: 0.579 m/unit, physproxy Metric3D estimate). Unity is left-handed, Y up; the axis swap
is a reflection, so triangle order is reversed (as physproxy does) to keep normals outward for colliders.

.tsm format (little endian):
  b"TSM1", int32 n_src_vertices, int32 n_triangles, int32 n_sh_coeffs (16 for degree 3), int32 atlas_w, int32 atlas_h
  float32[n_tri*3*3] corner positions (Unity frame, metres)
  float32[n_tri*3*2] corner UVs (v up)
  int32[n_tri*3]     source vertex id of each corner
  float16[n_src_vertices * n_sh_coeffs * 3] SH coefficients, [vertex][coeff][rgb], scene frame
                     (coeff 0 = DC; Unity's shader converts its view direction to the scene frame, meta.json)

Usage: python scripts/export_unity.py MODEL_DIR --out OUT [--units_to_m 0.579] [--upsample 0]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from export_textured_mesh import auto_upsample, bake  # noqa: E402
from physproxy.aabb import oriented_proxy_box  # noqa: E402

SH_C0 = 0.28209479177387814


def load_points(path):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)


def scene_frame(model_dir, radius_mult=1.5):
    """(frame [3,3], lo [3], hi [3]) as physproxy's ProxyTrainer: train cameras = all but every 8th (sorted, --eval)."""
    cams = sorted(json.load(open(os.path.join(model_dir, "cameras.json"))), key=lambda c: c["img_name"])
    train = [c for i, c in enumerate(cams) if i % 8 != 0]
    centres = torch.tensor([c["position"] for c in train], dtype=torch.float32)
    ups = torch.tensor([[-r[0][1], -r[1][1], -r[2][1]] for r in (c["rotation"] for c in train)], dtype=torch.float32)
    pts = load_points(os.path.join(model_dir, "input.ply"))
    return oriented_proxy_box(pts, centres, ups, radius_mult=radius_mult)


def unity_transform(frame, lo, hi, s):
    """Affine map scene -> Unity: u = A @ x + b. Returns A [3,3] (incl. scale), b [3]."""
    c = 0.5 * (lo + hi)
    P = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], np.float64)       # (lx, ly, lz) -> (lx, lz, ly)
    A = s * P @ frame
    b = -s * P @ np.array([c[0], c[1], lo[2]], np.float64)
    return A, b


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="model dir (point_cloud/iteration_30000, cameras.json, input.ply)")
    p.add_argument("--out", required=True)
    p.add_argument("--units_to_m", type=float, default=0.5791363418102264,
                   help="metres per scene unit (default: garden, physproxy metric_scale.json)")
    p.add_argument("--sdf", default=None, help="physproxy sdf_<it>.pt: take frame + box from it (exact match with the "
                   "colliders of unity/export_for_unity.py); default: recompute (same function, same inputs)")
    p.add_argument("--upsample", type=int, default=1, help="atlas tile upsampling k (1 = raw texels; with --texture_wd 1.0 k = 2/4 gave identical PSNR; 0 = auto)")
    a = p.parse_args()

    sd = torch.load(os.path.join(a.model, "point_cloud/iteration_30000/point_cloud_state_dict.pt"),
                    map_location="cpu", weights_only=False)
    verts = sd["triangles_points"].detach().double().numpy()
    faces = sd["_triangle_indices"].detach().long().numpy()
    shs = torch.cat([sd["features_dc"], sd["features_rest"]], 1).detach().float().numpy()   # [V, K, 3]
    deg = int(sd["active_sh_degree"])
    K = (deg + 1) ** 2
    shs = shs[:, :K]
    tex = sd.get("texture")
    tex = np.zeros((len(faces), 3, 2, 2), np.float32) if tex is None else tex.detach().float().numpy()
    T, S = len(faces), tex.shape[-1]

    if a.sdf:
        st = torch.load(a.sdf, map_location="cpu", weights_only=False)["state_dict"]
        frame, lo, hi = (st[k].double().numpy() for k in ("frame", "aabb_min", "aabb_max"))
    else:
        frame, lo, hi = (t.double().numpy() for t in scene_frame(a.model))
    A, b = unity_transform(frame, lo, hi, a.units_to_m)

    col0 = np.maximum(SH_C0 * shs[:, 0] + 0.5, 0.0)
    k = a.upsample or auto_upsample(T, S)
    atlas, uv = bake(faces, col0, tex, k)                       # uv [T, 3, 2], v up
    print(f"{T} triangles, {len(verts)} vertices, SH degree {deg}, texture {S}x{S}, atlas {atlas.shape[1]}x{atlas.shape[0]} (k={k})")

    rev = [0, 2, 1]                                             # reflection -> reverse corner order
    corners = faces[:, rev]
    pos = (verts[corners] @ A.T + b).astype(np.float32)          # [T, 3, 3]
    uvr = uv[:, rev].astype(np.float32)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "render.tsm"), "wb") as f:
        f.write(b"TSM1")
        f.write(struct.pack("<5i", len(verts), T, K, atlas.shape[1], atlas.shape[0]))
        f.write(pos.astype("<f4").tobytes())
        f.write(uvr.astype("<f4").tobytes())
        f.write(corners.astype("<i4").tobytes())
        f.write(np.ascontiguousarray(shs).astype("<f2").tobytes())
    Image.MAX_IMAGE_PIXELS = None
    Image.fromarray((np.clip(atlas, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(a.out, "atlas.png"), compress_level=1)

    # Unity world direction d_u = A d_scene / s  ->  d_scene = (A / s)^T d_u  (A / s is orthogonal).
    scene_dir_from_unity = (A / a.units_to_m).T
    meta = dict(model=os.path.abspath(a.model), units_to_m=a.units_to_m, sh_degree=deg, n_sh_coeffs=K,
                texture_res=S, atlas_upsample=k, n_triangles=T, n_src_vertices=len(verts),
                frame_rows=frame.tolist(), box_lo=lo.tolist(), box_hi=hi.tolist(),
                unity_from_scene_A=A.tolist(), unity_from_scene_b=b.tolist(),
                scene_dir_from_unity=scene_dir_from_unity.tolist(),
                scene_dir_from_unity_flat=scene_dir_from_unity.reshape(-1).tolist(),   # for Unity's JsonUtility
                box_size_unity=[float(hi[0] - lo[0]) * a.units_to_m, float(hi[2] - lo[2]) * a.units_to_m,
                                float(hi[1] - lo[1]) * a.units_to_m])
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    # Test-view cameras (every 8th, as --eval) in Unity's frame, for the viewer's V key (TriSplatViewer.cs).
    R_ = A / a.units_to_m
    cams = sorted(json.load(open(os.path.join(a.model, "cameras.json"))), key=lambda c: c["img_name"])[::8]
    out_c = []
    for c in cams:
        Rc = np.array(c["rotation"])                              # camera-to-world, OpenCV (x right, y down, z fwd)
        out_c.append(dict(name=c["img_name"], pos=(A @ np.array(c["position"]) + b).tolist(),
                          forward=(R_ @ Rc[:, 2]).tolist(), up=(-(R_ @ Rc[:, 1])).tolist(),
                          fovY=float(np.degrees(2 * np.arctan(c["height"] / (2 * c["fy"])))),
                          aspect=c["width"] / c["height"],
                          fx_n=c["fx"] / c["width"], fy_n=c["fy"] / c["height"]))   # focal / image size (fx != fy)
    json.dump(dict(cams=out_c), open(os.path.join(a.out, "cameras.json"), "w"), indent=1)
    mb = sum(os.path.getsize(os.path.join(a.out, f)) for f in ("render.tsm", "atlas.png")) / 2**20
    print(f"wrote {a.out} ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
