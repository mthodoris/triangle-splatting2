#!/usr/bin/env python3
"""fit_mesh_to_soup.py

Option 2 from the mesh-init discussion: instead of asking the triangle-splatting
optimizer to preserve a clean mesh's topology while it prunes/densifies (which
always fragments it into "triangle soup" — see create_from_mesh/add_new_gs),
decouple the two goals entirely.

1. Train an ordinary, fully unconstrained triangle-splatting run (no --mesh_path,
   no --fix_mesh) to get the best possible photometric reconstruction — the
   "soup" checkpoint this script reads.
2. Run this script to non-rigidly deform your own clean mesh's vertices so its
   surface matches the geometry implied by that soup, via a Chamfer distance
   data term plus Laplacian/normal regularizers that keep the mesh well-formed.
   The clean mesh's faces (its topology) are never touched, so it can never be
   broken into soup — only its vertex positions move.

Usage:
    python fit_mesh_to_soup.py \\
        --soup_checkpoint output/<soup_run>/point_cloud/iteration_30000/point_cloud_state_dict.pt \\
        --mesh_path garden_..._cleaned_mesh.ply \\
        --out fitted_mesh.ply
"""
import argparse
import sys

import numpy as np
import torch
import trimesh
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

SH_C0 = 0.28209479177387814


def sh_dc_to_rgb(features_dc):
    colors = SH_C0 * features_dc + 0.5
    return colors.clamp(0.0, 1.0)


def compact_mesh(verts, faces):
    """Drop vertices no surviving face references, and remap face indices."""
    used = torch.unique(faces)
    remap = torch.full((verts.shape[0],), -1, dtype=torch.long, device=verts.device)
    remap[used] = torch.arange(used.shape[0], device=verts.device)
    return verts[used], remap[faces]


def load_soup(checkpoint_path, importance_threshold, min_opacity, device):
    """Load a plain (non mesh-init) triangle-splatting checkpoint and keep only
    the triangles that actually contributed to a rendered view — the same
    filter train.py's own end-of-training cleanup uses — so junk/never-visible
    geometry doesn't get used as a fitting target."""
    sd = torch.load(checkpoint_path, map_location=device)
    verts = sd["triangles_points"].to(device).to(torch.float32)
    faces = sd["_triangle_indices"].to(device).to(torch.int64)
    vertex_weight = sd["vertex_weight"].to(device).to(torch.float32)
    features_dc = sd["features_dc"].to(device).to(torch.float32)
    importance_score = sd.get("importance_score", None)

    opacity = torch.sigmoid(vertex_weight).squeeze(-1)

    keep_tri = torch.ones(faces.shape[0], dtype=torch.bool, device=device)
    if importance_score is not None and importance_score.numel() == faces.shape[0]:
        keep_tri &= (importance_score.to(device) > importance_threshold)
    else:
        print("Warning: soup checkpoint has no usable importance_score; skipping that filter.")

    tri_opacity = opacity[faces].min(dim=1).values
    keep_tri &= (tri_opacity > min_opacity)

    n_before = faces.shape[0]
    faces = faces[keep_tri]
    n_after = faces.shape[0]
    print(f"Soup filter: kept {n_after}/{n_before} triangles "
          f"(importance > {importance_threshold}, min vertex opacity > {min_opacity}).")
    if n_after == 0:
        raise RuntimeError("No soup triangles survived filtering — loosen --importance_threshold / --min_opacity.")

    verts, faces = compact_mesh(verts, faces)
    colors = sh_dc_to_rgb(features_dc.squeeze(1))
    return verts, faces, colors


def load_target_mesh(mesh_path, device):
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int64, device=device)
    return verts, faces


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--soup_checkpoint", required=True,
                   help="point_cloud_state_dict.pt from a plain (no --mesh_path) triangle-splatting run.")
    p.add_argument("--mesh_path", required=True,
                   help="Your clean mesh to deform. Its faces/topology are never modified.")
    p.add_argument("--out", default="fitted_mesh.ply")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_samples", type=int, default=200_000,
                   help="Points sampled from each mesh surface per iteration for the Chamfer term.")
    p.add_argument("--w_chamfer", type=float, default=1.0)
    p.add_argument("--w_laplacian", type=float, default=0.1,
                   help="Uniform Laplacian smoothing weight — keeps the deformation locally smooth.")
    p.add_argument("--w_normal", type=float, default=0.01,
                   help="Normal-consistency weight — discourages the surface from flipping/spiking.")
    p.add_argument("--w_anchor", type=float, default=0.0,
                   help="L2 penalty pulling vertices back toward their original position. Use this if "
                        "regions with little/no soup coverage drift too far under Chamfer alone.")
    p.add_argument("--importance_threshold", type=float, default=0.5,
                   help="Soup triangles with importance_score <= this are treated as never-visible junk "
                        "and excluded from the fitting target (same threshold train.py's own cleanup uses).")
    p.add_argument("--min_opacity", type=float, default=0.05,
                   help="Soup triangles whose least-opaque vertex is below this are also excluded.")
    p.add_argument("--transfer_color", action="store_true", default=True)
    p.add_argument("--no_transfer_color", dest="transfer_color", action="store_false")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    soup_verts, soup_faces, soup_colors = load_soup(
        args.soup_checkpoint, args.importance_threshold, args.min_opacity, device)
    soup_mesh = Meshes(verts=[soup_verts], faces=[soup_faces])

    target_verts, target_faces = load_target_mesh(args.mesh_path, device)
    print(f"Target mesh: {target_verts.shape[0]} vertices, {target_faces.shape[0]} faces "
          f"(topology fixed for the entire fit).")

    verts_init = target_verts.clone()
    verts_param = torch.nn.Parameter(target_verts.clone())
    optimizer = torch.optim.Adam([verts_param], lr=args.lr)

    with torch.no_grad():
        soup_points = sample_points_from_meshes(soup_mesh, args.n_samples)

    for it in range(1, args.iterations + 1):
        optimizer.zero_grad()
        current_mesh = Meshes(verts=[verts_param], faces=[target_faces])

        pred_points = sample_points_from_meshes(current_mesh, args.n_samples)
        loss_chamfer, _ = chamfer_distance(pred_points, soup_points)

        loss_laplacian = mesh_laplacian_smoothing(current_mesh, method="uniform")
        loss_normal = mesh_normal_consistency(current_mesh)
        loss_anchor = ((verts_param - verts_init) ** 2).mean()

        loss = (args.w_chamfer * loss_chamfer
                + args.w_laplacian * loss_laplacian
                + args.w_normal * loss_normal
                + args.w_anchor * loss_anchor)

        loss.backward()
        optimizer.step()

        if it % args.log_every == 0 or it == 1:
            print(f"[{it:5d}/{args.iterations}] total {loss.item():.6f}  "
                  f"chamfer {loss_chamfer.item():.6f}  laplacian {loss_laplacian.item():.6f}  "
                  f"normal {loss_normal.item():.6f}  anchor {loss_anchor.item():.6f}")

    final_verts = verts_param.detach()

    if args.transfer_color:
        _, idx, _ = knn_points(final_verts.unsqueeze(0), soup_verts.unsqueeze(0), K=1)
        vertex_colors = soup_colors[idx.squeeze(0).squeeze(-1)]
    else:
        vertex_colors = torch.full((final_verts.shape[0], 3), 0.5, device=device)

    verts_np = final_verts.cpu().numpy()
    faces_np = target_faces.cpu().numpy()
    colors_u8 = (vertex_colors.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

    out_mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, vertex_colors=colors_u8, process=False)
    out_mesh.export(args.out)
    displacement = np.linalg.norm(verts_np - verts_init.cpu().numpy(), axis=1)
    print(f"Saved fitted mesh to {args.out}")
    print(f"Vertices: {verts_np.shape[0]}, Faces: {faces_np.shape[0]} (topology identical to --mesh_path)")
    print(f"Displacement from original mesh: mean {displacement.mean():.5f}, max {displacement.max():.5f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
