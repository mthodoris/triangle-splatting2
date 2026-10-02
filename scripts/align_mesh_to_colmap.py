#
# Move a mesh from one COLMAP reconstruction's frame into another's.
#
# Two COLMAP runs on the same images end up in different coordinate frames
# (here: gs2mesh's re-run vs. the original MipNeRF360 poses). The images they
# share give matching camera centers, from which we fit a similarity transform
# (scale, rotation, translation) and apply it to the mesh vertices.
#
# Usage:
#   python scripts/align_mesh_to_colmap.py \
#       --mesh mesh.ply --src_colmap <dataset the mesh was made from> \
#       --dst_colmap <dataset you will train on> --out mesh_aligned.ply
#

import os
import sys
from argparse import ArgumentParser

import numpy as np
import trimesh

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text, qvec2rotmat


def camera_centers(colmap_dir):
    sparse = os.path.join(colmap_dir, "sparse", "0")
    if os.path.exists(os.path.join(sparse, "images.bin")):
        extrinsics = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
    else:
        extrinsics = read_extrinsics_text(os.path.join(sparse, "images.txt"))
    return {e.name: -qvec2rotmat(e.qvec).T @ e.tvec for e in extrinsics.values()}


def fit_similarity(src, dst):
    """Umeyama: find s, R, t minimizing ||dst - (s * R @ src + t)||."""
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    U, S, Vt = np.linalg.svd(dst_c.T @ src_c / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / src_c.var(0).sum()
    t = mu_dst - s * R @ mu_src
    return s, R, t


if __name__ == "__main__":
    parser = ArgumentParser(description="Align a mesh between two COLMAP frames using shared camera poses")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--src_colmap", required=True, help="COLMAP dataset whose frame the mesh is in")
    parser.add_argument("--dst_colmap", required=True, help="COLMAP dataset to move the mesh into")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    src_centers = camera_centers(args.src_colmap)
    dst_centers = camera_centers(args.dst_colmap)
    names = sorted(set(src_centers) & set(dst_centers))
    if len(names) < 3:
        sys.exit("Only {} images in common between the two datasets, need at least 3".format(len(names)))

    src = np.array([src_centers[n] for n in names])
    dst = np.array([dst_centers[n] for n in names])
    s, R, t = fit_similarity(src, dst)
    residual = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print("{} shared cameras".format(len(names)))
    print("scale {:.6f}, rotation {:.3f} deg, translation {}".format(s, angle, t))
    print("camera residual after alignment: mean {:.5f}, max {:.5f}".format(residual.mean(), residual.max()))

    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    transform = np.eye(4)
    transform[:3, :3] = s * R
    transform[:3, 3] = t
    # apply_transform also rotates the vertex normals; vertex colors and faces are untouched
    mesh.apply_transform(transform)
    mesh.export(args.out)
    np.savetxt(os.path.splitext(args.out)[0] + "_transform.txt", transform)
    print("Saved aligned mesh to {}".format(args.out))
