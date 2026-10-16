#
# Which triangles of a trained model does any camera actually see? Renders every train and test view and keeps,
# per triangle, the largest blending weight it reached in any view (the same "importance" the training uses for
# pruning). Reports mesh and free groups separately and writes <model_path>/visibility.json and
# <model_path>/visibility/triangles_visibility.ply (face colours: green = seen, yellow = barely, red = never).
# With --reference <mesh.ply> --n_original N, the never-seen mesh faces are also split by whether the nearest
# face of that reference mesh is one of its first N (original) faces or a later one (a hole fill).
#
# Usage:
#   python scripts/triangle_visibility.py -m <model_path> -s <dataset> -i images_4 --eval [--iteration 30000] \
#       [--scale 2] [--reference filled.ply --n_original 47101]
#

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene import Scene
from triangle_renderer import render, TriangleModel
from arguments import ModelParams, PipelineParams, get_combined_args


if __name__ == "__main__":
    parser = ArgumentParser(description="Per-triangle visibility over all cameras")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--scale", default=2, type=int, help="render scaling (training used 2)")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--n_original", default=0, type=int)
    args = get_combined_args(parser)
    dataset, pipe = model.extract(args), pipeline.extract(args)

    with torch.no_grad():
        triangles = TriangleModel(dataset.sh_degree)
        triangles.scaling = args.scale
        scene = Scene(args=dataset, triangles=triangles, init_opacity=None, set_sigma=None,
                      load_iteration=args.iteration, shuffle=False)
        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        T = triangles._triangle_indices.shape[0]
        is_mesh = triangles.triangle_is_mesh()
        best = torch.zeros(T, device="cuda")
        views = list(scene.getTrainCameras()) + list(scene.getTestCameras())
        for i, view in enumerate(views):
            pkg = render(view, triangles, pipe, background)
            best = torch.maximum(best, pkg["max_blending"].detach())
            if i % 50 == 0:
                print("view {}/{}".format(i, len(views)), flush=True)
        best = best.cpu().numpy()
        is_mesh = is_mesh.cpu().numpy()

        result = {"iteration": scene.loaded_iter, "views": len(views)}
        for name, sel in (("mesh", is_mesh), ("free", ~is_mesh)):
            b = best[sel]
            result[name] = {"faces": int(sel.sum()),
                            "never_seen_pct(<0.01)": float(100 * (b < 0.01).mean()) if len(b) else 0.0,
                            "barely_seen_pct(<0.1)": float(100 * (b < 0.1).mean()) if len(b) else 0.0,
                            "below_0.5_pct": float(100 * (b < 0.5).mean()) if len(b) else 0.0}

        V = triangles.vertices.detach().cpu().numpy()
        F = triangles._triangle_indices.cpu().numpy().astype(np.int64)
        centroid = V[F].mean(1)
        if args.reference:
            ref = trimesh.load(args.reference, process=False, force="mesh")
            _, nearest = cKDTree(np.asarray(ref.triangles_center)).query(centroid[is_mesh])
            from_fill = nearest >= args.n_original
            low = best[is_mesh] < 0.01
            result["mesh_never_seen_near_a_fill_pct"] = float(100 * from_fill[low].mean()) if low.any() else 0.0
            result["mesh_faces_near_a_fill_pct"] = float(100 * from_fill.mean())
            result["fill_faces_never_seen_pct"] = float(100 * low[from_fill].mean()) if from_fill.any() else 0.0
            result["original_faces_never_seen_pct"] = float(100 * low[~from_fill].mean()) if (~from_fill).any() else 0.0

        out_dir = os.path.join(dataset.model_path, "visibility")
        os.makedirs(out_dir, exist_ok=True)
        colors = np.zeros((T, 4), dtype=np.uint8)
        colors[:] = [0, 170, 60, 255]
        colors[best < 0.1] = [240, 200, 0, 255]
        colors[best < 0.01] = [230, 30, 30, 255]
        colors[~is_mesh] = np.where((best[~is_mesh] < 0.5)[:, None], [140, 0, 200, 255], [0, 110, 255, 255])  # free: blue seen, purple weak
        trimesh.Trimesh(V, F, face_colors=colors, process=False).export(os.path.join(out_dir, "triangles_visibility.ply"))
        np.save(os.path.join(out_dir, "importance.npy"), best)
        with open(os.path.join(dataset.model_path, "visibility.json"), "w") as f:
            json.dump(result, f, indent=1)
        print(json.dumps(result, indent=1))
