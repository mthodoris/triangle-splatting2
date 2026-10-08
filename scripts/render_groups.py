#
# For a model trained with --free_triangles: PSNR / SSIM / LPIPS on the test views of the full
# model, the mesh group alone and the free group alone, plus where the free triangles are.
# The mesh-only numbers show what the mesh itself carries.
#
# Usage:
#   python scripts/render_groups.py -m <model_path> -s <dataset> -i images_4 --eval [--iteration 30000]
# Writes <model_path>/groups.json and <model_path>/groups/<variant>/<view>.png for --save_views.
#

import os
import sys
import json
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene import Scene
from triangle_renderer import render, TriangleModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips


if __name__ == "__main__":
    parser = ArgumentParser(description="Render the mesh and free triangle groups separately")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_views", nargs="*", type=int, default=[12], help="test views to save as images")
    args = get_combined_args(parser)
    dataset, pipe = model.extract(args), pipeline.extract(args)

    with torch.no_grad():
        triangles = TriangleModel(dataset.sh_degree)
        triangles.scaling = 4
        scene = Scene(args=dataset, triangles=triangles, init_opacity=None, set_sigma=None,
                      load_iteration=args.iteration, shuffle=False)
        if triangles._vertex_is_mesh is None:
            sys.exit("no triangle groups in this model (trained without --free_triangles)")
        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

        is_mesh = triangles.triangle_is_mesh()
        all_idx = triangles._triangle_indices.clone()
        all_size, all_importance = triangles.image_size, triangles.importance_score
        result = {"iteration": scene.loaded_iter, "mesh_faces": int(is_mesh.sum()), "free_faces": int((~is_mesh).sum())}

        for variant, keep in (("full", None), ("mesh_only", is_mesh), ("free_only", ~is_mesh)):
            triangles._triangle_indices, triangles.image_size, triangles.importance_score = all_idx.clone(), all_size, all_importance
            if keep is not None:
                triangles.prune_triangles(keep)
            out_dir = os.path.join(dataset.model_path, "groups", variant)
            os.makedirs(out_dir, exist_ok=True)
            p, s, l = [], [], []
            for i, view in enumerate(scene.getTestCameras()):
                image = render(view, triangles, pipe, background)["render"].clamp(0, 1)[None]
                gt = view.original_image[0:3].cuda()[None]
                p.append(psnr(image, gt).mean().item())
                s.append(ssim(image, gt).item())
                l.append(lpips(image, gt, net_type="vgg").item())
                if i in args.save_views:
                    torchvision.utils.save_image(image[0], os.path.join(out_dir, "{:05d}.png".format(i)))
            result[variant] = {"PSNR": float(np.mean(p)), "SSIM": float(np.mean(s)), "LPIPS": float(np.mean(l))}
            print(variant, result[variant], flush=True)

    with open(os.path.join(dataset.model_path, "groups.json"), "w") as f:
        json.dump(result, f, indent=1)
