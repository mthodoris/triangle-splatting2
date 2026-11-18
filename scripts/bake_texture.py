#!/usr/bin/env python3
"""Plan step 8.3, "paint afterwards" baseline: add textures to a FINISHED plain TS+ model and train only them.

Geometry, opacity and vertex colours stay frozen; each triangle gets a zero S x S residual texture that is optimised
with the same image loss as train.py (0.8 L1 + 0.2 D-SSIM), the same texture lr and the same texture shrink.
Default 17000 iterations = as many texture iterations as our joint method gets (textures from it13000 to 30000).
If our joint training beats this at the same triangle count, learning textures *with* the geometry matters.

The output folder is a normal model folder (cfg_args, cameras.json, point_cloud/iteration_30000) → render.py and
metrics.py work on it unchanged.

Usage (from combined/, venv active):
  python scripts/bake_texture.py --source <plain model dir> -s <dataset> -i images_8 -m <out dir> [--iterations 17000]
         [--texture_res 8] [--texture_lr 0.0025] [--texture_wd 0.3] [--test_every 1000]
"""
import argparse, os, shutil, sys
from random import randint

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arguments import ModelParams                      # noqa: E402
from scene import Scene                                 # noqa: E402
from scene.triangle_model import TriangleModel          # noqa: E402
from triangle_renderer import render                    # noqa: E402
from utils.loss_utils import l1_loss, ssim              # noqa: E402
from utils.image_utils import psnr                      # noqa: E402


def main():
    p = argparse.ArgumentParser()
    mp = ModelParams(p)
    p.add_argument("--source", required=True)
    p.add_argument("--iterations", type=int, default=17000)
    p.add_argument("--texture_lr", type=float, default=0.0025)
    p.add_argument("--texture_wd", type=float, default=0.3)
    p.add_argument("--lambda_dssim", type=float, default=0.2)
    p.add_argument("--test_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    S = a.texture_res
    assert S > 0

    # output = copy of the plain model folder; the textured result overwrites its point cloud at the end
    out = a.model_path
    src_pc = os.path.join(a.source, "point_cloud", "iteration_30000")
    os.makedirs(os.path.join(out, "point_cloud"), exist_ok=True)
    shutil.copytree(src_pc, os.path.join(out, "point_cloud", "iteration_30000"), dirs_exist_ok=True)
    for f in ("cameras.json", "input.ply"):
        if os.path.exists(os.path.join(a.source, f)):
            shutil.copy(os.path.join(a.source, f), out)
    dataset = mp.extract(a)
    with open(os.path.join(out, "cfg_args"), "w") as f:
        f.write(str(argparse.Namespace(**vars(dataset))))

    triangles = TriangleModel(dataset.sh_degree)
    triangles.scaling = 4                                   # as render.py (final TS+ setting)
    scene = Scene(dataset, triangles, None, None, load_iteration=30000)
    assert triangles.texture_res == 0, "--source must be a plain TS+ model"
    for t in (triangles.vertices, triangles.vertex_weight, triangles._features_dc, triangles._features_rest):
        t.requires_grad_(False)                             # frozen geometry, opacity and colours
    T = triangles._triangle_indices.shape[0]
    triangles.texture_res = S
    triangles.perspective_uv = not dataset.screen_space_uv
    triangles._texture = nn.Parameter(torch.zeros((T, 3, S, S), device="cuda"))
    if not hasattr(triangles, "image_size") or not torch.is_tensor(triangles.image_size) or triangles.image_size.numel() != T:
        triangles.image_size = torch.zeros(T, device="cuda")
    opt = torch.optim.Adam([triangles._texture], lr=a.texture_lr, eps=1e-15)
    print(f"baking {S}x{S} textures onto {T} frozen triangles of {a.source}, {a.iterations} its, "
          f"lr {a.texture_lr}, wd {a.texture_wd}", flush=True)

    bg = torch.zeros(3, device="cuda")
    pipe = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=1.0, debug=False,
                              atomic_vertex_grads=False)

    def test(it):
        with torch.no_grad():
            v = [float(psnr(render(c, triangles, pipe, bg)["render"].clamp(0, 1), c.original_image.cuda()).mean())
                 for c in scene.getTestCameras()]
        print(f"[ITER {it}] test PSNR {sum(v) / len(v):.3f}", flush=True)

    test(0)
    stack = []
    for it in range(1, a.iterations + 1):
        if not stack:
            stack = scene.getTrainCameras().copy()
        cam = stack.pop(randint(0, len(stack) - 1))
        img = render(cam, triangles, pipe, bg)["render"]
        gt = cam.original_image.cuda()
        loss = (1 - a.lambda_dssim) * l1_loss(img, gt) + a.lambda_dssim * (1 - ssim(img, gt))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if a.texture_wd > 0:
            with torch.no_grad():
                triangles._texture.mul_(1 - a.texture_lr * a.texture_wd)
        if it % a.test_every == 0 or it == a.iterations:
            test(it)
    triangles.save_parameters(os.path.join(out, "point_cloud", "iteration_30000"))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
