#!/usr/bin/env python3
"""Fit only the SDF proxy on a trained, frozen TS+ model, then score it on the test views.

With one-way coupling (Phase 4a) the triangles never see the SDF, so the proxy
can be fitted after training on the final triangles. This makes ablations of
the proxy settings take minutes instead of a full 30k-iteration run.

  python physproxy/fit_proxy.py -s SCENE -i images_8 -m MODEL --name thin_t01 \
      --steps 5000 [--proxy_* flags as in physproxy/args.py]

Output: MODEL/proxy_fits/<name>/ with physproxy_log.jsonl, sdf_<steps>.pt,
mesh_<steps>.ply, eval_<steps>.json (mesh stats, render<->proxy distances) and
views/summary.json (+ pictures): agree / blocks / see_through on the test views.
Every step renders a random training view (no grad) at --scale, like the end of
training, and takes one SDF step on its exact depth.
"""
import json
import os
import random
import sys
import time
from argparse import ArgumentParser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from physproxy.compat import use_cached_metric3d  # noqa: E402
use_cached_metric3d()

import torch  # noqa: E402
from arguments import ModelParams, PipelineParams  # noqa: E402
from scene import Scene  # noqa: E402
from scene.triangle_model import TriangleModel  # noqa: E402
from triangle_renderer import render  # noqa: E402
from physproxy.args import ProxyParams  # noqa: E402
from physproxy.depth import pixel_alpha  # noqa: E402
from physproxy.train_hook import ProxyTrainer  # noqa: E402
from physproxy.eval.proxy_views import compare_views  # noqa: E402


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    xp = ProxyParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--scale", type=int, default=4, help="render upsampling (training ends at 4)")
    parser.add_argument("--eval_scale", type=int, default=2)
    parser.add_argument("--save_images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dataset, pipe, xargs = lp.extract(args), pp.extract(args), xp.extract(args)
    dataset.eval = True
    xargs.proxy = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = os.path.join(dataset.model_path, "proxy_fits", args.name)
    os.makedirs(out, exist_ok=True)

    tri = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, tri, None, None, load_iteration=args.iteration, shuffle=False)
    with torch.no_grad():
        tri.vertices = tri.vertices.detach()
    proxy = ProxyTrainer(xargs, scene, tri, dataset.model_path, out_dir=out)
    cams = scene.getTrainCameras()
    bg = torch.zeros(3, device="cuda")

    t0 = time.time()
    tri.scaling = args.scale
    stack = []
    for k in range(1, args.steps + 1):
        if not stack:
            stack = list(range(len(cams)))
            random.shuffle(stack)
        view = cams[stack.pop()]
        with torch.no_grad():
            pkg = render(view, tri, pipe, bg)
        proxy._sdf_step(k, view, pkg["rend_ids"], pixel_alpha(pkg), tri.get_vertices, tri.get_triangle_indices)
    fit_s = time.time() - t0
    ev = proxy.export(args.steps, tri, final=True)
    proxy.log.close()

    # What you see vs what you hit, on the test views.
    import numpy as np
    import trimesh
    m = trimesh.load(os.path.join(out, f"mesh_{args.steps}.ply"), process=False)
    PV = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device="cuda")
    PF = torch.tensor(np.asarray(m.faces), dtype=torch.int64, device="cuda")
    tri.scaling = args.eval_scale
    sdf = proxy.sdf
    summ = compare_views(scene.getTestCameras(), tri, pipe, PV, PF, sdf.frame, sdf.aabb_min, sdf.aabb_max,
                         out_dir=os.path.join(out, "views"), save_images=args.save_images)
    summ["fit_seconds"] = fit_s
    summ["eval"] = ev
    with open(os.path.join(out, "views", "summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1, default=float)
    line = {"name": args.name, "agree": round(summ["agree"], 4), "blocks": round(summ["blocks"], 4),
            "see_through": round(summ["see_through"], 4),
            "render_to_proxy_p90": round(ev.get("render_to_proxy_p90", float("nan")), 4),
            "proxy_to_render_p90": round(ev.get("proxy_to_render_p90", float("nan")), 4),
            "faces": ev.get("mesh", {}).get("faces"), "components": ev.get("mesh", {}).get("components"),
            "watertight": ev.get("mesh", {}).get("watertight"), "fit_s": round(fit_s)}
    print("[fit_proxy]", json.dumps(line))
    with open(os.path.join(dataset.model_path, "proxy_fits", "summary.jsonl"), "a") as fh:
        fh.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
