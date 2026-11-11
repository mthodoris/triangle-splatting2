#!/usr/bin/env python3
"""Phase 6: "what you see is what you hit" on test (or training) views, for several colliders at once.

For every pixel whose visible render surface (exact TS+ depth) lies in the proxy box, the collider's
z-buffer depth is compared with the rendered depth (relative tolerance `--tol`):
agree / blocks (collider in front: invisible wall) / see_through (collider behind or missing).
This is proxy_views.compare_views run per mesh; see that module for details.

  python physproxy/eval/consistency.py -s SCENE -i images_8 -m MODEL --sdf SDF.pt \
      --mesh ours=MODEL/collision/collision.obj naive=MODEL/collision/naive.obj [--out JSON]
"""
import json
import os
import sys
from argparse import ArgumentParser

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physproxy.compat import use_cached_metric3d  # noqa: E402
use_cached_metric3d()
from arguments import ModelParams, PipelineParams  # noqa: E402
from scene import Scene  # noqa: E402
from scene.triangle_model import TriangleModel  # noqa: E402
from physproxy.eval.proxy_views import compare_views  # noqa: E402
from physproxy.eval.raycast import load_obj_mesh  # noqa: E402


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sdf", required=True, help="sdf_<it>.pt (box and frame)")
    parser.add_argument("--mesh", nargs="+", required=True, help="name=path of each collider")
    parser.add_argument("--split", default="test", choices=["test", "train"],
                        help="train: for choosing settings (e.g. the skin) without looking at the test views")
    parser.add_argument("--every", type=int, default=1, help="use every k-th view of the split")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--save_images", type=int, default=4)
    parser.add_argument("--out", default=None, help="JSON output (default MODEL/consistency.json)")
    args = parser.parse_args()
    dataset, pipe = lp.extract(args), pp.extract(args)
    dataset.eval = True

    sd = torch.load(args.sdf, map_location="cuda")["state_dict"]
    frame, lo, hi = sd["frame"], sd["aabb_min"], sd["aabb_max"]
    res = {"sdf": args.sdf, "tol": args.tol, "results": {}}
    with torch.no_grad():
        tri = TriangleModel(dataset.sh_degree)
        scene = Scene(dataset, tri, None, None, load_iteration=args.iteration, shuffle=False)
        tri.scaling = args.scale
        cams = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
        cams = cams[::args.every]
        res["split"], res["views"] = args.split, len(cams)
        for spec in args.mesh:
            name, path = spec.split("=", 1)
            PV, PF = load_obj_mesh(path)
            out_dir = os.path.join(os.path.dirname(path), f"views_{name}")
            s = compare_views(cams, tri, pipe, PV, PF, frame, lo, hi, tol=args.tol, out_dir=out_dir,
                              save_images=args.save_images if args.split == "test" else 0)
            s.pop("per_view", None)
            res["results"][name] = dict(path=path, faces=int(PF.shape[0]), **s)
            print(f"[consistency] {name:>10} ({PF.shape[0]} faces): agree {s['agree']:.4f}  blocks {s['blocks']:.4f}"
                  f"  see-through {s['see_through']:.4f}")
    out = args.out or os.path.join(args.model_path, "consistency.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[consistency] wrote {out}")


if __name__ == "__main__":
    main()
