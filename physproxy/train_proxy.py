#!/usr/bin/env python3
"""Train Triangle Splatting+ with the SDF collision proxy, without editing upstream code.

Usage: exactly like train.py, plus the --proxy* flags of physproxy/args.py:
  python physproxy/train_proxy.py -s SCENE -i images_8 -m OUT --eval --proxy [--proxy_start 20000 ...]
Without --proxy it is plain upstream training (same arguments, same calls).

How it hooks in (all at runtime, in this process only; upstream files are untouched):
  * train.Scene      -> subclass that builds the ProxyTrainer after the scene loads
  * train.render     -> wrapper that, for training-camera renders, snapshots
                        (view, rend_ids, validity, vertices, faces); tensors are kept
                        by reference, so later pruning (which replaces the tensors)
                        cannot desynchronize ids and faces
  * optimizer.step   -> wrapper that first runs the SDF step on the last snapshot
                        (once per TS+ iteration: upstream calls step once per
                        iteration, after pruning), then the original step
  * after training() returns: proxy.finish (extra SDF-only steps, export)
Test-camera renders (from the periodic report) are never used for training.
The vertices are detached (Phase 4a, one-way coupling), so the TS+ optimization
sees no change.
"""
import os
import sys
from argparse import ArgumentParser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from physproxy.compat import use_cached_metric3d  # noqa: E402
use_cached_metric3d()

import torch  # noqa: E402
import train as upstream  # noqa: E402  (its __main__ block does not run on import)
from arguments import ModelParams, PipelineParams, OptimizationParams, update_indoor  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from physproxy.args import ProxyParams  # noqa: E402
from physproxy.depth import pixel_alpha  # noqa: E402


def install_hooks(proxy_args):
    state = {"proxy": None, "scene": None, "last": None, "iteration": 0, "test_ids": set(), "train_ids": set()}
    orig_render = upstream.render
    orig_scene = upstream.Scene

    class ProxyScene(orig_scene):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            from physproxy.train_hook import ProxyTrainer
            state["scene"] = self
            state["train_ids"] = {id(c) for c in self.getTrainCameras()}
            state["test_ids"] = {id(c) for c in self.getTestCameras()}
            state["proxy"] = ProxyTrainer(proxy_args, self, self.triangles, self.model_path)

    def render(viewpoint_camera, pc, pipe, bg_color, *a, **kw):
        pkg = orig_render(viewpoint_camera, pc, pipe, bg_color, *a, **kw)
        opt = getattr(pc, "optimizer", None)
        if opt is not None and not getattr(opt, "_physproxy_wrapped", False):
            orig_step = opt.step

            def step(*sa, **skw):
                state["iteration"] += 1
                last = state["last"]
                if last is not None:
                    state["proxy"].step(state["iteration"], *last, pc)
                    state["last"] = None
                return orig_step(*sa, **skw)
            opt.step = step
            opt._physproxy_wrapped = True
        if id(viewpoint_camera) in state["train_ids"]:
            state["last"] = (viewpoint_camera, pkg["rend_ids"].detach(), pixel_alpha(pkg).detach(),
                             pc.get_vertices.detach(), pc.get_triangle_indices)
        return pkg

    upstream.Scene = ProxyScene
    upstream.render = render
    return state


def main():
    # Same arguments as upstream train.py's __main__, plus the proxy group.
    parser = ArgumentParser(description="Training script parameters (+ physproxy)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    xp = ProxyParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--indoor", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    import lpips
    upstream.lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")   # training_report reads this global
    safe_state(args.quiet)
    lps, ops, pps, xargs = lp.extract(args), op.extract(args), pp.extract(args), xp.extract(args)
    if args.indoor:
        ops = update_indoor(ops)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    state = install_hooks(xargs) if xargs.proxy else None
    upstream.training(lps, ops, pps, args.test_iterations, args.start_checkpoint, args.debug_from)
    if state is not None:
        scene = state["scene"]
        bg = torch.tensor([1, 1, 1] if lps.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        state["proxy"].finish(ops.iterations, scene, scene.triangles, pps, bg)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
