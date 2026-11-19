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
    state = {"proxy": None, "scene": None, "last": None, "iteration": 0, "test_ids": set(), "train_ids": set(),
             "support": None, "support_faces": None, "v_param": None}
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
                it = state["iteration"]
                proxy = state["proxy"]
                last = state["last"]
                if last is not None:
                    proxy.step(it, *last, pc)
                    state["last"] = None
                if proxy.coupling_active(it):
                    # (i) pull: only if upstream did not replace the vertex tensor this iteration
                    # (pruning/densification), otherwise its gradients are gone anyway.
                    if proxy_args.proxy_pull_w > 0 and pc.vertices is state["v_param"] and pc.vertices.grad is not None:
                        proxy.pull(it, pc)
                    # (ii) floater pruning, half-way between upstream's pruning iterations.
                    every = proxy_args.proxy_prune_every
                    if proxy_args.proxy_prune_floaters and it % every == every // 2:
                        sup = state["support"] if state["support_faces"] is pc.get_triangle_indices else None
                        proxy.prune_floaters(it, pc, sup)
                return orig_step(*sa, **skw)
            opt.step = step
            opt._physproxy_wrapped = True
        if id(viewpoint_camera) in state["train_ids"]:
            alpha = pixel_alpha(pkg).detach()
            state["last"] = (viewpoint_camera, pkg["rend_ids"].detach(), alpha,
                             pc.get_vertices.detach(), pc.get_triangle_indices)
            state["v_param"] = pc.vertices
            # View support: in how many training renders each triangle is the visible one
            # somewhere; reset whenever upstream replaces the face tensor.
            Fc = pc.get_triangle_indices
            if state["support_faces"] is not Fc:
                state["support"] = torch.zeros(Fc.shape[0], device=Fc.device)
                state["support_faces"] = Fc
            with torch.no_grad():
                ids = torch.round(pkg["rend_ids"].reshape(-1)).long()
                ok = (alpha.reshape(-1) > 0) & (ids >= 0) & (ids < Fc.shape[0])
                seen = torch.unique(ids[ok])
                state["support"][seen] += 1
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
