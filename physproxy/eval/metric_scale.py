#!/usr/bin/env python3
"""Metres per scene unit, from Metric3D metric depth vs the render mesh's depth.

COLMAP scenes have no absolute scale. Metric3D (ViT-L, already used by the dataloader for normals)
predicts metric depth when run with its official recipe: resize to its 616 x 1064 canonical input,
pad, normalise with ImageNet mean/std, and convert with the focal length
(depth_m = prediction * fx_resized / 1000). Comparing it with the z-depth of the render mesh on the
same pixels gives one metres-per-unit ratio per pixel. Only pixels whose surface lies inside the proxy
box are used (the region that matters for physics; sky and far background excluded). Reported: the
median over views of each view's median ratio, and the spread across views.

  python physproxy/eval/metric_scale.py -s SCENE -i images_8 -m MODEL [--metric_images images_4 --views 24]
"""
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physproxy.compat import use_cached_metric3d  # noqa: E402
use_cached_metric3d()
from arguments import ModelParams  # noqa: E402
from scene import Scene  # noqa: E402
from scene.triangle_model import TriangleModel  # noqa: E402
from physproxy.depth import view_rays, view_to_world  # noqa: E402
from physproxy.eval.zbuffer import p3d_zbuffer  # noqa: E402
from physproxy.eval.raycast import load_obj_mesh  # noqa: E402

INPUT = (616, 1064)
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)


@torch.no_grad()
def metric_depth(model, rgb_u8, fx):
    """rgb_u8 [H,W,3] uint8 -> metric z-depth [H,W] (metres), following Metric3D's hubconf example."""
    H, W = rgb_u8.shape[:2]
    s = min(INPUT[0] / H, INPUT[1] / W)
    h, w = int(H * s), int(W * s)
    x = torch.from_numpy(rgb_u8).permute(2, 0, 1)[None].float()
    x = Fn.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    ph, pw = INPUT[0] - h, INPUT[1] - w
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)          # left, right, top, bottom
    x = (x - MEAN) / STD
    x = Fn.pad(x, pad, value=0.0)                                 # padding colour = mean -> 0 after normalising
    pred, _, _ = model.inference({"input": x.cuda()})
    pred = pred[0, 0, pad[2]:INPUT[0] - pad[3], pad[0]:INPUT[1] - pad[1]]
    pred = Fn.interpolate(pred[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return pred * (fx * s / 1000.0)


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--metric_images", default="images_4", help="image folder to run Metric3D on (higher res is better)")
    parser.add_argument("--views", type=int, default=24, help="number of training views (evenly spaced)")
    parser.add_argument("--sdf", default=None)
    parser.add_argument("--mesh", default=None, help="render mesh (default MODEL/collision/render.obj)")
    args = parser.parse_args()
    dataset = lp.extract(args)
    dataset.eval = True
    torch.set_grad_enabled(False)
    from physproxy.export_collision import latest
    M = args.model_path
    sd = torch.load(args.sdf or latest(os.path.join(M, "proxy", "sdf_*.pt")), map_location="cuda")["state_dict"]
    frame, lo, hi = sd["frame"], sd["aabb_min"], sd["aabb_max"]
    V, F = load_obj_mesh(args.mesh or os.path.join(M, "collision", "render.obj"))

    tri = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, tri, None, None, load_iteration=args.iteration, shuffle=False)
    cams = scene.getTrainCameras()
    pick = np.linspace(0, len(cams) - 1, args.views).round().astype(int)
    model = torch.hub.load("yvanyin/metric3d:main", "metric3d_vit_large", pretrain=True, skip_validation=True).cuda().eval()

    per_view = []
    for i in pick:
        view = cams[i]
        img = Image.open(os.path.join(dataset.source_path, args.metric_images, view.image_name + ".JPG")).convert("RGB")
        rgb = np.asarray(img)
        H, W = rgb.shape[:2]
        fx = W / (2 * np.tan(view.FoVx / 2))
        dm = metric_depth(model, rgb, fx)
        z, fid = p3d_zbuffer(view, V, F, H, W)
        hit = fid >= 0
        p = view_to_world(view, (view_rays(view, H, W, "cuda") * z[..., None]).reshape(-1, 3)).reshape(H, W, 3)
        pl = p @ frame.T
        m = hit & ((pl >= lo) & (pl <= hi)).all(-1) & (dm > 0) & (dm < 80)
        if m.sum() < 1000:
            continue
        r = (dm[m] / z[m]).float()
        per_view.append(dict(view=view.image_name, pixels=int(m.sum()), median=float(r.median()),
                             p25=float(torch.quantile(r[:2_000_000], 0.25)), p75=float(torch.quantile(r[:2_000_000], 0.75))))
        print(f"[scale] {view.image_name}: {per_view[-1]['median']:.3f} m/unit (IQR {per_view[-1]['p25']:.3f}-{per_view[-1]['p75']:.3f}, {m.sum()} px)")
    med = np.array([v["median"] for v in per_view])
    out = dict(metres_per_unit=float(np.median(med)), views=len(per_view), across_views_p10=float(np.percentile(med, 10)),
               across_views_p90=float(np.percentile(med, 90)), images=args.metric_images, per_view=per_view)
    with open(os.path.join(M, "metric_scale.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[scale] metres per scene unit: {out['metres_per_unit']:.3f} (views p10-p90 {out['across_views_p10']:.3f}-"
          f"{out['across_views_p90']:.3f}, {len(per_view)} views) -> {M}/metric_scale.json")


if __name__ == "__main__":
    main()
