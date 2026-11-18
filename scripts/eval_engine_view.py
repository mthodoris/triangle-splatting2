#!/usr/bin/env python3
"""'Game-engine view' metrics: render a TS+ / textured model as an opaque mesh with a z-buffer
(nearest surface per pixel, as Unity draws it) on all test views and compare with the photos.

For comparison it can also use TS+'s own visibility rule (--visibility centre: first triangle in order of
centre depth). Colour: vertex colour (SH degree 0 = what our Unity export shows, or full SH with --sh)
+ texture residual with the exact CUDA formula (perspective-correct UV, clamp to triangle, bilinear).
Final pixel clamped to [0, 1]. Pixels not covered by any triangle are black (TS+ background).

Usage: python scripts/eval_engine_view.py <model_dir> --images <dataset>/images_4 [--sh] [--visibility zbuf|centre]
       [--save_dir dir] [--views 0,8]   -> prints mean PSNR / SSIM / LPIPS, writes runs/engine_eval/<run>__<tag>.json
"""
import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim
from lpipsPyTorch import lpips

SH_C0 = 0.28209479177387814


def load_model(model_dir, dev):
    sd = torch.load(os.path.join(model_dir, "point_cloud/iteration_30000/point_cloud_state_dict.pt"),
                    map_location=dev, weights_only=False)
    m = dict(verts=sd["triangles_points"].detach().double(), faces=sd["_triangle_indices"].long(),
             shs=torch.cat([sd["features_dc"], sd["features_rest"]], 1).detach().float().transpose(1, 2),
             deg=int(sd["active_sh_degree"]))
    m["tex"] = sd["texture"].detach().float() if sd.get("texture") is not None else None
    return m


@torch.no_grad()
def render(m, cam, W, H, visibility="zbuf", sh=False, budget=12_000_000, out=None):
    """out: optional dict; gets "depth" [H, W] (camera z of the visible surface, inf = no triangle; zbuf only)."""
    dev = m["verts"].device
    s = W / cam["width"]; fx, fy = cam["fx"] * s, cam["fy"] * s
    R = torch.tensor(cam["rotation"], dtype=torch.float64, device=dev)
    t = torch.tensor(cam["position"], dtype=torch.float64, device=dev)
    pc = (m["verts"] - t) @ R
    tri = pc[m["faces"]]; Z = tri[..., 2]
    xs = tri[..., 0] / Z * fx + W / 2; ys = tri[..., 1] / Z * fy + H / 2
    keep = (Z > 0.01).all(1) & (xs.max(1).values >= 0) & (xs.min(1).values < W) & (ys.max(1).values >= 0) & (ys.min(1).values < H)
    x0 = xs.min(1).values.floor().clamp(0, W - 1).long(); x1 = xs.max(1).values.ceil().clamp(0, W - 1).long()
    y0 = ys.min(1).values.floor().clamp(0, H - 1).long(); y1 = ys.max(1).values.ceil().clamp(0, H - 1).long()
    ids = torch.nonzero(keep)[:, 0]
    area = ((x1 - x0 + 1) * (y1 - y0 + 1))[ids]
    csum = torch.cumsum(area, 0)
    best = torch.full((H * W,), float("inf"), dtype=torch.float64, device=dev)
    win = torch.full((H * W,), -1, dtype=torch.long, device=dev)
    wts = torch.zeros((H * W, 3), dtype=torch.float64, device=dev)
    start = 0
    while start < len(ids):
        base = csum[start - 1] if start > 0 else 0
        end = max(start + 1, int(torch.searchsorted(csum, base + budget)))
        tid = ids[start:end]; ar = area[start:end]
        rep = torch.repeat_interleave(torch.arange(len(tid), device=dev), ar)
        off = torch.arange(int(ar.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(ar, 0) - ar, ar)
        tr = tid[rep]; bw = (x1 - x0 + 1)[tr]
        X = x0[tr] + off % bw; Y = y0[tr] + off // bw
        del rep, off, bw
        fxp, fyp = X.double() + 0.5, Y.double() + 0.5
        ax, ay = xs[tr], ys[tr]
        d = (ax[:, 1] - ax[:, 0]) * (ay[:, 2] - ay[:, 0]) - (ax[:, 2] - ax[:, 0]) * (ay[:, 1] - ay[:, 0])
        b1 = ((fxp - ax[:, 0]) * (ay[:, 2] - ay[:, 0]) - (ax[:, 2] - ax[:, 0]) * (fyp - ay[:, 0])) / d
        b2 = ((ax[:, 1] - ax[:, 0]) * (fyp - ay[:, 0]) - (fxp - ax[:, 0]) * (ay[:, 1] - ay[:, 0])) / d
        b0 = 1 - b1 - b2
        ins = (b0 >= 0) & (b1 >= 0) & (b2 >= 0) & (d.abs() > 1e-12)
        b = torch.stack([b0, b1, b2], 1)[ins]; tt = tr[ins]; pix = (Y * W + X)[ins]
        del ax, ay, d, b0, b1, b2, fxp, fyp, X, Y, tr
        bz = b / Z[tt]; depth = 1.0 / bz.sum(1); bp = bz * depth[:, None]   # perspective-correct weights
        key = depth if visibility == "zbuf" else Z[tt].mean(1)
        kmin = torch.full((H * W,), float("inf"), dtype=torch.float64, device=dev).scatter_reduce(0, pix, key, "amin")
        better = (key == kmin[pix]) & (key < best[pix])
        best[pix[better]] = key[better]; win[pix[better]] = tt[better]; wts[pix[better]] = bp[better]
        start = end
    if out is not None:
        out["depth"] = best.reshape(H, W) if visibility == "zbuf" else None
    # colours
    if sh:
        dirs = (m["verts"] - t).float(); dirs = dirs / dirs.norm(dim=1, keepdim=True)
        col = torch.clamp_min(eval_sh(m["deg"], m["shs"], dirs) + 0.5, 0.0)
    else:
        col = torch.clamp_min(SH_C0 * m["shs"][:, :, 0] + 0.5, 0.0)
    img = torch.zeros((H * W, 3), device=dev)
    msk = win >= 0; tw = win[msk]; w = wts[msk].float()
    f = m["faces"][tw]
    c = w[:, 0:1] * col[f[:, 0]] + w[:, 1:2] * col[f[:, 1]] + w[:, 2:3] * col[f[:, 2]]
    if m["tex"] is not None:
        S = m["tex"].shape[-1]
        u, v = w[:, 1].clamp_min(0), w[:, 2].clamp_min(0); sm = (u + v).clamp_min(1)
        x, y = u / sm * (S - 1), v / sm * (S - 1)
        xi = torch.clamp(x.floor().long(), max=S - 2); yi = torch.clamp(y.floor().long(), max=S - 2)
        fx_, fy_ = (x - xi)[:, None], (y - yi)[:, None]
        tt = m["tex"][tw]; r = torch.arange(len(tw), device=dev)
        c = c + ((1 - fx_) * (1 - fy_) * tt[r, :, yi, xi] + fx_ * (1 - fy_) * tt[r, :, yi, xi + 1]
                 + (1 - fx_) * fy_ * tt[r, :, yi + 1, xi] + fx_ * fy_ * tt[r, :, yi + 1, xi + 1])
    img[msk] = c.clamp(0, 1)
    return img.reshape(H, W, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model"); p.add_argument("--images", required=True)
    p.add_argument("--visibility", default="zbuf", choices=["zbuf", "centre"])
    p.add_argument("--sh", action="store_true")
    p.add_argument("--save_dir"); p.add_argument("--views", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "../../runs/engine_eval"))
    a = p.parse_args()
    dev = "cuda"
    m = load_model(a.model, dev)
    cams = sorted(json.load(open(os.path.join(a.model, "cameras.json"))), key=lambda c: c["img_name"])
    test = cams[::8]                                         # llffhold 8, as TS+ --eval
    if a.views:
        test = [test[int(i)] for i in a.views.split(",")]
    res = []
    for cam in test:
        gt = Image.open(os.path.join(a.images, cam["img_name"] + ".JPG")).convert("RGB")
        W, H = gt.size
        gt = torch.tensor(np.asarray(gt), dtype=torch.float32, device=dev) / 255
        img = render(m, cam, W, H, a.visibility, a.sh)
        x, y = img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
        psnr = float(-10 * torch.log10(((x - y) ** 2).mean()))
        res.append(dict(name=cam["img_name"], psnr=psnr, ssim=float(ssim(x, y)), lpips=float(lpips(x, y, net_type="vgg"))))
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(os.path.join(a.save_dir, cam["img_name"] + ".png"))
    mean = {k: float(np.mean([r[k] for r in res])) for k in ("psnr", "ssim", "lpips")}
    tag = a.tag or f"{a.visibility}{'_sh' if a.sh else ''}"
    os.makedirs(a.out_dir, exist_ok=True)
    run = os.path.basename(a.model.rstrip("/"))
    json.dump(dict(model=a.model, mean=mean, views=res), open(os.path.join(a.out_dir, f"{run}__{tag}.json"), "w"), indent=1)
    print(f"{os.path.basename(a.model.rstrip('/'))} [{tag}] {len(res)} views: PSNR {mean['psnr']:.2f} SSIM {mean['ssim']:.3f} LPIPS {mean['lpips']:.3f}")


if __name__ == "__main__":
    main()
