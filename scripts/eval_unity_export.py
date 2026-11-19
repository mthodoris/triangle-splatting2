#!/usr/bin/env python3
"""Check an export_unity.py folder (plan step 9.4): render render.tsm + atlas.png exactly as the Unity shader
TriSplat/SHAtlas does, on the test views, and report PSNR / SSIM / LPIPS. Should be close to
eval_engine_view.py --sh (zbuf_sh) of the same model.

Everything is read from the exported files (positions in Unity's frame, 8-bit atlas, float16 SH); only the cameras
come from the model's cameras.json and are moved into Unity's frame with meta.json. The view direction is computed
in Unity's frame and turned into the scene frame with meta.json's scene_dir_from_unity, as the shader does.
Visibility: z-buffer (eval_engine_view.render). Interpolation: perspective-correct (as a GPU).

Usage: python scripts/eval_unity_export.py EXPORT_DIR --images <dataset>/images_4 [--views 0,8] [--save_dir dir]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from eval_engine_view import render  # noqa: E402
from utils.sh_utils import eval_sh  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402
from lpipsPyTorch import lpips  # noqa: E402

SH_C0 = 0.28209479177387814


def read_tsm(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"TSM1"
        nV, T, K, aw, ah = struct.unpack("<5i", f.read(20))
        pos = np.frombuffer(f.read(T * 9 * 4), "<f4").reshape(T, 3, 3)
        uv = np.frombuffer(f.read(T * 6 * 4), "<f4").reshape(T, 3, 2)
        src = np.frombuffer(f.read(T * 3 * 4), "<i4").reshape(T, 3)
        sh = np.frombuffer(f.read(nV * K * 3 * 2), "<f2").reshape(nV, K, 3)
    return pos, uv, src, sh, (aw, ah)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("export"); p.add_argument("--images", required=True)
    p.add_argument("--views", default=""); p.add_argument("--save_dir")
    a = p.parse_args()
    dev = "cuda"
    meta = json.load(open(os.path.join(a.export, "meta.json")))
    pos, uv, src, sh, _ = read_tsm(os.path.join(a.export, "render.tsm"))
    T = len(pos)
    Image.MAX_IMAGE_PIXELS = None
    atlas = torch.tensor(np.asarray(Image.open(os.path.join(a.export, "atlas.png")).convert("RGB")), device=dev).float() / 255
    AH, AW, _ = atlas.shape
    A = np.array(meta["unity_from_scene_A"]); b = np.array(meta["unity_from_scene_b"])
    Mdir = torch.tensor(meta["scene_dir_from_unity"], dtype=torch.float32, device=dev)
    deg = meta["sh_degree"]

    # Geometry in Unity's frame; cameras moved there too (rotation R -> A R / s, a reflection: fine for projection
    # because render() only uses R to get camera-space coordinates).
    s = meta["units_to_m"]
    m = dict(verts=torch.tensor(pos.reshape(-1, 3), dtype=torch.float64, device=dev),
             faces=torch.arange(3 * T, device=dev).reshape(T, 3),
             shs=torch.zeros(3 * T, 3, 1, device=dev), deg=0, tex=None)
    uvt = torch.tensor(uv.reshape(-1, 2), device=dev)
    srct = torch.tensor(src.reshape(-1).astype(np.int64), device=dev)
    sht = torch.tensor(sh.astype(np.float32), device=dev).transpose(1, 2)          # [V, 3, K]
    c0 = torch.clamp_min(SH_C0 * sht[:, :, 0] + 0.5, 0.0)                            # [V, 3]

    cams = sorted(json.load(open(os.path.join(meta["model"], "cameras.json"))), key=lambda c: c["img_name"])[::8]
    if a.views:
        cams = [cams[int(i)] for i in a.views.split(",")]
    res = []
    for cam in cams:
        gt = Image.open(os.path.join(a.images, cam["img_name"] + ".JPG")).convert("RGB")
        W, H = gt.size
        gt = torch.tensor(np.asarray(gt), dtype=torch.float32, device=dev) / 255
        cu = dict(cam)
        cu["position"] = (A @ np.array(cam["position"]) + b).tolist()
        cu["rotation"] = (A @ np.array(cam["rotation"]) / s).tolist()
        out = {}
        render(m, cu, W, H, "zbuf", False, out=out)
        win, w = out["win"], out["wts"].float()
        msk = win >= 0; tw = win[msk]; w = w[msk]
        # vertex stage: per corner r = c - c0, view direction in Unity frame -> scene frame
        corner = torch.stack([3 * tw, 3 * tw + 1, 3 * tw + 2], 1)                   # [n, 3] soup vertex ids
        camu = torch.tensor(cu["position"], dtype=torch.float32, device=dev)
        vu = m["verts"].float()
        cid = torch.unique(corner)
        du = vu[cid] - camu; du = du / du.norm(dim=1, keepdim=True)
        ds = du @ Mdir.T
        sv = srct[cid]
        r_c = torch.zeros(3 * T, 3, device=dev)
        r_c[cid] = torch.clamp_min(eval_sh(deg, sht[sv], ds) + 0.5, 0.0) - c0[sv]
        r = (w[:, :, None] * r_c[corner]).sum(1)
        U = (w[:, :, None] * uvt[corner]).sum(1)
        # fragment stage: bilinear atlas sample (pixel centres, clamp), + r, saturate
        x = (U[:, 0] * AW - 0.5).clamp(0, AW - 1); y = ((1 - U[:, 1]) * AH - 0.5).clamp(0, AH - 1)
        x0 = torch.clamp(x.floor().long(), max=AW - 2); y0 = torch.clamp(y.floor().long(), max=AH - 2)
        fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
        tex = ((1 - fx) * (1 - fy) * atlas[y0, x0] + fx * (1 - fy) * atlas[y0, x0 + 1]
               + (1 - fx) * fy * atlas[y0 + 1, x0] + fx * fy * atlas[y0 + 1, x0 + 1])
        img = torch.zeros(H * W, 3, device=dev)
        img[msk] = (tex + r).clamp(0, 1)
        img = img.reshape(H, W, 3)
        xi, yi = img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
        psnr = float(-10 * torch.log10(((xi - yi) ** 2).mean()))
        res.append(dict(name=cam["img_name"], psnr=psnr, ssim=float(ssim(xi, yi)), lpips=float(lpips(xi, yi, net_type="vgg"))))
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(os.path.join(a.save_dir, cam["img_name"] + ".png"))
    mean = {k: float(np.mean([r_[k] for r_ in res])) for k in ("psnr", "ssim", "lpips")}
    json.dump(dict(export=a.export, mean=mean, views=res), open(os.path.join(a.export, "eval_unity.json"), "w"), indent=1)
    print(f"{a.export} [unity shader] {len(res)} views: PSNR {mean['psnr']:.2f} SSIM {mean['ssim']:.3f} LPIPS {mean['lpips']:.3f}")


if __name__ == "__main__":
    main()
