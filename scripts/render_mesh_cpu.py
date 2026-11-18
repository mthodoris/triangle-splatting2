#!/usr/bin/env python3
"""CPU z-buffer render of an exported textured mesh (as a game engine draws it), for checking exports.

Opaque triangles, nearest wins (z-buffer), perspective-correct UVs, bilinear atlas lookup with
pixel-centre convention, no mipmaps. Renders one camera from TS+'s cameras.json and compares
against the CUDA render and the photo.

Usage: python scripts/render_mesh_cpu.py <model_dir> <atlas.png> --cam DSC07956 --render r.png --gt g.png --out o.png
       (UVs are rebuilt with export_textured_mesh.bake using the same k as the atlas)
"""
import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import export_textured_mesh as E

Image.MAX_IMAGE_PIXELS = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model"); p.add_argument("atlas")
    p.add_argument("--cam", required=True); p.add_argument("--width", type=int, default=1297)
    p.add_argument("--mode", default="atlas", choices=["atlas", "exact", "vcol"],
                   help="atlas: sample the exported atlas; exact: CUDA formula from the raw texture "
                        "(isolates z-buffer visibility); vcol: vertex colours only")
    p.add_argument("--visibility", default="zbuf", choices=["zbuf", "centre"],
                   help="zbuf: nearest surface per pixel (game engine); centre: first triangle in order of "
                        "centre depth (TS+ CUDA sort, alpha 0.99 -> first one wins)")
    p.add_argument("--sh", action="store_true", help="full view-dependent SH colour for this camera (as CUDA)")
    p.add_argument("--render"); p.add_argument("--gt"); p.add_argument("--out", required=True)
    a = p.parse_args()

    try:
        verts, faces, dc, tex = E.load(a.model)
    except SystemExit:   # plain TS+ run: no texture
        sd = torch.load(os.path.join(a.model, "point_cloud/iteration_30000/point_cloud_state_dict.pt"), map_location="cpu", weights_only=False)
        verts = sd["triangles_points"].detach().numpy(); faces = sd["_triangle_indices"].long().numpy()
        dc = sd["features_dc"].detach().numpy()[:, 0]; tex = np.zeros((len(faces), 3, 2, 2), np.float32); a.mode = "vcol"
    T, _, S, _ = tex.shape
    atlas = np.asarray(Image.open(a.atlas)).astype(np.float32) / 255.0 if a.mode == "atlas" else np.zeros((1, 1, 3))
    Ww, Hh = E.atlas_layout(T, (S - 1) * E.auto_upsample(T, S) + 1)[1:] if a.mode != "atlas" else atlas.shape[1::-1]
    # find k from the atlas size, rebuild UVs
    k = next(k for k in range(1, 9) if E.atlas_layout(T, (S - 1) * k + 1)[1:] == (Ww, Hh))
    tpr, _, _ = E.atlas_layout(T, (S - 1) * k + 1)
    P = (S - 1) * k + 1
    tx, ty = np.arange(T) % tpr, np.arange(T) // tpr
    px = np.stack([tx * P + 0.5, tx * P + (P - 1) + 0.5, tx * P + 0.5], 1)
    py = np.stack([ty * P + 0.5, ty * P + 0.5, ty * P + (P - 1) + 0.5], 1)
    uvpix = torch.tensor(np.stack([px, py], -1), dtype=torch.float64)           # atlas pixel coords [T,3,2]

    cam = next(c for c in json.load(open(os.path.join(a.model, "cameras.json"))) if c["img_name"] == a.cam)
    W = a.width; H = round(cam["height"] * W / cam["width"]); s = W / cam["width"]
    fx, fy = cam["fx"] * s, cam["fy"] * s
    R = torch.tensor(cam["rotation"], dtype=torch.float64); t = torch.tensor(cam["position"], dtype=torch.float64)
    pc = (torch.tensor(verts, dtype=torch.float64) - t) @ R                     # camera coords, z forward
    tri = pc[torch.tensor(faces)]                                               # [T,3,3]
    Z = tri[..., 2]
    keep = (Z > 0.01).all(1)
    xs = tri[..., 0] / Z * fx + W / 2; ys = tri[..., 1] / Z * fy + H / 2
    x0 = xs.min(1).values.floor().clamp(0, W - 1).long(); x1 = xs.max(1).values.ceil().clamp(0, W - 1).long()
    y0 = ys.min(1).values.floor().clamp(0, H - 1).long(); y1 = ys.max(1).values.ceil().clamp(0, H - 1).long()
    keep &= (xs.max(1).values >= 0) & (xs.min(1).values < W) & (ys.max(1).values >= 0) & (ys.min(1).values < H)
    ids = torch.nonzero(keep)[:, 0]
    area = ((x1 - x0 + 1) * (y1 - y0 + 1))[ids]

    zbuf = torch.full((H * W,), float("inf"), dtype=torch.float64)
    win = torch.full((H * W,), -1, dtype=torch.long)
    wbuf = torch.zeros((H * W, 3), dtype=torch.float64)
    budget, start, csum = 20_000_000, 0, torch.cumsum(area, 0)
    while start < len(ids):
        base = csum[start - 1] if start > 0 else 0
        end = max(start + 1, int(torch.searchsorted(csum, base + budget)))
        tid = ids[start:end]; ar = area[start:end]
        rep = torch.repeat_interleave(torch.arange(len(tid)), ar)
        off = torch.arange(int(ar.sum())) - torch.repeat_interleave(torch.cumsum(ar, 0) - ar, ar)
        bw = (x1 - x0 + 1)[tid][rep]
        X = x0[tid][rep] + off % bw; Y = y0[tid][rep] + off // bw
        fxp, fyp = X.double() + 0.5, Y.double() + 0.5
        ax, ay = xs[tid][rep], ys[tid][rep]
        d = (ax[:, 1] - ax[:, 0]) * (ay[:, 2] - ay[:, 0]) - (ax[:, 2] - ax[:, 0]) * (ay[:, 1] - ay[:, 0])
        b1 = ((fxp - ax[:, 0]) * (ay[:, 2] - ay[:, 0]) - (ax[:, 2] - ax[:, 0]) * (fyp - ay[:, 0])) / d
        b2 = ((ax[:, 1] - ax[:, 0]) * (fyp - ay[:, 0]) - (fxp - ax[:, 0]) * (ay[:, 1] - ay[:, 0])) / d
        b0 = 1 - b1 - b2
        ins = (b0 >= 0) & (b1 >= 0) & (b2 >= 0) & (d.abs() > 1e-12)
        b = torch.stack([b0, b1, b2], 1)[ins]; tt = tid[rep][ins]; pix = (Y * W + X)[ins]
        bz = b / Z[tt]; depth = 1.0 / bz.sum(1); bp = bz * depth[:, None]       # perspective-correct weights
        key = depth if a.visibility == "zbuf" else Z[tt].mean(1)
        zmin = torch.full((H * W,), float("inf"), dtype=torch.float64).scatter_reduce(0, pix, key, "amin")
        better = (key == zmin[pix]) & (key < zbuf[pix])
        zbuf[pix[better]] = key[better]; win[pix[better]] = tt[better]; wbuf[pix[better]] = bp[better]
        start = end
    img = np.zeros((H * W, 3), np.float32)
    m = win >= 0
    uv = (wbuf[m][:, :, None] * uvpix[win[m]]).sum(1).numpy()
    if a.mode == "atlas":
        img[m.numpy()] = E.bilinear_clamp(atlas, uv[:, 0] - 0.5, uv[:, 1] - 0.5)
    else:
        col = E.vertex_colours(dc); tw = win[m].numpy()
        if a.sh:
            from utils.sh_utils import eval_sh
            sd = torch.load(os.path.join(a.model, "point_cloud/iteration_30000/point_cloud_state_dict.pt"), map_location="cpu", weights_only=False)
            shs = torch.cat([sd["features_dc"], sd["features_rest"]], 1).detach().float().transpose(1, 2)   # [V,3,16]
            dirs = torch.tensor(verts) - t.float(); dirs = dirs / dirs.norm(dim=1, keepdim=True)
            col = torch.clamp_min(eval_sh(sd["active_sh_degree"], shs, dirs) + 0.5, 0.0).numpy(); w = wbuf[m].numpy()
        c = sum(w[:, j:j + 1] * col[faces[tw, j]] for j in range(3))
        if a.mode == "exact":   # CUDA: uv = (b1, b2) -> weights of vertex 1, 2; clamp to the triangle; bilinear
            u, v = np.maximum(w[:, 1], 0), np.maximum(w[:, 2], 0); sm = np.maximum(u + v, 1)
            x, y = u / sm * (S - 1), v / sm * (S - 1)
            x0_ = np.minimum(np.floor(x).astype(int), S - 2); y0_ = np.minimum(np.floor(y).astype(int), S - 2)
            fx_ = (x - x0_)[:, None]; fy_ = (y - y0_)[:, None]; tt_ = tex[tw]; r_ = np.arange(len(tw))
            c = c + ((1 - fx_) * (1 - fy_) * tt_[r_, :, y0_, x0_] + fx_ * (1 - fy_) * tt_[r_, :, y0_, x0_ + 1]
                     + (1 - fx_) * fy_ * tt_[r_, :, y0_ + 1, x0_] + fx_ * fy_ * tt_[r_, :, y0_ + 1, x0_ + 1])
        img[m.numpy()] = np.clip(c, 0, 1)
    img = img.reshape(H, W, 3)
    print(f"{a.cam}: {W}x{H}, covered {m.float().mean():.1%}")
    rows = [img]
    for name, f in (("CUDA render", a.render), ("photo", a.gt)):
        if f:
            ref = np.asarray(Image.open(f).convert("RGB").resize((W, H))).astype(np.float32) / 255
            print(f"mesh render vs {name}: PSNR {-10*np.log10(((img-ref)**2).mean()):.2f} dB")
            rows.append(ref)
    Image.fromarray((np.concatenate(rows, 0) * 255 + 0.5).astype(np.uint8)).save(a.out)


if __name__ == "__main__":
    main()
