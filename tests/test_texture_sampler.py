"""Step 2 of notes/plan.md: test cuda_rasterizer/texture_sampler.h on its own.

Reference: torch.nn.functional.grid_sample(mode='bilinear', padding_mode='border',
align_corners=True), with grid = 2 * uv - 1 (our uv is in [0, 1], grid_sample's in [-1, 1]).

Usage (from combined/, venv active):  python tests/test_texture_sampler.py
The CUDA test extension is compiled on the first run into tests/build/ (takes ~1 min).
"""
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER_DIR = os.path.join(HERE, "..", "submodules", "diff-triangle2-rasterization", "cuda_rasterizer")
BUILD = os.path.join(HERE, "build", "texture_sampler_ext")
os.makedirs(BUILD, exist_ok=True)
ext = load(name="texture_sampler_ext", sources=[os.path.join(HERE, "texture_sampler_ext.cu")],
           extra_include_paths=[HEADER_DIR], build_directory=BUILD,
           extra_cuda_cflags=["--use_fast_math"],  # same flag as the rasterizer (setup.py)
           verbose=False)

dev = "cuda"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")


def torch_reference(tex, uv, tex_id):
    """Per-sample grid_sample; returns values [N, C] and a function for the gradients."""
    tex = tex.clone().requires_grad_(True)
    uv = uv.clone().requires_grad_(True)
    grid = (2 * uv - 1).view(-1, 1, 1, 2)                      # x = u (column), y = v (row)
    out = F.grid_sample(tex[tex_id.long()], grid, mode="bilinear", padding_mode="border",
                        align_corners=True)[:, :, 0, 0]       # [N, C]

    def grads(dout):
        g_tex, g_uv = torch.autograd.grad(out, (tex, uv), dout, retain_graph=True)
        return g_tex, g_uv
    return out.detach(), grads


def uv_sets(N, S, g):
    """Coordinate groups that exercise the edge handling."""
    r = lambda *s: torch.rand(*s, device=dev, generator=g)
    inside = r(N, 2)
    edge = r(N, 2)
    k = torch.randint(0, 4, (N,), device=dev, generator=g)
    edge[k == 0, 0] = 0.0
    edge[k == 1, 0] = 1.0
    edge[k == 2, 1] = 0.0
    edge[k == 3, 1] = 1.0
    corners = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], device=dev).repeat(N // 4, 1)
    outside = r(N, 2) * 1.4 - 0.2                               # [-0.2, 1.2], mixes inside and outside
    centres = torch.randint(0, S, (N, 2), device=dev, generator=g).float() / (S - 1)
    return dict(inside=inside, edge=edge, corners=corners, outside=outside, texel_centres=centres)


def main():
    g = torch.Generator(device=dev).manual_seed(0)
    T, C, N = 7, 3, 20000
    for S in (4, 8, 16):
        print(f"\nS = {S}")
        tex = torch.randn(T, C, S, S, device=dev, generator=g)
        for name, uv in uv_sets(N, S, g).items():
            uv = uv.contiguous()
            tex_id = torch.randint(0, T, (uv.shape[0],), device=dev, generator=g, dtype=torch.int32)
            ours = ext.forward(tex, uv, tex_id)
            ref, ref_grads = torch_reference(tex, uv, tex_id)
            err = (ours - ref).abs().max().item()
            check(f"{name:13s} forward", err <= 1e-5, f"max |ours - torch| = {err:.1e}")

            dout = torch.randn(uv.shape[0], C, device=dev, generator=g)
            d_tex, d_uv = ext.backward(tex, uv, tex_id, dout)
            r_tex, r_uv = ref_grads(dout)
            rel = ((d_tex - r_tex).abs().max() / r_tex.abs().max()).item()
            check(f"{name:13s} dL/dtexture", rel <= 1e-5, f"max rel diff = {rel:.1e}")
            # dL/duv: PyTorch treats a coordinate exactly on the border specially (one-sided slope
            # towards a zero texel), and dL/duv is not used for training (decision 2), so compare
            # it where the coordinate is strictly inside or clearly outside the texture.
            if name in ("inside", "outside"):
                x = uv * (S - 1)
                m = ((x > 1e-3) & (x < S - 1 - 1e-3)) | (x < -1e-3) | (x > S - 1 + 1e-3)
                rel_uv = ((d_uv - r_uv).abs()[m].max() / r_uv.abs()[m].max().clamp_min(1e-12)).item()
                check(f"{name:13s} dL/duv", rel_uv <= 1e-4, f"max rel diff = {rel_uv:.1e} ({m.float().mean():.0%} of values)")

        # properties that the triangle textures rely on
        uv = (torch.rand(N, 2, device=dev, generator=g) * 1.4 - 0.2).contiguous()
        tid = torch.zeros(N, dtype=torch.int32, device=dev)
        zero = ext.forward(torch.zeros(1, C, S, S, device=dev), uv, tid)
        check("zero texture -> exactly 0 (zero-init = plain TS+)", torch.equal(zero, torch.zeros_like(zero)))
        const = ext.forward(torch.full((1, C, S, S), 0.37, device=dev), uv, tid)
        err = (const - 0.37).abs().max().item()
        check("constant texture -> same constant everywhere, incl. edges/outside (no fading)", err <= 1e-6, f"max err {err:.1e}")
        ys, xs = torch.meshgrid(torch.arange(S, device=dev).float(), torch.arange(S, device=dev).float(), indexing="ij")
        lin = (0.3 * xs - 0.2 * ys + 0.5).expand(1, C, S, S).contiguous()
        out = ext.forward(lin, uv, tid)[:, 0]
        uc = uv.clamp(0, 1) * (S - 1)
        err = (out - (0.3 * uc[:, 0] - 0.2 * uc[:, 1] + 0.5)).abs().max().item()
        check("linear texture reproduced exactly (clamped outside)", err <= 1e-5, f"max err {err:.1e}")

    print("\nRESULT:", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)} checks)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
