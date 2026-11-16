"""Step 4 of notes/plan.md: the rasterizer draws the per-triangle texture (forward only).

Checks on the local fixture (plain TS+ garden images_8 model):
  1. zero texture  -> image + all aux outputs bit-identical to "no texture" and to the ORIGINAL
     TS+ rasterizer (the key test: zero-initialised textures start exactly as plain TS+);
  2. constant texture c on every triangle -> image = no-texture image + c * alpha (alpha = 1 - T_final);
  3. UV mapping: on a copy of the model where every triangle has its own 3 vertices, a texture that
     is linear in (u, v) must give the same image as adding the corner values to the vertex colours
     (checks texture orientation, v0/v1/v2 -> (0,0)/(1,0)/(0,1), and that triangle i reads texture i);
  4. a wrong texture shape is rejected; backward still works with a zero texture (texture gets no
     gradient yet) and refuses a non-zero texture (gradients would be wrong until step 5);
  5. writes a checkerboard picture for a visual check.

Usage (from combined/, venv active):
  python tests/test_texture_forward.py --model <dir with point_cloud_state_dict.pt> --cameras <cameras.json> \
      [--views 0 50 100] [--downscale 8] [--scaling 4] [--out <dir for the picture>]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compare_rasterizers import load_model, camera_from_json  # noqa: E402
import diff_triangle_rasterization as orig  # noqa: E402
import diff_textured_triangle_rasterization as ours  # noqa: E402

dev = "cuda"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")


def render(pkg, m, cam, scaling, texture=None, colors_precomp=None, grad=False):
    settings = pkg.TriangleRasterizationSettings(
        image_height=cam["H"] * scaling, image_width=cam["W"] * scaling,
        tanfovx=cam["tanfovx"], tanfovy=cam["tanfovy"],
        bg=torch.zeros(3, device=dev), scale_modifier=1.0,
        viewmatrix=cam["viewmatrix"], projmatrix=cam["projmatrix"],
        sh_degree=m["sh_degree"], campos=cam["campos"], prefiltered=False, debug=False)
    rast = pkg.TriangleRasterizer(raster_settings=settings)
    kw = dict(vertices=m["vertices"].detach().clone().requires_grad_(grad),
              triangles_indices=m["triangles_indices"], vertex_weights=m["vertex_weights"],
              sigma=m["sigma"], scaling=torch.zeros(m["triangles_indices"].shape[0], device=dev))
    if colors_precomp is not None:
        kw.update(shs=None, colors_precomp=colors_precomp)
    else:
        kw.update(shs=m["shs"], colors_precomp=None)
    if texture is not None:
        kw["texture"] = texture
    image, radii, _, allmap, max_blending = rast(**kw)
    return dict(image=image, radii=radii, allmap=allmap, max_blending=max_blending, vertices=kw["vertices"])


def same_outputs(a, b):
    """Bit-identical image, radii, max_blending, aux channels 0-5, and aux channel 6 (triangle id)
    where it is defined (alpha > 0.5, see compare_rasterizers.py)."""
    valid = a["allmap"][1] > 0.5
    return (torch.equal(a["image"], b["image"]) and torch.equal(a["radii"], b["radii"])
            and torch.equal(a["max_blending"], b["max_blending"])
            and torch.equal(a["allmap"][:6], b["allmap"][:6])
            and torch.equal(a["allmap"][6][valid], b["allmap"][6][valid]))


def unshare_vertices(m):
    """Same geometry, but every triangle gets its own 3 vertices (triangle i -> vertices 3i, 3i+1, 3i+2)."""
    idx = m["triangles_indices"].long()                                  # [T, 3]
    T = idx.shape[0]
    return dict(m, vertices=m["vertices"][idx].reshape(-1, 3).contiguous(),
                vertex_weights=m["vertex_weights"][idx].reshape(-1).contiguous(),
                shs=m["shs"][idx].reshape(T * 3, *m["shs"].shape[1:]).contiguous(),
                triangles_indices=torch.arange(3 * T, device=dev, dtype=torch.int32).view(T, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--views", type=int, nargs="+", default=[0, 50, 100])
    ap.add_argument("--downscale", type=int, default=8)
    ap.add_argument("--scaling", type=int, default=4, help="TS+ supersampling factor")
    ap.add_argument("--S", type=int, default=8, help="texture resolution")
    ap.add_argument("--out", default=None, help="where to write the checkerboard picture")
    args = ap.parse_args()

    m = load_model(args.model)
    T, S = m["triangles_indices"].shape[0], args.S
    cams = json.load(open(args.cameras))
    print(f"model: {m['vertices'].shape[0]} vertices, {T} triangles, texture {S}x{S}")
    g = torch.Generator(device=dev).manual_seed(0)

    with torch.no_grad():
        for v in args.views:
            cam = camera_from_json(cams[v], args.downscale)
            print(f"\nview {v} ({cam['W'] * args.scaling}x{cam['H'] * args.scaling})")
            base = render(ours, m, cam, args.scaling)
            ref = render(orig, m, cam, args.scaling)

            # 1. zero texture
            zero = render(ours, m, cam, args.scaling, texture=torch.zeros(T, 3, S, S, device=dev))
            check("zero texture == no texture, bit-identical (image + aux)", same_outputs(zero, base))
            check("zero texture == ORIGINAL TS+ rasterizer, bit-identical", same_outputs(zero, ref))

            # 2. constant texture
            c = torch.tensor([0.3, -0.2, 0.1], device=dev)
            const = render(ours, m, cam, args.scaling, texture=c.view(1, 3, 1, 1).expand(T, 3, S, S).contiguous())
            alpha = base["allmap"][1]
            err = (const["image"] - (base["image"] + c.view(3, 1, 1) * alpha)).abs().max().item()
            check("constant texture c -> image + c * alpha", err <= 1e-5, f"(max err {err:.1e})")
            check("constant texture: geometry outputs unchanged (aux 0-5, radii, max_blending)",
                  torch.equal(const["allmap"][:6], base["allmap"][:6]) and torch.equal(const["radii"], base["radii"])
                  and torch.equal(const["max_blending"], base["max_blending"]))

            # 3. UV mapping with a texture that is linear in (u, v)
            mu = unshare_vertices(m)
            corner = torch.rand(T, 2, 3, device=dev, generator=g) - 0.5      # value at v1 (u=1) and v2 (v=1)
            lin = torch.linspace(0, 1, S, device=dev)
            # texture[t, c, y, x] = corner_u[t,c] * x/(S-1) + corner_v[t,c] * y/(S-1)
            tex = (corner[:, 0, :, None, None] * lin.view(1, 1, 1, S)
                   + corner[:, 1, :, None, None] * lin.view(1, 1, S, 1)).contiguous()
            base_col = torch.rand(3 * T, 3, device=dev, generator=g) * 0.5
            vert_col = base_col.clone().view(T, 3, 3)
            vert_col[:, 1] += corner[:, 0]
            vert_col[:, 2] += corner[:, 1]
            a = render(ours, mu, cam, args.scaling, colors_precomp=base_col, texture=tex)
            b = render(ours, mu, cam, args.scaling, colors_precomp=vert_col.view(-1, 3).contiguous())
            d = (a["image"] - b["image"]).abs()
            err, err99 = d.max().item(), d.flatten().quantile(0.99).item() if d.numel() < 2**24 else float("nan")
            # max error is a few pixels where TS+'s barycentrics are slightly outside the triangle
            # (the texture clamps them onto it, the vertex colours extrapolate)
            check("linear texture == same values added to the vertex colours (UV orientation, triangle i -> texture i)",
                  err <= 1e-3 and err99 <= 1e-5, f"(max err {err:.1e}, 99% of values <= {err99:.1e})")
            same_geo = render(ours, mu, cam, args.scaling, colors_precomp=base_col)
            check("  (sanity) unshared-vertex copy renders the same geometry",
                  torch.equal(same_geo["allmap"][1], base["allmap"][1]))

    # 4. input checks and backward
    print("\nAPI")
    cam = camera_from_json(cams[args.views[0]], args.downscale)
    for bad, why in [(torch.zeros(T - 1, 3, S, S, device=dev), "wrong triangle count"),
                     (torch.zeros(T, 4, S, S, device=dev), "4 channels"),
                     (torch.zeros(T, 3, S, S + 1, device=dev), "not square"),
                     (torch.zeros(T, 3, 1, 1, device=dev), "S = 1"),
                     (torch.zeros(T, 3, S, S, device=dev, dtype=torch.float64), "float64")]:
        try:
            render(ours, m, cam, args.scaling, texture=bad)
            check(f"rejects texture with {why}", False)
        except RuntimeError:
            check(f"rejects texture with {why}", True)

    tex0 = torch.zeros(T, 3, S, S, device=dev, requires_grad=True)
    out = render(ours, m, cam, args.scaling, texture=tex0, grad=True)
    out["image"].sum().backward()
    check("backward runs with a zero texture; vertices get a gradient, texture none (step 5)",
          out["vertices"].grad is not None and tex0.grad is None)
    tex1 = torch.full((T, 3, S, S), 0.1, device=dev, requires_grad=True)
    out = render(ours, m, cam, args.scaling, texture=tex1, grad=True)
    try:
        out["image"].sum().backward()
        check("backward refuses a non-zero texture", False)
    except NotImplementedError:
        check("backward refuses a non-zero texture", True)

    # 5. checkerboard picture (on 30% of the triangles, others no texture)
    if args.out:
        import torchvision
        os.makedirs(args.out, exist_ok=True)
        ys, xs = torch.meshgrid(torch.arange(S, device=dev), torch.arange(S, device=dev), indexing="ij")
        board = (((xs // max(S // 4, 1)) + (ys // max(S // 4, 1))) % 2).float() * 0.8 - 0.4   # +-0.4
        pick = torch.rand(T, device=dev, generator=g) < 0.3
        tex = torch.zeros(T, 3, S, S, device=dev)
        tex[pick] = board.expand(int(pick.sum()), 3, S, S)
        with torch.no_grad():
            img_t = render(ours, m, cam, args.scaling, texture=tex)["image"]
            img_b = render(ours, m, cam, args.scaling)["image"]
        pic = torch.cat([img_b, img_t], dim=2).clamp(0, 1)
        path = os.path.join(args.out, f"checkerboard_view{args.views[0]}.png")
        torchvision.utils.save_image(pic, path)
        print(f"\nwrote {path} (left: plain, right: checkerboard on 30% of triangles)")

    print("\nRESULT:", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)} checks)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
