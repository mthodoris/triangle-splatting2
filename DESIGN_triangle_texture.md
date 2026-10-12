# Design: per-triangle texture for the clean mesh (devel4)

Status: proposal, not implemented. Written 2026-10-12.

## Why

The clean mesh (Large Steps, sharp triangles, render scale 2–4, no free triangles) plateaus at about 20.9 dB on garden, while plain triangle splatting reaches 25.1 dB (as a soup). A per-region error split on the 24 test views shows where the gap is:

| Region | Share of pixels | Clean mesh | Plain TS | Gap | Share of the clean mesh's error |
|---|---|---|---|---|---|
| Table top | 13% | 21.6 dB | 24.9 dB | −3.3 | 8% |
| Lawn, paving, bushes | 80% | 20.1 dB | 24.9 dB | −4.8 | 71% |
| Depth edges | 7% | 14.9 dB | 20.6 dB | −5.7 | 21% |

The error maps show the 71% is fine texture (grass, gravel, foliage). Today each pixel's colour is the barycentric blend of its triangle's three vertex colours (`forward.cu`, colour interpolation), so a triangle can only show a smooth gradient. Doubling the vertices by refinement improved SSIM/LPIPS but not PSNR. The depth-edge part is geometry and is addressed separately (normal-only Large Steps runs).

## Representation

Each mesh triangle gets a small grid of colour samples on a regular barycentric lattice of level `n`:

- Samples at barycentric positions `(i/n, j/n, k/n)` with `i + j + k = n`: K = (n+1)(n+2)/2 samples per face (n=2: 6, n=3: 10, n=4: 15).
- The lattice splits the triangle into n² sub-triangles. A pixel's texture value is the linear interpolation of the 3 samples of the sub-triangle it falls in.
- **Colour = interpolated vertex colour (SH, view dependent) + interpolated texture residual (RGB, view independent).** Texels start at 0, so training starts exactly as today and the vertex colours keep carrying the low-frequency, view-dependent part.

Two variants:

- **v1, per-face texels (recommended first):** each face owns its K samples, indexed `face_id * K + local`. Simple indexing in CUDA, easy to prune, resample and save. Neighbouring faces do not share edge samples, so small colour seams across edges are possible; triangles are only a few pixels wide, so this is likely minor.
- **v2, shared edge and vertex samples ("Mesh Colors", Yuksel et al. 2010):** samples on vertices are the mesh vertices, samples on edges are shared by the two faces, interior samples per face. Seamless by construction, about 40% less memory, but needs an edge-index table kept in sync through refinement. Only if v1 shows seams.

Lattice lookup for a pixel with barycentrics `(b0, b1, b2)` (weights of vertices 0, 1, 2):
- `u = b1 * n`, `v = b2 * n`, `i = floor(u)`, `j = floor(v)`, `fu = u - i`, `fv = v - j` (clamp so `i + j <= n - 1`).
- If `fu + fv <= 1`: lower sub-triangle with samples `(i, j)`, `(i+1, j)`, `(i, j+1)` and weights `(1 - fu - fv, fu, fv)`; otherwise upper sub-triangle `(i+1, j+1)`, `(i, j+1)`, `(i+1, j)` with weights `(fu + fv - 1, 1 - fu, 1 - fv)`.
- Local index of sample `(i, j)` (row i, i + j <= n): `i * (n + 1) - i * (i - 1) / 2 + j`.

## Changes

1. **CUDA forward** (`renderCUDA` in `forward.cu`): new inputs `face_tex` (float, F×K×3) and `tex_res` (n; 0 = off, identical to today). After computing the vertex-interpolated colour, add the texture value from the lattice lookup above.
2. **CUDA backward** (`backward.cu`): include the texture value in `interp_color` (needed for the alpha gradient); add `dL_dface_tex` with `atomicAdd` into the 3 samples × 3 channels, weight `dchannel_dcolor * dL_dpixel * lattice_weight`. v1 leaves the texture's contribution out of the gradient with respect to the vertex positions (barycentric derivative); geometry moves very little with Large Steps, and this can be added later.
3. **Bindings**: `rasterize_points.cu` (forward/backward signatures, returning `grad_face_tex`), `diff_triangle_rasterization/__init__.py` (autograd function and `TriangleRasterizer.forward(..., face_tex=None, tex_res=0)`), `triangle_renderer/__init__.py` (pass `pc._face_tex` and `pc.tex_res` for the mesh group; free triangles keep `tex_res` texels at 0 or a separate flag).
4. **Model** (`scene/triangle_model.py`): `_face_tex` parameter (F×K×3) in the optimizer (`--tex_lr`), created with zeros in `create_from_mesh`; indexed by `prune_triangles` masks; appended in `densification_postfix`; on refinement, each child's samples are computed by evaluating the parent's texture at the child's lattice positions (exact, done in Python); saved and loaded with the checkpoint.
5. **Training**: flags `--tex_res` (default 0 = off), `--tex_lr`, optional `--lambda_tex_tv` (smoothness of texels within a face, against noise), optional `--tex_from` (start texture training after the vertex colours settle, e.g. 5k).
6. **Export**: `mesh.ply` keeps the base mesh; a second file subdivides every face to the lattice (n² sub-triangles each) with per-sample colours (vertex colour + texel), so the exported mesh carries the texture as dense vertex colours. A UV-atlas export can come later.

## Memory and speed

- Texels: F × K × 3 floats. Full input mesh (2.9M faces) at n=3: 87M floats ≈ 350 MB, plus Adam state ≈ 1 GB total. The 600k-face coarse-to-fine meshes (~1–2.5M faces): 0.2–0.4 GB. Fits on the cluster's 32 GB GPUs.
- Per pixel-triangle contribution: 3 extra reads in the forward pass, 9 extra atomic adds in the backward pass. Expect a modest slowdown (10–30%).

## Tests before any cluster run

1. **Identity:** `tex_res = 0` and `tex_res = n` with zero texels both reproduce today's renders exactly (bit-for-bit for 0).
2. **Gradient check:** finite differences on a 2-triangle scene for texels and vertex colours.
3. **Overfit one view:** with texture the clean mesh must reach a clearly higher PSNR on a single training view than without (shows the capacity is there).
4. **Short garden run** (3k iterations, 600k mesh, scale 2): PSNR, SSIM, LPIPS and seams by eye; then cluster runs at n = 2, 3, 4.

## Code location and build (decision needed)

`submodules/diff-triangle2-rasterization` is a git submodule of the upstream repository (github.com/trianglesplatting2/diff-triangle2-rasterization), and the Docker image builds it from upstream. Options:

- **A (recommended):** copy the rasterizer into the main repo as a tracked package with a new name (e.g. `submodules/diff-triangle2-rasterization-tex`, Python module `diff_triangle_rasterization_tex`), so it never clashes with the installed one; the cluster YAML builds it at job start (`pip install --no-build-isolation ./submodules/diff-triangle2-rasterization-tex`, a few minutes). No external repository, everything on `devel4`.
- **B:** fork the rasterizer to your GitHub account, point the submodule at the fork, rebuild and push the Docker image. Cleaner long term, more setup.

## Effort

About 2–3 days: CUDA forward/backward and the gradient check (~1 day), bindings and renderer (~0.5 day), model integration (prune, refine, save, export) and the short runs (~1 day).

## Risks

- Seams between faces (v1): visible as edge discontinuities; mitigate with v2.
- Noise or view-specific "baking": texels are view independent, so they cannot fake reflections, but could overfit noise; `--lambda_tex_tv` and a texture learning rate lower than the vertex colours'.
- Texels on triangles that are rarely drawn get few gradients; with the drawable-size rules this is limited to small, mostly invisible triangles.
