# physproxy — notes on the upstream code (Phase 0)

Branch `devel8`, created from `main` at `4f0af82`. Everything below was checked in the
source and is not taken from the paper. File:line references are against that commit.

## Entry points

| What | Where |
|---|---|
| Training | `train.py` → `training()` |
| Arguments | `arguments/__init__.py` (`ModelParams`, `OptimizationParams`, `PipelineParams`; every attribute becomes `--name` automatically, a leading `_` adds a one-letter short flag) |
| Triangle model | `scene/triangle_model.py` → `TriangleModel` |
| Scene and cameras | `scene/__init__.py` → `Scene`; `scene/cameras.py` → `Camera`; `scene/dataset_readers.py` (COLMAP / Blender) |
| Renderer (Python) | `triangle_renderer/__init__.py` → `render()` |
| Rasterizer (CUDA) | `submodules/diff-triangle2-rasterization` (upstream submodule, installed into `.venv`) |
| Mesh export | `create_ply.py SCENE_DIR --out mesh.ply` (reads `point_cloud_state_dict.pt`) |
| Eval | `render.py`, `metrics.py`, `full_eval.py` |

Run used for the baseline (see `RESULTS.md`):
`python train.py -s /data/manousis/360_v2/garden -i images_8 -m output/physproxy/baseline_garden8 --eval`

## Triangle parametrization

The scene is an **indexed mesh with shared vertices**, not a triangle soup.

| Tensor | Shape | Meaning |
|---|---|---|
| `vertices` | `[V,3]` float, `nn.Parameter` | world positions |
| `_triangle_indices` | `[T,3]` int32 | vertex ids per triangle, **sorted ascending** (`create_from_pcd` sorts), so winding carries no orientation |
| `vertex_weight` | `[V,1]` logits | opacity through `opacity_activation = floor + (1-floor)·sigmoid(x)` |
| `_features_dc` / `_features_rest` | `[V,1,3]` / `[V,15,3]` | per-vertex SH color, interpolated barycentrically in the rasterizer |
| `_sigma` | scalar (log) | window sharpness, `get_sigma = exp(_sigma)` |
| `importance_score`, `image_size` | `[T]` | max blending weight, max screen size (used for pruning and densification) |

- Init (`create_from_pcd`): GPU Delaunay tetrahedralization of the SfM points; all tet faces are unique-d and become triangles. The initial mesh is therefore every face of a 3D Delaunay, which is massively non-manifold.
- Densification (`add_new_gs`, every 500 its from 500 to 13k): the chosen triangles are split 1→4 at edge midpoints. Neighbours are **not** split, so T-junctions appear and the edge-sharing breaks.
- Pruning (every 500 its after 4k): a triangle is dropped if its min vertex opacity ≤ threshold, its importance ≤ threshold (the threshold rises by 0.01 each step after 5k), or its screen size > 1400. Vertices that are unused and below the threshold are dropped too.
- Opacity floor: from 5k to 24k `update_min_weight` raises the floor from 0.1 to 0.9999, so **every triangle is opaque at the end**. `iteration_mesh = 5000` also turns on the normal loss.
- Sigma: annealed linearly from `set_sigma=1.0` to 1e-4 by `sigma_until=30k`, so triangles go from soft to sharp.
- Render upsampling: `triangles.scaling` is 1 until `start_upsampling=20k`, then `upscaling_factor=2`, then **hard-coded 4 at iteration 25000** (`train.py:96`). The image is rendered at `scaling·H × scaling·W` and area-downsampled.
- After the loop: one pass over all training views recomputes importance and drops every triangle with max blending ≤ 0.5, then `scene.save()`.

## Training loop and losses (`train.py:94–260`)

```
loss = (1-λ_dssim)·L1 + λ_dssim·(1-SSIM)        # loss_image
     + λ_weight · mean(vertex opacity)           # only before start_opacity_floor (5k)
     + λ_normals · mean(1 - <rend_normal, gt_normal>)   # only after iteration_mesh (5k)
loss.backward()                                  # train.py:162
... pruning/densification (no_grad, every 500 its)
triangles.optimizer.step(); zero_grad()          # train.py:256
```

- **Hook for new losses:** add to `loss` just before `loss.backward()` (line 162). Optimizer steps happen at line 256, after pruning, so a second optimizer (SDF) can step alongside.
- One camera per iteration, drawn without replacement from a shuffled stack.
- `gt_normal`: Metric3D (`torch.hub yvanyin/metric3d`, ViT-L) normals in **camera space**, cached as PNG in `<scene>/normals_<n>/`; computed on the first load of a resolution. The hub model is loaded on **every** run even when the cache exists (slow start, needs `~/.cache/torch/hub`).
- Single Adam optimizer, 4 parameter groups (`f_dc`, `f_rest`, `vertices`, `vertex_weight`). The vertex LR is 0 for the first 1000 its, then exponential decay 1.5e-3 → 1.5e-5.

## Renderer outputs (`triangle_renderer/render()`)

The rasterizer returns `color [3,H,W]`, `radii [T]`, `scaling [T]` (screen size), `allmap [7,H,W]`, `max_blending [T]`. Channels of `allmap`, from `forward.cu:678–688`:

| ch | Written in CUDA | Python name in `render()` | Notes |
|---|---|---|---|
| 0 | `D = Σ wᵢ·zᵢ` (expected depth; overwrites `last_contributor`) | `entropy` (misnamed) | not normalized by alpha |
| 1 | `1 − T` | `rend_alpha` | accumulated alpha |
| 2–4 | `Σ wᵢ·nᵢ` | `rend_normal` | **view-space** normal, flipped to face the camera, not normalized |
| 5 | median depth (z of the last triangle drawn while T > 0.5) | `surf_depth`, `depth_full` | |
| 6 | id of that triangle (`pixel_influence`) | `rend_ids` | **uninitialized where alpha < 0.5**: mask before use |

Details that matter for the proxy:
- **Depth is the view-space z of each triangle's centroid** (`depths[idx] = p_view_triangle.z`, `forward.cu:446`), constant over the whole triangle. It is not a per-pixel ray/plane intersection, so it is wrong by up to half a triangle's depth extent at grazing angles. Units: view-space z (not ray distance), scene units.
- `depth_full` and `rend_ids` are returned at the **render resolution** (`scaling·H × scaling·W`); the other maps are area-downsampled to `H×W`. Averaging ids or depth across a silhouette is meaningless, so use the full-resolution maps.
- Culling (`forward.cu:395`): a triangle is not drawn if its on-screen inradius is < 1 render pixel, its size is < 1 or > 1600 px, it lies outside the near plane (z < 0.2), or it is nearly edge-on (|cos| < 0.001). Small triangles therefore get no depth and no gradient at scaling 1.
- `surf_normal`, from finite differences of the median depth, is computed but unused.

## Cameras and coordinates

- COLMAP world frame, no normalization applied (the `translate` from `getNerfppNorm` is computed but never used). **Units are COLMAP units, not metres.**
- `Camera.world_view_transform` is W2C **transposed** (row-vector convention: `p_view = [p,1] @ world_view_transform`), and `full_proj_transform = W2C^T @ P^T`. `camera_center` is in world space. View space: +z forward, +x right, +y down (OpenCV/COLMAP). `znear 0.01`, `zfar 100`.
- Pixel → ray: `utils/point_utils.depths_to_points` builds rays with `z = 1`, so `point = z_depth · rays_d + rays_o` uses **z-depth**, matching channel 5.

## Scene scale (garden, `images_8` = 648×420, 185 images, 161 train / 24 test with `--eval`)

- Camera centers lie on a ring: distance to their mean is 3.2 / 3.7 / 4.5 (min/median/max); camera bbox ≈ [−3.8, −2.7, −4.0] … [3.8, 3.0, 4.1]. `cameras_extent` (`radius`) ≈ 4.9.
- SfM: 138,766 points. Distance from the camera center: median 5.5, p90 11.3, p99 17.1, max 61.9. The 5–95 percentile box is [−8.9, −3.3, −7.2] … [6.4, 5.6, 8.8].
- World up is not an axis: mean camera up ≈ (0, −0.88, −0.47).
- So for a proxy AABB, the camera ring (r ≈ 4) encloses the table and the near lawn, while the far background (trees, house) reaches 10–60 units. A box from robust SfM percentiles clipped to about 1.5× the camera radius is a reasonable first choice.

## Logging

**TensorBoard is not installed in the local `.venv`**, so `tb_writer` is `None` and nothing is logged except stdout. Also `--quiet` (`safe_state`) silences stdout entirely, including the PSNR prints, so leave it off, or evaluate afterwards with `render.py` + `metrics.py`. Decision (2026-11-10): new terms are logged by `physproxy/log.py` (`JsonLogger`) to `<model_path>/physproxy_log.jsonl`, one JSON record per line, and read back with `read_log(path, tag)`. TensorBoard is not added.

When present, TensorBoard `SummaryWriter(model_path)` is created in `prepare_output_and_logger` and passed as `tb_writer` to `training_report` (logs `train_loss_patches/*`, `iter_time`, test/train PSNR every 1000 its). New terms are logged with `tb_writer.add_scalar('physproxy/<name>', v, iteration)`. `cfg_args` (a dump of the argument namespace) is written to the output dir.

## Upstream quirks noted (left untouched)

- `--save_iterations` is ignored: only the final model is saved (`scene.save(iteration)` after the loop).
- The test/train evaluation runs every 1000 its regardless of `--test_iterations`.
- Checkpoint restore is broken (`capture()` does not include vertices or indices, and `training_setup` is called with the wrong signature).
- `training_setup` is defined twice in `TriangleModel`; the second definition wins.
- `iteration == 25000 → scaling = 4` is hard-coded, so shorter schedules never reach scale 4 unless `--iterations` > 25000.

## Phase 1 decision (depth access)

Use the rasterizer's own output and do not add nvdiffrast or Open3D for training-time depth:
`rend_ids` (channel 6, full render resolution) gives the opaque triangle seen at each pixel. An exact per-pixel depth is then one ray/plane intersection per pixel in PyTorch (camera ray vs. the plane of that triangle), which costs O(pixels), needs no new dependency, and fixes the centroid-depth error. Pixels with alpha < 0.5 or an out-of-range id are invalid. Once the floor is at 0.9999 (from about 24k) the median triangle is the visible one. Open3D `RaycastingScene` stays an offline cross-check (Phase 6 needs it anyway).

## Phase 1 result (done, 2026-11-10)

- `physproxy/depth.py`: `exact_depth(view, vertices, faces, rend_ids, rend_alpha)` returns z-depth, validity, triangle id, barycentrics and world points per render pixel, differentiable w.r.t. the vertices. `render_exact_depth(view, triangles, render_pkg)` wraps a `render()` output.
- Upstream hook: `render()` also returns `alpha_full` (`allmap[1:2]` at render resolution), marked `# [physproxy]`. It adds one dict key and changes nothing else, so the baseline is unaffected.
- Check (`physproxy/eval/check_depth.py`, baseline garden `images_8`, 4 test views, render scale 4 = 2592×1680; output in `output/physproxy/baseline_garden8/physproxy_depth_check/`):
  - valid pixels 99.97–99.999 %; every hit lies on its triangle (min barycentric ≥ 0).
  - Independent z-buffer (pytorch3d, perspective-correct, same mesh): where both pick the same triangle (73–74 % of pixels) the depth agrees to 6e-8 relative (max 2e-6). On the remaining pixels both hit the same surface through different triangles of the stacked layers (median 0.15 %, p90 0.6 % apart). TS+ sorts by centroid per tile, and 9 % of those pixels lie on triangles TS+ culls.
  - Units and alignment vs COLMAP: at the keypoints of 4.7–5.9k SfM points per view, median |Δz|/z = 0.2–0.3 %, and 98–99 % are within 5 %. So the depth is in scene (COLMAP) units and in the right frame.
  - Centroid depth (rasterizer) vs exact: mean 0.24–0.29 %, p99 0.7–0.8 % at the end of training, where the triangles are small. The error matters most early in training, when the triangles are large, and for gradients, since the centroid depth is constant over a triangle.
  - The depth images (RGB | centroid | exact | difference) match the RGB visually.
- Tests: `physproxy/tests/test_depth_and_log.py` (tilted plane with analytic depth, rotated camera vs a known point, world↔view round-trip, JSON logger).

## Phase 2 result (done, 2026-11-10)

- `physproxy/sdf.py`, `SDFField`: f(x) = prior(x) + out_scale · MLP([multi-res grid features, x_n]). Dense pyramid of `[C,R,R,R]` grids (default C=4, R=16·2^l for 4 levels, so 16…128), trilinear `grid_sample` (border padding), a 2×64 Softplus MLP and a **zero-initialized last layer**, so at init f equals the analytic prior exactly (sphere, plane or none) and |∇f| = 1. Scene units, negative inside.
- ∇f uses central finite differences, h = half the finest voxel by default, with the 6 offsets in one batched call. Double backward through 3D `grid_sample` is not implemented in PyTorch 2.7 (checked: "derivative for aten::grid_sampler_3d_backward is not implemented"). The FD gradient is itself differentiable w.r.t. the parameters, so eikonal and normal losses train the field.
- `aabb_from_points(sfm, cam_centers)`: SfM points within 1.5× the max camera distance of the camera centroid, 2–98 % percentiles per axis, 5 % padding. On garden: [−4.87, 0.04, −3.26] … [5.70, 4.99, 6.09] (extent 10.6 × 4.9 × 9.4). With the default grid that is 9.6M parameters and a finest voxel of 0.083 / 0.039 / 0.074. One step (200k f + 50k FD gradients + backward) takes 31 ms and peaks at 0.69 GB on the 3060. With base 32 (finest 256): 76.7M parameters, 44 ms, 1.19 GB.
- **Caveat for Phase 4/5:** garden's up axis is tilted (≈ (0, −0.88, −0.47)), so an axis-aligned box wastes volume and the y-percentile cut sits at about camera height. Consider an up-aligned (oriented) box, i.e. rotating into a frame whose z is the mean camera up, before choosing a resolution. The grid is also cubic in R per axis while the box is not.
- Tests (`physproxy/tests/test_sdf.py`, all pass): shapes and f == prior at init; FD ∇f vs the analytic sphere (error < 1e-3 for r > 0.25, error ∝ (h/r)²); FD vs autograd's first derivative on a rough random field (converges as h → 0: median relative error 0.46 at h=1e-2, 3e-4 at h=1e-4); FD eikonal loss gives finite gradients on all grids; fitting an off-centre sphere from a plane init (1500 Adam steps, 3.5 s) gives mean |f − f*| = 0.0022 over a 2-unit box, |f| = 0.0020 on the surface, mean ||∇f| − 1| = 0.035; the AABB helper drops a far background.
