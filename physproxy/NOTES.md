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
- **No upstream file is modified** (user rule, 2026-11-10; an earlier `alpha_full` key in `render()` was reverted). Pixel validity comes from unmodified outputs: `physproxy.depth.pixel_alpha(pkg)` = `depth_full > 0` (the median depth and `rend_ids` are written in the same branch) and `rend_alpha ≥ 0.5` (repeated from image to render resolution).
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
- **Tilted up axis: resolved.** `physproxy/aabb.py` builds a gravity-aligned box: z = mean camera up (or the normal of the camera ring), x/y = principal axes of the scene in the ground plane, robust percentiles, top raised to the highest camera (`top_to_cameras`), since SfM pulls the upper percentile down and cut the vase top. `SDFField(frame=...)` takes the rotation (rows = local axes in world), and the box corners are in that frame. The prior, FD gradient and outputs stay in world units (a rotation preserves distances). With `isotropic=True` (default) the per-axis resolution follows the box extent, so voxels are about cubic. Garden: box 11.21 × 10.23 × 4.49 (local), covering 96.6 % of the SfM points within the radius and 100 % of the central table column. Grid (128, 117, 51) at the finest level, 3.5M parameters (vs 9.6M axis-aligned), voxel 0.088–0.090. Before the fix the axis-aligned box was 10.6 × 4.9 × 9.4 (y cut at camera height).
- Tests (`physproxy/tests/test_sdf.py`, all pass): shapes and f == prior at init; FD ∇f vs the analytic sphere (error < 1e-3 for r > 0.25, error ∝ (h/r)²); FD vs autograd's first derivative on a rough random field (converges as h → 0: median relative error 0.46 at h=1e-2, 3e-4 at h=1e-4); FD eikonal loss gives finite gradients on all grids; fitting an off-centre sphere from a plane init (1500 Adam steps, 3.5 s) gives mean |f − f*| = 0.0022 over a 2-unit box, |f| = 0.0020 on the surface, mean ||∇f| − 1| = 0.035; the AABB helper drops a far background.

## Phase 3 result (done, 2026-11-10)

`physproxy/losses.py`. Every term has its own weight in `ProxyLossConfig`, and a weight of 0 skips both the term and its sampling. Distances are in scene units.

| Term | Samples | Formula |
|---|---|---|
| `L_surf` | area-weighted barycentric points on render triangles that overlap the box (triangle bounds in the box frame vs the box) | mean \|f\| |
| `L_free` | stratified points along rays to valid exact-depth pixels, s ∈ (near = 0.2, D − margin/\|d\|) | mean ReLU(δ − f) |
| `L_occ` | s ∈ (D + margin/\|d\|, D + thickness/\|d\|) | mean ReLU(δ + f) |
| `L_eik` | uniform in the oriented box + surface samples jittered by N(0, eik_sigma) | mean (\|∇f\| − 1)² |
| `L_normal` | surface samples and their triangle normals | mean 1 − \|cos(n_tri, ∇f)\| (unsigned) |
| `L_simple` | `curvature`: near-surface points, normal change over a random step; `area`: uniform box points | mean ‖n(x) − n(x+εu)‖², or mean e^{−\|f\|/β}/(2β) (= surface area / box volume) |

- s is the view z-depth (ray = o + s·d, d = R_c2w (x/z, y/z, 1)), as in `depth.py`. Margins are true distances along the ray.
- Defaults: margin 0.05, thickness 0.2, δ 0.02, weights surf 1 / free 1 / occ 0.5 / eik 0.1 / normal 0.05 / simple 0. **δ is kept small on purpose:** a free-space point is only guaranteed f > 0; it can sit next to another surface when the ray grazes it.
- Surface sampling only uses triangles overlapping the box. Without that, 80 % of the area-weighted samples fell on big background triangles outside it (6.6k of 32k in the box, now 31.4k). Every sample set is clipped to the box, since outside it the grid is clamped.
- Tests (`physproxy/tests/test_losses.py`, 11, all pass):
  - Area-weighted sampling: 75 % ± 0.5 % on the 3×-area triangle, centroid recovered.
  - Ray intervals respect margin and thickness along the ray.
  - `L_surf` on an icosphere: < 1e-3, and 0.100 for a 0.1-too-small field.
  - Exact `L_free` / `L_occ` on a ray toward a wall: 0 for the right wall; (3−m−2+δ)²/2/(3−m) for a too-near wall; δ+4−3.125 for a too-far one.
  - `L_eik` = 0 / 1.000 for a unit / doubled sphere.
  - `L_normal` = 0 for sphere normals (and flipped ones), 1 at 90°, 0.5 at 60°.
  - Area density × volume = 4πr² and 4.0 for a plane, within 3 %.
  - Curvature: 0 on a plane, > 4× larger on a sphere of a quarter the radius.
  - No triangle in the box gives finite losses.
  - **Integration:** an `SDFField` started as a plane at z = 1.5, trained 800 steps from one camera's exact depth of a wall at z = 3 with all five terms, ends with its zero crossing on the optical axis at z = 3.002.
- Garden smoke test (baseline model, oriented box, plane init, default config, one view per step): 31.4k surface, 62k free and 12.8k occupied samples; 23 ms for samples + losses + backward; 2.0 GB peak including the TS+ model. Not trained yet; that is Phase 4.
- For Phase 4: the plane init sits at the box centre (local z ≈ −1.1), while the ground is at local z ≈ −3. A better init is a plane at a low percentile of the SfM heights.

## Phase 4a result (done, 2026-11-10): one-way coupling

**Integration without upstream edits.** `physproxy/train_proxy.py` takes the same arguments as `train.py` plus `--proxy*` (`physproxy/args.py`). It runs the unmodified `train.training()`, after replacing, inside its own process only:
- `train.Scene` with a subclass that builds the `ProxyTrainer`,
- `train.render` with a wrapper that snapshots, for training cameras only, the view, `rend_ids`, validity, and vertex/face references (pruning replaces these tensors, so the references stay consistent with the ids),
- `optimizer.step` with a wrapper that first runs the SDF step on the last snapshot (under `enable_grad`, since upstream calls `step` inside `no_grad`).

`finish()` runs after `training()` returns. Upstream does not call `step` on the last iteration, so the SDF skips that one step. `physproxy/compat.py` routes the Metric3D `torch.hub.load` to the local cache within physproxy processes, because a GitHub 504 had aborted startup.

**Run:** `python physproxy/train_proxy.py -s .../garden -i images_8 -m output/physproxy/p4a_oneway --eval --proxy --proxy_start 20000 --proxy_mesh_every 2500 --proxy_extra_iters 3000`. Defaults: ground-plane init at local z = −3.20 (5th percentile of in-box SfM heights), grid 16→128 (finest 128×117×51, voxel ≈ 0.09), losses as in Phase 3. Took 36 min total on the 3060 (baseline 28). The SDF step costs 29 ms at render scale 1–2 and 48 ms at scale 4.

**Rendering is unchanged** (expected with detached triangles): test 25.33 dB / SSIM 0.801 / LPIPS 0.192 (`render.py` + `metrics.py`) vs baseline 25.34 / 0.801 / 0.192.

**Convergence** (JSON log, every 100 steps): L_surf 0.025 → 0.011 within 1k steps, then flat around 0.0095–0.011; L_free ≈ 3e-4 throughout; L_eik 0.092 → 0.057; L_normal 0.24 → 0.157, still slowly falling. Exported meshes (render→proxy p50/p90; proxy→render p50/p90, scene units):

| it | proxy faces | components | watertight | render→proxy p50 / p90 | proxy→render p50 / p90 |
|---|---|---|---|---|---|
| 20000 (init plane) | 244k | — | yes | 0.089 / 1.39 | 0.050 / 0.139 |
| 22500 | 966k | — | yes | 0.0069 / 0.027 | 0.036 / 0.122 |
| 25000 | 997k | — | yes | 0.0063 / 0.025 | 0.036 / 0.123 |
| 27500 | 998k | — | yes | 0.0067 / 0.027 | 0.037 / 0.128 |
| 33000 (after 3k SDF-only) | 1.01M | 719 (largest 99.6 % of area) | yes, manifold, consistent winding | 0.0068 / 0.027 | 0.037 / 0.130 |

So the field converges within about 2.5k steps. 99.95 % of the render surface in the box lies within one voxel of the zero level.

**What you see vs what you hit** (`physproxy/eval/proxy_views.py`, 24 test views, 18.1M in-box pixels, tolerance 2 % of depth): **agree 81.8 %, blocks 17.3 %, see-through 0.9 %.** Pictures are in `output/physproxy/p4a_oneway/proxy/views_mesh_33000/`.
- Ground, lawn, paving and the table top are right (green).
- Almost all the "blocks" error (the proxy in front of the visible surface) is **thin structures made fat**: table legs become thick blobs, the table top is a slab thicker than the real top (red rim under it), the dried flowers on the vase are one blob, and bush edges bulge. Causes: voxel 0.09 and `thickness` 0.2 (L_occ forces 0.2 of inside behind every surface, which is thicker than a leg or the table top), plus free-space samples spread along the whole ray, so few of them carve near the silhouettes.
- See-through is rare (0.9 %): the proxy almost never lets a ray pass through something visible.
- This must be fixed before two-way coupling, since 4b would pull triangles towards a bloated proxy. Candidate levers: finer grid (base 32 → voxel 0.044), thinner `thickness` (0.05–0.1) and smaller `margin`, and free-space samples concentrated in a band before the surface.

## Bloat fix (2026-11-10): proxy-settings sweep on the frozen 4a model

`physproxy/fit_proxy.py` fits only the SDF (3000 steps, one random training view per step, render scale 4) on the trained `p4a_oneway` triangles and scores it on the 24 test views (`compare_views`, depth tolerance 2 %). The in-training 4a proxy scored 81.8 / 17.3 / 0.9; the refit with the same settings (A) gives 83.9 / 14.1 / 2.1, so refits are a fair stand-in. Results also go to `output/physproxy/p4a_oneway/proxy_fits/summary.jsonl`.

| Fit | Settings | Agree | Blocks (proxy in front) | See-through | render→proxy p90 | Faces (MC) | Components |
|---|---|---|---|---|---|---|---|
| A_default | 4a defaults: grid 16, thickness 0.2, margin 0.05, w_free 1, 4096 rays × 16 | 83.9 % | 14.1 % | 2.06 % | 0.0264 | 952,996 | 446 |
| B_t01_m003 | A + thickness 0.1, margin 0.03 | 88.8 % | 10.2 % | 1.03 % | 0.0271 | 971,648 | 398 |
| G_res32 | A + grid 32 | 80.9 % | 18.4 % | 0.67 % | 0.0220 | 2,574,152 | 1455 |
| C_t01_m003_res32 | B + grid 32 | 87.3 % | 12.4 % | 0.30 % | 0.0223 | 2,688,768 | 1166 |
| D_C_band05 | C + 50 % free samples in a 0.3 band before the surface | 90.6 % | 8.9 % | 0.49 % | 0.0227 | 2,583,780 | 1271 |
| E_D_t005 | D + thickness 0.05 | 91.9 % | 7.7 % | 0.36 % | 0.0211 | 2,716,068 | 966 |
| F_D_wocc02 | D + w_occ 0.2 | 92.0 % | 5.8 % | 2.22 % | 0.0170 | 2,649,112 | 1592 |
| H_res16_t005_band05 | E with grid 16 | 90.9 % | 8.2 % | 0.92 % | 0.0273 | 977,972 | 480 |
| I_E_6k | E, 6000 steps | 91.9 % | 7.8 % | 0.28 % | 0.0209 | 2,768,320 | 1053 |
| J_E_band075_w02 | E + band 75 %, width 0.2 | 91.7 % | 7.8 % | 0.53 % | 0.0210 | 2,690,808 | 1069 |
| K_E_t003 | E + thickness 0.03, margin 0.02 | 92.5 % | 7.1 % | 0.35 % | 0.0203 | 2,770,088 | 1042 |
| L_K_wfree4 | K + w_free 4 | 94.8 % | 3.3 % | 1.91 % | 0.0179 | 2,600,208 | 1711 |
| M_K_rays16k | K + 16384 rays × 8 | 92.7 % | 6.9 % | 0.38 % | 0.0199 | 2,772,980 | 1056 |
| N_K_wfree4_rays16k | K + w_free 4 + 16384 rays × 8 | 95.0 % | 3.8 % | 1.20 % | 0.0185 | 2,627,236 | 1652 |
| O_wfree2_rays16k | **K + w_free 2 + 16384 rays × 8 (new defaults)** | 94.1 % | 5.3 % | 0.59 % | 0.0193 | 2,713,532 | 1199 |
| P_wfree4_wsurf3_rays16k | N + w_surf 3 | 94.5 % | 3.7 % | 1.78 % | 0.0104 | 2,843,312 | 1704 |

Findings:
- **Inside-band thickness is the main bloat lever** (0.2 → 0.1 → 0.05 → 0.03: 14.1 → 10.2 → 7.7 → 7.1 % with the other changes). L_occ forced 0.2 of inside behind every surface, thicker than a leg or the table top.
- **The finer grid helps only once the band is thin**: grid 32 alone is worse (18.4 %, more floater components), and grid 32 with thickness 0.05 beats grid 16 (7.7 vs 8.2 % bloat, 0.36 vs 0.92 % holes). More steps (6k) change nothing, so the fits are not undertrained.
- **Free-space band sampling** (half the free samples within 0.3 before the surface) helps: 12.4 → 8.9 %.
- **The remaining bloat is not sub-voxel.** For K, the gap on bloat pixels is median 0.37, p90 2.7 units. It is solid proxy where the camera sees through to the background: mostly a rim under the table-top edge (the underside is barely seen from the training cameras, which mostly look down), plus bush fringes and leg drips.
- **A stronger free-space weight removes it but punches holes:** w_free 4 gives 3.3 % bloat but 1.9 % see-through, visible as speckle holes in the thin table top. That is worse for a collider (things fall through). A stronger surface term (w_surf 3) does not prevent the holes.
- **Chosen (O, now the defaults in `physproxy/args.py`):** grid base 32, thickness 0.03, margin 0.02, w_free 2, 16384 rays × 8 samples with 50 % in the band, MC resolution 384. **94.1 % agree / 5.3 % blocks / 0.59 % see-through**, vs 83.9 / 14.1 / 2.06 for the 4a settings. render→proxy p90 0.019. The meshes are larger (2.7M faces at MC 384); Phase 5 decimates.
- Bloat measurement: `compare_views` now also reports the gap distribution on bloat pixels (`blocks_gap_p50/p90/p99`, `blocks_frac_gap_over_t`).

## Phase 4b result (2026-11-10): two-way coupling

Two full garden runs (30k plus 3k SDF-only steps, about 45 min each on the RTX 3060). Both use the post-sweep defaults and turn the proxy on at 20k. R0 is one-way. R1 adds two-way coupling from 23k: `--proxy_pull_w 5e-5 --proxy_prune_floaters`.

| | R0 one-way | R1 two-way |
|---|---|---|
| Test PSNR / SSIM / LPIPS | 25.37 / 0.800 / 0.192 | 25.31 / 0.801 / 0.191 |
| See-vs-hit agree / blocks / see-through | 94.7 / 3.8 / 1.5 % | 94.7 / 3.5 / 1.8 % |
| render→proxy \|f\| p50 / p90 | 0.0044 / 0.016 | 0.0046 / 0.017 |
| Proxy faces / components / largest-comp. area | 2.76M / 2,322 / 97.6 % | 2.68M / 2,812 / 96.0 % |
| Render triangles / components | 399,514 / 251,884 | 398,615 / 252,426 |

Findings:
- **Two-way coupling at these settings is a no-op.** Rendering stays within the 0.2 dB budget (−0.05 dB), but none of the geometry metrics improve.
- **The pull is too weak to move vertices.** λ 5e-5 was chosen to match the median photometric gradient. Vertex |f| stays at p50 0.0065 / p90 0.02 while it acts, and render→proxy is unchanged, because 99% of vertices were already within 0.04.
- **Pruning removed few triangles and did not clean the proxy.** It ran 14 times and removed 6,243 triangles, 1.6% of the mesh. Every candidate came from the small-proxy-component criterion; the |f| > τ criterion never fired. Only 0.2% fewer render triangles remain, so upstream densification and pruning absorb the change. The proxy then has more small components (2,812 vs 2,322), not fewer: removing triangles leaves small, unsupported pieces that the SDF still closes into blobs.
- Without decimation, the proxy is about 2.7M faces at MC 384. The 50k budget is a Phase 5 problem.

Next options: a stronger pull (5e-4 to 5e-3, checking that PSNR holds), or skip two-way coupling for now. The floater problem is better handled on the proxy side in Phase 5 (component area threshold plus decimation), so the next step is Phase 5.
