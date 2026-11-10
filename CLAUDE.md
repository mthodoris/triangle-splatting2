# CLAUDE.md — Physics-ready reconstruction on top of Triangle Splatting+

## Project goal

This repo is a clone of **Triangle Splatting+** (Held et al., arXiv 2509.25122), which optimizes opaque, semi-connected triangles for novel-view synthesis. We are extending it into a research prototype:

> **Jointly optimize the render triangles and a closed collision proxy (SDF field) from the same images, so the exported scene is usable for physics in game engines (Unity / PhysX). "What you see is what you hit."**

Two outputs per scene:
1. **Render mesh** — the dense opaque triangles from Triangle Splatting+ (unchanged purpose: looks right).
2. **Collision mesh** — watertight, manifold, low-poly mesh extracted from a co-trained SDF, guaranteed to lie within tolerance ε of the rendered surface.

Later: a benchmark of physical usability (raycast, drop test, navigation) to compare against baselines.

## Ground rules

- **Explore before changing.** Read the training loop, the triangle parametrization, the rasterizer interface, the config/argument system, and the export code before writing anything. Do not assume file names or APIs — check them.
- **Minimal upstream changes.** Put all new code in a new package `physproxy/`. Touch upstream files only to add small, clearly marked hooks (`# [physproxy]` comment). Every new feature is behind a flag that defaults to **off**, so the baseline reproduces exactly.
- **Small, verifiable steps.** Each phase ends with a check that actually runs (unit test, short training run, or exported file inspected). Don't move to the next phase until the current one is verified.
- **Use a small scene and short schedules while developing** (e.g. a Mip-NeRF360 scene at reduced resolution, few thousand iterations). Full runs only once a phase works.
- **Log everything new** (each loss term, SDF stats, proxy triangle count) to the same logger upstream uses.
- Ask before installing heavy dependencies (tiny-cuda-nn, custom CUDA extensions). Prefer pure PyTorch first.

## Notation

- `f(x)`: SDF, negative inside, positive outside, metres (or scene units — document which).
- `D`: per-pixel depth rendered from the opaque triangles for a training view.
- `ε`: tolerance between collision surface and render surface.
- AABB: region of interest for the proxy (from camera centres / SfM points, robust percentiles; sky and far background excluded).

## Phases

### Phase 0 — Understand and baseline
- Map the codebase: entry point, training loop, where losses are summed, where densification/pruning happen, how the final mesh is exported, whether the rasterizer already returns per-pixel depth and normals.
- Write `physproxy/NOTES.md` summarising this (files, functions, tensor shapes, coordinate conventions, scene scale).
- Train the baseline on one small scene; export the mesh.
- `physproxy/eval/mesh_stats.py`: triangle count, connected components, boundary edges, non-manifold edges, watertightness (trimesh), area of largest component. Run it on the baseline mesh. This is our "before" picture.

### Phase 1 — Depth access
- If the rasterizer returns depth, expose it through a hook. If not, render depth of the opaque triangles with nvdiffrast or a ray caster (Open3D `RaycastingScene`) — decide after Phase 0 and note why.
- Verify: save depth maps for a few views, check they match the RGB visually and are in the same units as the scene.

### Phase 2 — SDF module (`physproxy/sdf.py`)
- Pure PyTorch: dense multi-resolution feature grid (trilinear via `grid_sample`) + tiny MLP → scalar SDF. Domain = AABB, configurable resolution.
- **Gradients of f w.r.t. x:** use central finite differences (step ≈ half a voxel) rather than autograd double-backward through `grid_sample`, which is not reliably supported.
- Geometric init (sphere or plane) so training starts stable.
- Unit tests (`physproxy/tests/`): output shapes, finite-difference gradient sanity on an analytic sphere, fitting a sphere SDF from samples converges.

### Phase 3 — Losses (`physproxy/losses.py`)
Each term separately weighted and individually switchable:
1. `L_surf`: sample points on opaque render triangles (area-weighted barycentric sampling) → `mean |f(x)|`.
2. `L_free` (free-space carving): sample points along camera rays at depths `< D − margin` → `mean ReLU(δ − f(x))`.
3. `L_occ`: points at depths in `(D + margin, D + thickness)` → `mean ReLU(δ + f(x))` (inside, gives thickness).
4. `L_eik`: `(|∇f| − 1)²` at random points in the AABB and near the surface.
5. `L_normal`: `1 − |cos(n_tri, ∇f/|∇f|)|` at surface samples (absolute value at first; orientation handled later).
6. `L_simple` (optional): penalise SDF curvature / surface area to keep the proxy coarse.
- Margins and thickness in scene units, exposed as flags. Ignore pixels with invalid depth.
- Unit test each loss on a synthetic case (plane, sphere) with known answers.

### Phase 4 — Joint training
- Start SDF optimisation after iteration `K` (once triangles are mostly opaque), separate optimizer + LR for the SDF.
- Phase 4a: one-way coupling — SDF learns from the triangles, gradients do **not** flow into triangles (detach). Verify the SDF converges and the zero level set looks right (export intermediate meshes).
- Phase 4b: two-way coupling (flag): (i) pull opaque triangle vertices toward `f = 0`; (ii) extra pruning of triangles with `|f| > τ` and low view support (floaters). Compare render PSNR/SSIM/LPIPS with and without — rendering quality must not drop meaningfully.

### Phase 5 — Collision mesh export (`physproxy/export_collision.py`)
- Marching cubes (scikit-image) on the SDF at configurable resolution inside the AABB.
- Keep components above an area threshold; quadric decimation (pymeshlab or `fast-simplification`) to a triangle budget **with max error ε**.
- Check with `mesh_stats.py`: must be watertight + manifold. Report Hausdorff distance (both directions) between collision mesh and render mesh, triangle count.
- Write `render.obj` and `collision.obj` side by side in the same frame and units.

### Phase 6 — First physics evaluation (engine-independent)
- `physproxy/eval/raycast.py`: random rays inside the AABB; compare hit/miss and hit distance between two meshes (reference vs test). Use Open3D raycasting.
- `physproxy/eval/consistency.py`: for test views, distance between rendered depth and collision-mesh depth ("what you see is what you hit").
- Run on: baseline render mesh, baseline + naive cleanup (component removal + decimation), and our collision mesh.

### Phase 7 — Unity (later, separate folder `unity/`)
- Importer script: render mesh as visual, collision mesh as static `MeshCollider`.
- Drop test and capsule navigation test scripts with fixed timestep and seeds, logging to CSV.

## Experiment hygiene
- One config per experiment; output dir contains config dump, git hash, metrics JSON, exported meshes.
- Ablations planned from the start: each loss off one at a time; one-way vs two-way coupling; SDF resolution; ε budget.
- Keep a running `physproxy/RESULTS.md` table (scene, method, PSNR, triangle count, watertight, Hausdorff, raycast agreement).

## Done-criteria for the first milestone
On one scene: collision mesh is watertight and manifold, ≤ 50k triangles, Hausdorff to render surface ≤ ε, floaters visibly removed, raycast agreement better than the naive-cleanup baseline, render PSNR within ~0.2 dB of the original.
