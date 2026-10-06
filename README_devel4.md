# devel4: training triangle splatting from an existing mesh

This branch trains Triangle Splatting 2 starting from a mesh made by another method (gs2mesh), not from the SfM point cloud. The goal is a result that is still a clean, connected surface close to that mesh, not a soup of loose triangles. This file covers what was added, how the experiments are set up and run, and where things stand, so a new developer or Claude session can carry on.

Branch history: `main` (upstream release) → `devel4`, 8 commits (`5b47069` … `b4cb248`). `--no_prune` was cherry-picked from `devel3` (`91cf696`). All new options are **off by default**, so a run without the new flags behaves exactly like `main`.

Results deck from the earlier session (plots, mesh viewer, renders): https://claude.ai/artifact/8uFmcPZB3DeESaGxvYH8S8

---

## 1. What the branch adds

| Commit | Change | Files |
|---|---|---|
| `5b47069` | `--mesh_path`: initialize triangles from a mesh's own vertices, faces and vertex colors (no Delaunay on the point cloud) | `arguments/__init__.py`, `scene/__init__.py`, `scene/triangle_model.py` (`create_from_mesh`) |
| `422cd00` | Write the trained triangles as a colored mesh at the end of training: `<model_path>/mesh/iteration_<N>/mesh.ply` (colors from the SH DC term) | `scene/triangle_model.py` (`extract_mesh`), `train.py` |
| `cd7b09d` | Plain-text logs in `<model_path>/logs/`, because cluster stdout is not accessible | `utils/train_logger.py`, `train.py` |
| `7d8ecfc` | `scripts/align_mesh_to_colmap.py`: move a mesh from one COLMAP frame to another | `scripts/` |
| `0192525` | Three optional mesh regularizers (`--lambda_laplacian`, `--lambda_normal_consistency`, `--lambda_edge_smooth`) | `utils/loss_utils.py`, `train.py`, `arguments/__init__.py` |
| `ba3649b` | `scripts/compare_to_init_mesh.py`: measure how far training moved the mesh | `scripts/` |
| `02d5f25` | `--no_prune`: turn off **all** triangle and vertex pruning | `train.py`, `arguments/__init__.py` |
| `b4cb248` | Save the opacity floor in the checkpoint instead of assuming 0.9999 when loading | `scene/triangle_model.py` |

### 1.1 Mesh initialization (`--mesh_path`)

`TriangleModel.create_from_mesh()` loads the mesh with `trimesh` (`process=False`, so vertex order and faces are kept). Vertices become the trainable `vertices`, faces become `_triangle_indices`, and vertex colors (gray if there are none) become the SH DC coefficients. Opacity (`vertex_weight`) and sigma are set the same way as for point-cloud init. The vertices are **not frozen**: they train with the normal learning rate. Densification still runs (it subdivides triangles), so with pruning off the face count grows from 2.92M to about 5.55M on garden.

### 1.2 Aligning the mesh to the training cameras

gs2mesh ran its own COLMAP, so its mesh is in a different frame (scale, rotation, translation) from the Mip-NeRF 360 poses used for training. `align_mesh_to_colmap.py` matches camera centers of images present in both reconstructions, fits a similarity transform (Umeyama), and applies it to the mesh.

```bash
python scripts/align_mesh_to_colmap.py \
    --mesh <gs2mesh mesh>.ply \
    --src_colmap <dataset gs2mesh used> \
    --dst_colmap /datasets/manousis/360_v2/garden \
    --out <mesh>_aligned_360v2.ply
```

The script also writes `<out>_transform.txt` (the 4×4 similarity matrix). For garden the transform is close to identity: scale ≈ 1.001, rotation of a few degrees, translation ≈ 0.02.

**The garden init mesh used in every run so far:**

| | Path |
|---|---|
| Local, aligned (use this) | `/data/manousis/dev/gs2mesh/output/MipNerf360_nw_iterations30000_DLNR_Middlebury_baseline7_0p/garden_downsample3/garden_downsample3_MipNerf360_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel16_512_trunc2_60_2026_09_21_12_13_32_cleaned_mesh_aligned_360v2.ply` |
| Local, original gs2mesh output | same folder, same name without `_aligned_360v2` |
| Cluster copy (used by the YAMLs) | `/datasets/manousis/garden_downsample3_MipNerf360_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel16_512_trunc2_60_2026_09_21_12_13_32_cleaned_mesh_aligned_360v2.ply` |

1.56M vertices, 2.92M faces, 94.7% manifold edges, mean edge length 0.030 scene units.

It was made with (run from the repo root, local venv):

```bash
M=/data/manousis/dev/gs2mesh/output/MipNerf360_nw_iterations30000_DLNR_Middlebury_baseline7_0p/garden_downsample3/garden_downsample3_MipNerf360_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel16_512_trunc2_60_2026_09_21_12_13_32_cleaned_mesh.ply
.venv/bin/python scripts/align_mesh_to_colmap.py --mesh "$M" \
    --src_colmap /data/manousis/dev/gs2mesh/data/MipNerf360/garden_downsample3 \
    --dst_colmap /data/manousis/360_v2/garden \
    --out "${M%.ply}_aligned_360v2.ply"
```

For a new scene, use the same pattern: `--src_colmap` is the gs2mesh data folder of that scene, `--dst_colmap` is the `360_v2` scene, then copy the aligned mesh to `/datasets/manousis/` for the cluster.

### 1.3 Mesh regularizers

All three are in `utils/loss_utils.py`, are added to the loss in `train.py`, and start only after `--iteration_mesh` (default 5000). The edge connectivity (`mesh_topology()`) is cached and rebuilt after every prune/densify step (every 500 iterations).

| Flag | What it does | Related knobs |
|---|---|---|
| `--lambda_laplacian` | Uniform Laplacian on the vertices (each vertex pulled toward the average of its neighbors), normalized by mean edge length. Boundary vertices are excluded so open borders do not shrink inward. Each vertex is scaled by its smallest crease weight so sharp creases are not rounded. | `--crease_sigma` (default 0.1) |
| `--lambda_normal_consistency` | `1 − |cos|` between normals of the two faces sharing each manifold edge, down-weighted across sharp creases with `exp(−(1−cos)/crease_sigma)`. `|cos|` because face winding is not guaranteed consistent. | `--crease_sigma` |
| `--lambda_edge_smooth` | Image-space smoothness of the rendered normal map, relaxed where the ground-truth image has edges (`exp(−edge_alpha·|∇I|)`). | `--edge_alpha` (default 10) |

The Laplacian's boundary and crease handling was checked on a synthetic folded grid before any cluster run: the first version shrank boundaries and rounded the crease; the current version denoises and keeps the crease.

### 1.4 `--no_prune`

On `main`, `--start_pruning` only delays the periodic triangle prune. Vertex pruning and the end-of-training importance cleanup (deletes triangles with max blending weight ≤ 0.5) always run. `--no_prune` turns off all three. Densification is **not** affected.

### 1.5 Opacity floor (important for interpreting results)

Upstream training ramps a minimum opacity from `--start_opacity_floor` (5000) to 0.9999 at `--final_opacity_iter` (24000), via `triangles.update_min_weight()`. With pruning on, near-transparent triangles get deleted before the floor reaches them. With `--no_prune` they stay and get forced opaque, which is the main reason no-prune runs render worse (see §4).

To disable the floor, set `--start_opacity_floor` higher than `--iterations` (e.g. `31000`).

Until `b4cb248`, `load_parameters()` (used by `render.py` and `metrics.py`) set the floor to 0.9999 on load, so a run trained without the floor was scored as fully opaque. The floor is now saved as `opacity_floor` in `point_cloud_state_dict.pt`; older checkpoints still load with 0.9999. **Any no-floor run must be trained and evaluated at `b4cb248` or later.**

### 1.6 Training logs

`utils/train_logger.py` writes to `<model_path>/logs/`:

- `stdout.txt`: all prints and tracebacks
- `config.txt`: command line, git commit, machine, all arguments
- `metrics.txt`: tab-separated loss, sigma, vertex/face counts every 100 iterations
- `eval.txt`: test/train L1, PSNR, SSIM, LPIPS, FPS at the test iterations (every 1k in the cluster runs)
- `summary.txt`: `finished`/`failed`, initial and final vertex/face counts, peak GPU memory, mesh path

### 1.7 Comparing a trained mesh to the init mesh

```bash
python scripts/compare_to_init_mesh.py \
    --init <aligned init mesh>.ply \
    --meshes runA/mesh/iteration_30000/mesh.ply runB/mesh/iteration_30000/mesh.ply \
    --names runA runB \
    [--samples 1000000] [--tau 1.0] [--save_dist_ply out_dir]
```

It samples 1M points on each surface and computes the exact distance to the nearest triangle of the other surface. All distances are in units of the **init mesh's mean edge length** (0.030 for garden).

| Metric | Meaning |
|---|---|
| `acc_*` (accuracy) | trained → init distance: has the surface drifted? |
| `comp_*` (completeness) | init → trained distance: is the init surface still covered? |
| `precision_%` / `recall_%` / `fscore_%` | share of points within `--tau` (1 edge) in each direction, and their F-score |
| `normal_cos` | mean `|cos|` between normals at matched points |
| `dihedral_*`, `dihedral_>30deg_%` | roughness: angle between adjacent faces ("folds" > 30°). The init mesh has 13.2% |
| `manifold_%` / `boundary_%` | share of edges with 2 faces / 1 face. Tells whether the mesh fell apart |

`--save_dist_ply` writes copies of the trained meshes with faces colored by distance to init (blue = 0, red ≥ 5 edges). Sanity checks done when writing it: init vs itself gives 0 distance and 100% F-score; adding noise raises distance and roughness and lowers normal agreement, as expected.

---

## 2. Experiment setup

Shared by every run so far:

- **Scene**: Mip-NeRF 360 *garden*, `-i images_4` (1297×840), `--eval` (24 test views, every 8th image).
- **Init**: the aligned gs2mesh mesh above, via `--mesh_path`.
- **Schedule**: upstream defaults, 30k iterations, test evaluation every 1k.
- **After training**: `render.py --iteration 30000 --eval --skip_train`, then `metrics.py` (PSNR/SSIM/LPIPS on the test views → `results.json`, `per_view.json`), then `compare_to_init_mesh.py` run locally on the saved meshes, inside the training Docker image (`docker run --gpus all -v <repo>:/repo:ro -v <aligned mesh>:/m.ply:ro -v /data/manousis/cluster-results/devel4:/r …`). That is why the tables list the init mesh as `/m.ply`.
- **Hardware**: one RTX 5090 per job on the lab Kubernetes cluster; 17 to 43 minutes per run, 8 to 11 GB of GPU memory.
- **Statistics**: one scene, one seed per setting. Differences around 0.1 dB PSNR or 1% F-score are noise.

### 2.1 Cluster jobs

YAMLs: `/data/manousis/cluster_yamls/triangle_splatting/devel4/`
Results (NAS, mounted locally): `/data/manousis/cluster-results/devel4/` (written by the job to `/nas/results/triangle-splatting2/devel4/<RUN_NAME>`)

Each YAML is a `batch/v1 Job` that:

1. Uses the image `195.251.117.42:31000/triangle-splatting2:latest_main` (devel4 only changes Python, so the compiled CUDA extensions from `main` work) and runs `git fetch origin devel4 && git checkout -B devel4 origin/devel4`. **The job trains whatever is on `origin/devel4` when it starts, so push before submitting**, and check `env.txt` for the commit that actually ran.
2. Checks that the GPU is free: if more than 2 GB is already in use it writes to `status.txt` and exits. This was added after two jobs on node `iti-1073` OOMed because ~29.7 GB was held by something outside the pod. The node name is recorded in `status.txt`/`env.txt`.
3. Writes `env.txt` (node, git commit, `nvidia-smi`, memory), a 60 s `heartbeat.txt` (GPU memory/utilization), and symlinks `<model_path>/logs` to the NAS so logs are readable while the job runs.
4. Trains, renders the test views, runs metrics, and copies the whole output folder to the NAS. `status.txt` records the exit code of each step.

Naming (from the fixed-mesh experiments onward): YAML file is `<method>-triangle-splatting2-garden-devel4.yaml`; `EXP` inside it is the same without dashes, and the run folder is `<EXP>_<YYYY_MM_DD_HH_MM_SS>`. To add an experiment, copy the closest YAML, change `metadata.name`, `EXP`, the comment explaining what differs, and the `train.py` flags.

Contents of each run folder: `cfg_args`, `cameras.json`, `input.ply`, `point_cloud/iteration_30000/point_cloud_state_dict.pt`, `mesh/iteration_30000/mesh.ply`, `test/ours_30000/` (renders and GT), `results.json`, `per_view.json`, `logs/`, `train_log.txt`, `status.txt`, `env.txt`, `heartbeat.txt`.

Mesh comparison output: `compare_to_init_table_all.txt` (all runs), `compare_to_init_table.txt` and `compare_to_init_table_followup.txt` (stage subsets), and distance-colored meshes in `compare_to_init/`.

### 2.2 Runs

Regularizer weights below are Laplacian / normal consistency / edge smoothness. Every run uses `--mesh_path` with the aligned mesh.

| Stage | YAML (`…-triangle-splatting2-garden-devel4.yaml`) | Name in tables / deck | Flags beyond `--mesh_path` | Commit |
|---|---|---|---|---|
| 1 | `aligned-mesh-init` | baseline | none (pruning on) | `7d8ecfc` |
| 1 | `laplacian-edge-aware` | reg | λ 0.01 / 0.01 / 0.001 | `0192525` |
| 2 | `noprune` | noprune | `--no_prune` | `02d5f25` |
| 2 | `noprune-laplacian-edge-aware` | np_reg | `--no_prune`, λ 0.01 / 0.01 / 0.001 | `02d5f25` |
| 3 | `noprune-lap-edge-third` | np_reg_third | `--no_prune`, λ 0.003 / 0.003 / 0.0003 | `02d5f25` |
| 3 | `noprune-edge-aware-nolap` | np_nolap | `--no_prune`, λ 0 / 0.01 / 0.001 | `02d5f25` |
| 4 | `noprune-nolap-nc0p03` | nolap_nc0.03 | `--no_prune`, λ 0 / 0.03 / 0.001 | `b4cb248` |
| 4 | `noprune-nolap-noedgesmooth` | nolap_noedge | `--no_prune`, λ 0 / 0.01 / 0 | `b4cb248` |
| 4 | `noprune-nolap-nofloor` | nolap_nofloor | `--no_prune`, λ 0 / 0.01 / 0.001, `--start_opacity_floor 31000` | `b4cb248` |

`nofloor_opaque` in the follow-up table is not a separate run. It is the `nolap_nofloor` mesh with only the faces whose opacity is ≥ 0.5 (237k of 5.55M). A face's opacity is the mean of its three vertex opacities, `floor + (1 − floor)·sigmoid(vertex_weight)`, using the floor stored in the checkpoint. It was made with this inline snippet, run from `/data/manousis/cluster-results/devel4` inside the training Docker image:

```python
import glob, torch, trimesh, numpy as np
d = glob.glob('noprunenolapnofloor_*')[0]
st = torch.load(d + '/point_cloud/iteration_30000/point_cloud_state_dict.pt', map_location='cpu')
floor = st.get('opacity_floor')
op = (floor + (1 - floor) * torch.sigmoid(st['vertex_weight'])).squeeze().detach().numpy()
F = st['_triangle_indices'].long().numpy()
tri_op = op[F].mean(1)
m = trimesh.load(d + '/mesh/iteration_30000/mesh.ply', process=False)  # same vertex/face order as the checkpoint
keep = tri_op >= 0.5
trimesh.Trimesh(m.vertices, m.faces[keep], process=False).export('nofloor_opaque.ply')
```

That output was then passed to `compare_to_init_mesh.py` along with the other meshes (`--names … nofloor_opaque`).

The stages, in order:

1. **Regularizers with pruning on.** The mesh falls apart either way: pruning deletes about two-thirds of the faces and 85 to 88% of the remaining edges are boundaries. Regularizers cannot help once the mesh is disconnected.
2. **Pruning off.** The mesh stays connected (84% manifold) but crumples (72% folds). Adding the regularizers removes most of the folds.
3. **Which regularizer.** The uniform Laplacian pulls regions away from the init surface and costs about 0.6 dB, at full or one-third weight. Dropping it gives the best geometry.
4. **Follow-ups.** Stronger normal consistency (0.03) is about the same. The edge-smoothness term has no measurable effect. Turning off the opacity floor recovers 0.6 dB but leaves the scene drawn by translucent layers (§4).

---

## 3. Current state

**Recommended config for mesh-init runs:**

```bash
python train.py -s <scene> -i images_4 -m <out> --eval \
    --mesh_path <aligned mesh>.ply \
    --no_prune --lambda_normal_consistency 0.01
```

(Laplacian and edge smoothness off. `nolap_noedge` is exactly this run.)

Headline numbers on garden (test PSNR / LPIPS; F-score vs init at 1 edge; manifold edges):

| Run | PSNR | LPIPS | F-score | Manifold |
|---|---|---|---|---|
| baseline (pruned, no reg) | 22.40 | 0.288 | 23.7% | 11.5% |
| noprune | 21.34 | 0.401 | 69.8% | 84.2% |
| recommended (`nolap_noedge`) | 21.34 | 0.425 | 79.5% | 84.2% |
| nolap_nofloor | 21.94 | 0.321 | 71.9% | 84.3% |

Per-run numbers are in each `results.json` and in `compare_to_init_table_all.txt`. Training curves, per-view metrics and renders are in the deck linked at the top.

---

## 4. Known issues and open questions

- **The opacity floor trade-off.** No-prune runs are about 1 dB below the pruned baseline, mostly because of the floor. Every no-prune run peaks around 13k to 15k iterations and then declines as the floor forces unprunable triangles opaque. Without the floor, PSNR recovers 0.6 dB but 96% of faces end below opacity 0.5; the 4% that are opaque are as broken as the pruned baseline (F-score 28%, 31% manifold).
- **Options being considered** (not implemented):
  - (a) Apply the floor only to the init mesh's faces and their subdivisions, and let a separate set of free triangles stay transparent for foliage and background. Needs code: per-face "is mesh" flag that survives densification.
  - (b) Ramp the floor to about 0.5 instead of 0.9999. Only a flag/argument change.
  - (c) Accept the trade-off and move on.
- **Only one scene.** All conclusions are from garden with one seed. The next obvious step is to repeat the recommended config on another scene, with its own aligned gs2mesh mesh (align it with §1.2 first).
- **The geometry metric is relative to gs2mesh.** F-score vs init rewards staying close to the init mesh, not being correct. A check against an independent reference has not been done.
- **Cluster GPUs can be occupied** by processes outside the pod. Keep the pre-flight check in new YAMLs.

## 5. Checklist for a new experiment

1. Commit and push the code to `origin/devel4`.
2. If using a new scene, align its mesh (`align_mesh_to_colmap.py`) and put it under `/datasets/manousis/`.
3. Copy the closest YAML in `/data/manousis/cluster_yamls/triangle_splatting/devel4/`, rename following §2.1, change `EXP`, `metadata.name`, the explanatory comment and the flags.
4. After the job: check `status.txt` (all exit codes 0) and `env.txt` (expected commit), then read `results.json`.
5. Run `compare_to_init_mesh.py` against the init mesh, including the earlier runs you want to compare against, and save the table next to the others in `/data/manousis/cluster-results/devel4/`.

Steps 4 and 5 are automated by the Claude Code agent below.

## 6. Claude Code agent: `devel4-run-eval`

Defined in [.claude/agents/devel4-run-eval.md](.claude/agents/devel4-run-eval.md) (project-level, so it is available to any Claude Code session opened in this repo). In a session, ask for example:

> use the devel4-run-eval agent to evaluate the new runs

or name specific run folders. It first syncs from the cluster archive, `smb://195.251.117.42/manousis-archive/results/triangle-splatting2/devel4/`, which is mounted through GVFS under `/run/user/1003/gvfs/`:

- **Finished runs missing locally** are copied with `rsync` into `<run>.partial`. Each copy is checked (no file differs in size, `mesh.ply` is byte-identical) and then renamed.
- **Stale local copies** (local `status.txt` differs from the share's) are re-synced the same way.
- **Running and failed runs** are only reported: running ones with their iteration and last heartbeat, failed ones with the reason and the end of their log.
- **The share is never modified.** If it isn't mounted, the agent says so and continues with the local runs.

Then, for each run, it:

1. Checks `status.txt` (exit codes, GPU-busy stops), `env.txt` (commit; no-floor runs need `b4cb248`+), `logs/config.txt` (flags match what the YAML says it tests) and `logs/summary.txt` (which init mesh was used). Failed runs are reported with the end of their log and skipped.
2. Reads PSNR/SSIM/LPIPS from `results.json`, and the peak test PSNR and its iteration from `logs/eval.txt` (to spot the opacity-floor decline).
3. Runs `compare_to_init_mesh.py` with the local venv on the new runs plus two reference runs (`baseline`, `nolap_noedge`), writing `compare_to_init_table_<YYYY_MM_DD>.txt` and distance-colored meshes into `compare_to_init/`.
4. Returns a short report: one table with rendering and geometry metrics, differences from the recommended config, and anything suspicious.

By default it picks every run folder that is not yet a column in an existing `compare_to_init_table*.txt`. It never edits code or YAMLs, never commits, never submits jobs, and never overwrites existing tables. It runs on Opus (`model:` in the file header); change that line to `sonnet` for a cheaper, faster run.

When the experiment setup changes (new scene, new reference config, new short names), update the agent file too: its run-name mapping, init mesh path and reference runs are written into it.

## 7. Claude Code agent: `devel4-deck-update` (results deck)

The results deck (https://claude.ai/artifact/8uFmcPZB3DeESaGxvYH8S8) is built from a kit in `/data/manousis/cluster-results/devel4/deck/` (see its `README.md`):

- `deck.js`: slides and the run list
- `runs.json`: which run folders to include
- `build.py`: pulls every number from the run folders and the comparison tables
- `render_geo.py`: renders the mesh images

Each run's 24 test renders are stored as one image strip, because an artifact holds at most 255 files.

The agent is defined in [.claude/agents/devel4-deck-update.md](.claude/agents/devel4-deck-update.md). Run it **after** `devel4-run-eval`, which produces the comparison table and distance meshes it needs:

> use the devel4-deck-update agent to add the new runs to the deck

It:

1. Reads the live artifact and merges any edits made there since the last publish.
2. Registers each new run in `runs.json` and `deck.js`, and writes the stage text from the measured numbers.
3. Renders the mesh image and runs `build.py`, which must pass its checks.
4. Screenshots the key slides to check them.
5. Republishes to the same URL, uploading only the new images.

It doesn't rewrite existing conclusions. If new results contradict one, it reports that instead. It can also rebuild the deck from scratch.

Publishing needs the Artifact tool. If a subagent doesn't get it, the agent stops before publishing and returns the exact publish parameters for the main session.
