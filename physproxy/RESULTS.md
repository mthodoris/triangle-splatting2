# physproxy results

Rendering metrics are on the 24 test views (`--eval`, every 8th image) via `render.py` + `metrics.py`. Mesh metrics come from `physproxy/eval/mesh_stats.py`. Hausdorff and raycast columns get filled from Phase 5/6 on.

| Scene | Method | Commit | PSNR | SSIM | LPIPS | Triangles | Components | Largest comp. (area %) | Boundary edges | Non-manifold edges | Watertight | Hausdorff | Raycast agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| garden `images_8` | TS+ baseline, upstream defaults, 30k | `4f0af82` (main) | 25.34 | 0.801 | 0.192 | 396,680 | 250,382 | 0.05 % (20 faces) | 895,596 (85.9 %) | 0 | no | — | — |
| garden `images_8` | 4a one-way SDF proxy (render mesh unchanged) | `0cfa3d7` | 25.33 | 0.801 | 0.192 | proxy (MC 256, not decimated): 1,008,520 | 719 | 99.6 % | 0 | 0 | **yes** | render→proxy p50 0.007 / p90 0.027 | see-what-you-hit: agree 81.8 %, blocks 17.3 %, see-through 0.9 % |
| garden `images_8` | 4b R0: one-way, new proxy defaults (grid 32, band) | `534dd90` | 25.37 | 0.800 | 0.192 | render 399,514; proxy (MC 384): 2,755,584 | render 251,884; proxy 2,322 | proxy 97.6 % | proxy 0 | 0 | proxy **yes** | render→proxy p50 0.004 / p90 0.016 | see-what-you-hit: agree 94.7 %, blocks 3.8 %, see-through 1.5 % |
| garden `images_8` | 4b R1: two-way, pull λ 5e-5 + floater pruning | `534dd90` | 25.31 | 0.801 | 0.191 | render 398,615; proxy (MC 384): 2,682,408 | render 252,426; proxy 2,812 | proxy 96.0 % | proxy 0 | 0 | proxy **yes** | render→proxy p50 0.005 / p90 0.017 | see-what-you-hit: agree 94.7 %, blocks 3.5 %, see-through 1.8 % |
| garden `images_8` | Phase 5 collision mesh from R0, f = 0 surface (MC 192, decimated to 50k, repaired; superseded by the skin row) | (this commit) | — (render as R0) | | | **49,262** | 100 | — | 0 | 0 | **yes** (consistent outward winding) | render→collision p50 0.017 / p90 0.057 / p99 0.19 / max 0.53; collision→render max 0.89 | — |
| garden `images_8` | **Phase 6 collision mesh from R0, skin 0.015** (`--iso 0.015`, default) | (this commit) | 25.37 | 0.800 | 0.192 | **48,852** | 47 | — | 0 | 0 | **yes** | render→collision p50 0.029 / p90 0.071 / p99 0.11 / max 0.41 | random rays: of ref hits caught 75.4 %, pass 3.7 %; down: caught 90.0 %; drop falls through 0.01 %; test-view agree 91.1 % |
| garden `images_8` | Phase 6 naive cleanup of R0 render mesh (weld 0.02, crumbs < 1e-3, decimate 50k) | (this commit) | — | | | 50,000 | 20,920 | — | 93,836 (79.7 %) | 5,675 | no | — | random rays: caught 66.9 %, pass 18.9 %; down: caught 74.3 %; drop falls through 24.8 %; test-view agree 72.0 % |
| garden `images_8` | Phase 6 collision mesh from R1 (two-way), skin 0.015 | (this commit) | 25.31 | 0.801 | 0.191 | 49,006 | 61 | — | 0 | 0 | **yes** | render→collision p50 0.028 / p90 0.070 / p99 0.11 / max 0.25 | random: caught 76.0 %, pass 3.2 %; down: caught 90.0 %; drop falls through 0.01 %; test-view agree 91.6 % |

## Baseline "before" picture (garden `images_8`, 2026-11-10)

Run: `python train.py -s /data/manousis/360_v2/garden -i images_8 -m output/physproxy/baseline_garden8 --eval` (local RTX 3060, about 28 min), then `create_ply.py` → `output/physproxy/baseline_garden8/render_mesh.ply`.

- 396,680 opaque triangles on 803,883 vertices, so most triangles do not share vertices with any other. 178,677 components are a single triangle, and the largest component has only 20 faces.
- No non-manifold edges and no degenerate or duplicate faces. Winding is meaningless, because the indices are stored sorted.
- So the render mesh is a **triangle soup** and unusable as a collider. Raycasts can pass through the gaps between triangles, and it encloses no volume. This is the gap the SDF proxy has to close.
