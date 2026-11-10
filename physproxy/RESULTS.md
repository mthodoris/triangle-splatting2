# physproxy results

Rendering metrics are on the 24 test views (`--eval`, every 8th image) via `render.py` + `metrics.py`. Mesh metrics come from `physproxy/eval/mesh_stats.py`. Hausdorff and raycast columns get filled from Phase 5/6 on.

| Scene | Method | Commit | PSNR | SSIM | LPIPS | Triangles | Components | Largest comp. (area %) | Boundary edges | Non-manifold edges | Watertight | Hausdorff | Raycast agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| garden `images_8` | TS+ baseline, upstream defaults, 30k | `4f0af82` (main) | 25.34 | 0.801 | 0.192 | 396,680 | 250,382 | 0.05 % (20 faces) | 895,596 (85.9 %) | 0 | no | — | — |
| garden `images_8` | 4a one-way SDF proxy (render mesh unchanged) | `0cfa3d7` | 25.33 | 0.801 | 0.192 | proxy (MC 256, not decimated): 1,008,520 | 719 | 99.6 % | 0 | 0 | **yes** | render→proxy p50 0.007 / p90 0.027 | see-what-you-hit: agree 81.8 %, blocks 17.3 %, see-through 0.9 % |

## Baseline "before" picture (garden `images_8`, 2026-11-10)

Run: `python train.py -s /data/manousis/360_v2/garden -i images_8 -m output/physproxy/baseline_garden8 --eval` (local RTX 3060, about 28 min), then `create_ply.py` → `output/physproxy/baseline_garden8/render_mesh.ply`.

- 396,680 opaque triangles on 803,883 vertices, so most triangles do not share vertices with any other. 178,677 components are a single triangle, and the largest component has only 20 faces.
- No non-manifold edges and no degenerate or duplicate faces. Winding is meaningless, because the indices are stored sorted.
- So the render mesh is a **triangle soup** and unusable as a collider. Raycasts can pass through the gaps between triangles, and it encloses no volume. This is the gap the SDF proxy has to close.
