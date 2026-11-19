# physproxy in Unity (Phase 7)

These scripts load a trained scene into Unity: the render mesh as the visual, and a collision mesh as a static `MeshCollider`. They then run repeatable drop and walking tests against several colliders and write CSV files.

They were written without access to a Unity install. They use only stable APIs: local physics scenes and `CharacterController`, both available since 2018.3. They target Unity 2021 LTS or newer, including Unity 6 with URP (the user's project is 6000.5 + URP). They support both the new Input System and the legacy Input Manager, and use `linearVelocity` on Unity 6.

## 1. Export on the training machine

```bash
python unity/export_for_unity.py -s /data/manousis/360_v2/garden -i images_8 \
    -m output/physproxy/p4b_R0_oneway
```

This writes `MODEL/unity/`, about 20 MB:

| File | Content |
|---|---|
| `render.ppm` | full render mesh with vertex colors (visual) |
| `collision.ppm` | our collision mesh |
| `naive.ppm` | naive cleanup of the render mesh |
| `soup.ppm` | render triangles inside the box, used as a collider as-is |
| `drops.csv` | sphere start points (space the cameras see as empty) and the height of the visible surface below each |
| `walks.csv` | start and goal points on the ground |
| `meta.json` | frame, scale, provenance |

**Frame.** Unity is Y-up and left-handed. y = 0 is the bottom of the proxy box, and x/z are centred on the box. `--units_to_m` converts scene units to metres; gravity is 9.81 m/s², so the scale matters. By default it comes from `physproxy/eval/metric_scale.py`, which compares Metric3D metric depth with the render mesh. On garden that gives 0.579 m per unit (24 views, p10–p90 0.53–0.64), and the table then measures 0.78 m tall and 1.8 m across.

## 2. Unity project

1. Copy `unity/Assets/PhysProxy/` into your project's `Assets/`.
2. Copy `MODEL/unity/*` into `Assets/StreamingAssets/physproxy/garden/`.
3. **Viewer:** add an empty GameObject with `PhysProxyViewer`, then press Play. Space throws a ball from the camera, R drops a rain of 50 balls over the whole box, C shows or hides the collider, and 1/2/3 switch between collision, naive and soup. Put the camera somewhere around (0, 1.5, −4) looking at the table.
   **First person:** also add `PhysProxyFirstPerson` to the same GameObject. In Play mode, F switches to first person: WASD to move, mouse to look, Shift to run, Space to jump, left click to throw a ball, Esc to free the mouse. You walk on whichever collider is active (1/2/3).
4. **Benchmark:** add an empty GameObject with `PhysProxyBenchmark` and press Play. The Console reports each step, and the CSVs go to `Application.persistentDataPath` unless `outputFolder` is set.

## 3. Summarise

```bash
python unity/summarize.py <folder with drops_*.csv and walks_*.csv>
```

## What the tests measure

- **Drop.** Spheres (radius 0.05 m, continuous collision detection) are released from every `drops.csv` point and simulated for 4 s at a fixed 0.01 s step.
  - Fell through: the sphere ended below y = −0.5.
  - Rest height: for spheres that did not roll away, the rest height minus the visible surface below the start point.
  - Started overlapping: the start point was already inside the collider.
- **Walk.** A `CharacterController` capsule (radius 0.2 m, height 1 m, step 0.15 m, slope 45°) walks straight at 1 m/s with gravity, for up to 12 s.
  - Reached: got within one radius of the goal.
  - Stuck: less than 0.05 m of progress in 2 s.
  - Fell through: ended below y = −0.5.
- Each collider runs in its own local physics scene. Inputs are fixed and nothing is random, so repeated runs on the same machine give the same results.
