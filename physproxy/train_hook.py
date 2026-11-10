"""Joint training of the SDF proxy next to Triangle Splatting+ (Phase 4).

Driven by physproxy/train_proxy.py, which runs the UNMODIFIED upstream
train.training() and wraps its render/optimizer at runtime (no upstream edits):
  ProxyTrainer(args, scene, triangles, model_path)   when the scene is built
  proxy.step(iteration, view, ids, alpha, V, F)      once per iteration, before the
        TS+ optimizer step; (ids, alpha, V, F) are snapshots taken when the
        training view was rendered, so ids match those faces even if pruning
        has replaced the triangle tensors since
  proxy.finish(iteration, scene, triangles, pipe, bg) after training() returns

Phase 4a (one-way): the SDF learns from the triangles; vertex positions are
detached, so the TS+ optimization is unchanged except for timing.
The SDF has its own Adam optimizer and steps once (or proxy_steps_per_iter
times) per TS+ iteration from proxy_start on, using the exact depth of the view
being trained.

Outputs in <model_path>/proxy/: physproxy_log.jsonl (via JsonLogger, in the
model path), sdf_<iter>.pt, mesh_<iter>.ply, eval_<iter>.json.
"""
import json
import os
import random
import time

import numpy as np
import torch
import trimesh

from physproxy import losses as L
from physproxy.aabb import camera_centers, camera_up_vectors, oriented_proxy_box
from physproxy.depth import exact_depth, pixel_alpha
from physproxy.eval.mesh_stats import mesh_stats
from physproxy.extract import extract_mesh
from physproxy.log import JsonLogger
from physproxy.sdf import SDFField


def _load_points(model_path):
    from scene.dataset_readers import fetchPly
    return np.asarray(fetchPly(os.path.join(model_path, "input.ply")).points, dtype=np.float32)


class ProxyTrainer:
    def __init__(self, args, scene, triangles, model_path):
        assert args.proxy_coupling == "one_way", "two_way coupling is Phase 4b"
        self.args = args
        self.out = os.path.join(model_path, "proxy")
        os.makedirs(self.out, exist_ok=True)
        cams = scene.getTrainCameras()
        pts = _load_points(model_path)
        frame, lo, hi = oriented_proxy_box(pts, camera_centers(cams), camera_up_vectors(cams),
                                           radius_mult=args.proxy_box_radius_mult)
        kw = dict(frame=frame, base_res=args.proxy_base_res, n_levels=args.proxy_levels,
                  feat_dim=args.proxy_feat, hidden=args.proxy_hidden)
        ground = None
        if args.proxy_init == "plane_ground":
            # Plane with normal = local up through a low percentile of the in-box SfM heights.
            pl = torch.as_tensor(pts) @ frame.T
            inb = ((pl >= lo) & (pl <= hi)).all(-1)
            ground = float(torch.quantile(pl[inb][:, 2], args.proxy_ground_pct / 100))
            self.sdf = SDFField(lo, hi, init="plane", init_offset=ground, **kw)
        elif args.proxy_init == "plane_center":
            self.sdf = SDFField(lo, hi, init="plane", init_offset=float(0.5 * (lo[2] + hi[2])), **kw)
        else:
            self.sdf = SDFField(lo, hi, init=args.proxy_init, **kw)
        self.sdf = self.sdf.cuda()
        self.opt = torch.optim.Adam(self.sdf.param_groups(args.proxy_lr_grid, args.proxy_lr_mlp))
        self.cfg = L.ProxyLossConfig(
            w_surf=args.proxy_w_surf, w_free=args.proxy_w_free, w_occ=args.proxy_w_occ, w_eik=args.proxy_w_eik,
            w_normal=args.proxy_w_normal, w_simple=args.proxy_w_simple, simple_mode=args.proxy_simple_mode,
            margin=args.proxy_margin, thickness=args.proxy_thickness, delta=args.proxy_delta,
            n_surf=args.proxy_n_surf, n_rays=args.proxy_n_rays, n_per_ray=args.proxy_n_per_ray)
        self.log = JsonLogger(model_path, config=dict(vars(args)))
        self.log.log(0, "setup", box_lo=lo, box_hi=hi, frame=frame, ground_z_local=ground,
                     grid_shapes=self.sdf.grid_shapes, n_params=self.sdf.num_params(),
                     voxel=self.sdf.voxel_size, loss_config=self.cfg.as_dict())
        self.n_steps = 0
        self._t_acc = 0.0
        print(f"[physproxy] SDF proxy on: box {[round(float(x), 2) for x in hi - lo]} (local), "
              f"grid {self.sdf.grid_shapes[-1]}, {self.sdf.num_params() / 1e6:.1f}M params, "
              f"ground z {ground}, start at {args.proxy_start}")

    # ------------------------------------------------------------------ training
    def _sdf_step(self, iteration, view, ids, alpha, V, Fc):
        t0 = time.time()
        V = V.detach()
        with torch.no_grad():
            dep = exact_depth(view, V, Fc, ids.detach(), alpha.detach())
        for _ in range(self.args.proxy_steps_per_iter):
            s = L.build_samples(self.cfg, self.sdf, V, Fc, view, dep)
            total, terms = L.proxy_losses(self.cfg, self.sdf, s)
            self.opt.zero_grad(set_to_none=True)
            total.backward()
            self.opt.step()
            self.n_steps += 1
        torch.cuda.synchronize()
        self._t_acc += time.time() - t0
        if self.n_steps % self.args.proxy_log_every == 0:
            self.log.log(iteration, "loss", total=total, step=self.n_steps,
                         ms_per_step=1000 * self._t_acc / self.args.proxy_log_every,
                         n_surf=s.get("surf", torch.zeros(0)).shape[0], n_free=s.get("free", torch.zeros(0)).shape[0],
                         n_occ=s.get("occ", torch.zeros(0)).shape[0], valid_depth_frac=dep["valid"].float().mean(),
                         **terms)
            self._t_acc = 0.0

    def step(self, iteration, view, ids, alpha, V, Fc, triangles):
        if iteration < self.args.proxy_start:
            return
        with torch.enable_grad():   # upstream calls optimizer.step() under torch.no_grad()
            self._sdf_step(iteration, view, ids, alpha, V, Fc)
        if self.args.proxy_mesh_every > 0 and iteration % self.args.proxy_mesh_every == 0:
            self.export(iteration, triangles)

    def finish(self, iteration, scene, triangles, pipe, bg):
        from triangle_renderer import render
        n = self.args.proxy_extra_iters
        if n > 0:
            print(f"[physproxy] {n} SDF-only steps on the final triangles")
            cams = scene.getTrainCameras()
            stack = []
            for k in range(n):
                if not stack:
                    stack = list(range(len(cams)))
                    random.shuffle(stack)
                view = cams[stack.pop()]
                with torch.no_grad():
                    pkg = render(view, triangles, pipe, bg)
                self._sdf_step(iteration + k + 1, view, pkg["rend_ids"], pixel_alpha(pkg),
                               triangles.get_vertices, triangles.get_triangle_indices)
        self.export(iteration + n, triangles, final=True)
        self.log.close()

    # ------------------------------------------------------------------ export + eval
    @torch.no_grad()
    def export(self, iteration, triangles, final=False):
        t0 = time.time()
        torch.save({"state_dict": self.sdf.state_dict(), "grid_shapes": self.sdf.grid_shapes,
                    "ctor_kwargs": self.sdf.ctor_kwargs,
                    "args": vars(self.args)}, os.path.join(self.out, f"sdf_{iteration}.pt"))
        v, f = extract_mesh(self.sdf, res=self.args.proxy_mesh_res)
        ev = {"iteration": iteration, "final": final}
        if f.shape[0] > 0:
            trimesh.Trimesh(v.cpu().numpy(), f.cpu().numpy(), process=False).export(
                os.path.join(self.out, f"mesh_{iteration}.ply"))
            st = mesh_stats(v.double().cpu().numpy(), f.cpu().numpy())
            ev["mesh"] = {k: st[k] for k in ["faces", "vertices", "components", "watertight", "winding_consistent",
                                              "boundary_edges", "nonmanifold_edges", "largest_component_area_frac", "volume"]
                          if k in st}
            ev.update(self.surface_agreement(v, triangles))
        ev["export_s"] = time.time() - t0
        with open(os.path.join(self.out, f"eval_{iteration}.json"), "w") as fh:
            json.dump(ev, fh, indent=1, default=float)
        self.log.log(iteration, "export", **{k: v for k, v in ev.items() if not isinstance(v, dict) and k != "iteration"},
                     **{f"mesh_{k}": v for k, v in ev.get("mesh", {}).items()})
        print(f"[physproxy] it {iteration}: proxy mesh {ev.get('mesh', {}).get('faces', 0)} faces, "
              f"watertight {ev.get('mesh', {}).get('watertight')}, "
              f"render->proxy p50/p90 {ev.get('render_to_proxy_p50', float('nan')):.4f}/{ev.get('render_to_proxy_p90', float('nan')):.4f}, "
              f"proxy->render p50/p90 {ev.get('proxy_to_render_p50', float('nan')):.4f}/{ev.get('proxy_to_render_p90', float('nan')):.4f}")
        return ev

    @torch.no_grad()
    def surface_agreement(self, proxy_verts, triangles, n=500_000):
        """Distances between the render surface (in-box samples) and the proxy (scene units).
        render->proxy: |f| at render-surface samples (how well the proxy covers what is seen).
        proxy->render: proxy vertex -> nearest render sample (floaters / extra proxy surface;
        large values also appear where the proxy closes holes or caps the box walls)."""
        from pytorch3d.ops import knn_points
        V = triangles.get_vertices.detach()
        Fc = triangles.get_triangle_indices
        tl = self.sdf.to_local(V)[Fc.long()]
        in_box = ((tl.max(1).values >= self.sdf.aabb_min) & (tl.min(1).values <= self.sdf.aabb_max)).all(-1)
        p, _, _ = L.sample_surface(V, Fc, n, in_box)
        p = p[self.sdf.inside_aabb(p)]
        r2p = torch.cat([self.sdf(c).abs() for c in p.split(2 ** 18)])
        q = proxy_verts
        if q.shape[0] > 400_000:
            q = q[torch.randperm(q.shape[0], device=q.device)[:400_000]]
        d = knn_points(q[None].float(), p[None].float(), K=1).dists[0, :, 0].sqrt()
        out = {}
        for name, x in [("render_to_proxy", r2p), ("proxy_to_render", d)]:
            out[f"{name}_p50"] = float(x.median())
            out[f"{name}_p90"] = float(torch.quantile(x[:1_000_000].float(), 0.9))
            out[f"{name}_mean"] = float(x.mean())
        out["render_within_voxel_frac"] = float((r2p < float(self.sdf.voxel_size.max())).float().mean())
        return out
