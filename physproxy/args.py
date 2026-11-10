"""Command-line flags for the collision proxy. Everything is off unless --proxy."""
from arguments import ParamGroup


class ProxyParams(ParamGroup):
    def __init__(self, parser):
        self.proxy = False                 # master switch; off = upstream behaviour, bit for bit
        self.proxy_start = 20000           # first iteration with SDF steps (triangles mostly opaque by then)
        self.proxy_extra_iters = 0         # SDF-only steps after the TS+ loop (triangles frozen)
        self.proxy_steps_per_iter = 1
        self.proxy_coupling = "one_way"    # one_way (4a): triangles detached; two_way (4b): pull + floater pruning below
        # Phase 4b (two_way only), from proxy_couple_start on
        self.proxy_couple_start = 23000    # after the SDF has converged (~2.5k steps after proxy_start)
        self.proxy_pull_w = 0.0            # lambda for mean|f(v)| back-propagated into the vertices (SDF frozen)
        self.proxy_pull_tau = 0.1          # only vertices in the box with |f| < tau are pulled
        self.proxy_prune_floaters = False
        self.proxy_prune_every = 500       # run at iteration % every == every/2 (between upstream's prunes)
        self.proxy_prune_tau = 0.1         # criterion 1: |f(centroid)| > tau
        self.proxy_prune_comp_area = 0.05  # criterion 2: on a proxy component with area < this (scene units^2)
        self.proxy_prune_max_support = 3   # and seen as the visible triangle in <= this many training renders
        self.proxy_prune_mesh_res = 256    # marching cubes resolution for the component test
        # Field and box
        self.proxy_init = "plane_ground"   # plane_ground | plane_center | sphere | none
        self.proxy_ground_pct = 5.0        # ground height = this percentile of in-box SfM heights
        self.proxy_base_res = 32           # finest level 256 along the longest axis (garden voxel ~0.044)
        self.proxy_levels = 4
        self.proxy_feat = 4
        self.proxy_hidden = 64
        self.proxy_box_radius_mult = 1.5
        self.proxy_lr_grid = 1e-2
        self.proxy_lr_mlp = 1e-3
        # Losses (see physproxy/losses.py); a weight of 0 switches a term off
        self.proxy_w_surf = 1.0
        self.proxy_w_free = 2.0
        self.proxy_w_occ = 0.5
        self.proxy_w_eik = 0.1
        self.proxy_w_normal = 0.05
        self.proxy_w_simple = 0.0
        self.proxy_simple_mode = "curvature"
        self.proxy_margin = 0.02
        self.proxy_thickness = 0.03         # thin: 0.2 made legs / table top / flowers fat (bloat sweep 2026-11-10)
        self.proxy_delta = 0.02
        self.proxy_n_surf = 32768
        self.proxy_n_rays = 16384
        self.proxy_n_per_ray = 8
        self.proxy_free_band_frac = 0.5    # share of free-space samples in a band just before the surface
        self.proxy_free_band = 0.3         # that band's width along the ray, scene units
        # Logging / export
        self.proxy_log_every = 100
        self.proxy_mesh_every = 0          # export the zero level set every N its (0 = only at the end)
        self.proxy_mesh_res = 384          # marching-cubes cells along the longest box axis
        super().__init__(parser, "Collision proxy (physproxy)")
