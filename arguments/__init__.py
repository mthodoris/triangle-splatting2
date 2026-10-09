#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.mesh_path = ""
        self.free_triangles = False # with --mesh_path: add free triangles from SfM points away from the mesh
        self.free_min_dist = 3.0 # SfM points farther than this many init mean edges from the mesh seed free triangles
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 1.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.lambda_dssim = 0.2

        self.densification_interval = 500

        self.densify_from_iter = 500
        self.densify_until_iter = 13000

        self.random_background = False
        
        self.feature_lr = 0.0016 # 0.0025
        self.max_points = 3000000

        # Opacity & weight
        self.set_weight = 0.28
        self.weight_lr =  0.03
        self.lambda_weight = 1.9e-06

        # Normal loss
        self.iteration_mesh = 5000
        self.lambda_normals = 0.05

        # Mesh regularization (all off by default, active after iteration_mesh)
        self.lambda_laplacian = 0.0
        self.lambda_normal_consistency = 0.0
        self.lambda_edge_smooth = 0.0
        self.crease_sigma = 0.1 # dihedral falloff: smaller keeps more creases sharp (90 deg -> weight ~0.05)
        self.edge_alpha = 10.0 # image-gradient falloff for edge-aware smoothness
        self.lambda_tangential_laplacian = 0.0 # Laplacian along the surface only (needs --mesh_path)
        self.normal_only_motion = False # vertices move only along their init normal (needs --mesh_path)
        self.max_normal_offset = 5.0 # with --normal_only_motion: max distance from the init position, in init mean edges

        self.add_percentage = 1.23

        # PARAMETER FIRST STAGE
        self.set_sigma = 1.0
        self.set_weight = 0.28

        # Add new triangles or vertices
        self.intervall_add_triangles = 500

        # Prune triangles and vertices
        self.prune_triangles_threshold = 0.235

        # PARAMETER SECOND STAGE
        self.lr_triangles_points_init = 0.0015

        self.start_opacity_floor = 5000

        self.start_pruning = 4000
        self.no_prune = False
        self.sigma_until = 30000
        self.final_opacity_iter = 24000
        self.final_opacity = 0.9999 # opacity floor reached at final_opacity_iter (mesh group with --free_triangles)
        self.free_final_opacity = 0.9999 # same for the free group (--free_triangles)
        self.large_steps = False # smooth mesh-vertex updates, Nicolet et al. 2021 (needs --no_mesh_densify)
        self.ls_lambda = 20.0 # smoothing strength: x = (I + lambda L)^-1 u
        self.ls_lr_mult = 1.0 # mesh-vertex learning rate for large steps, as a multiple of the vertex lr schedule
        self.ls_cg_iters = 10 # conjugate-gradient iterations for the gradient solve (positions use 5, warm-started)
        self.free_exclusion = 0.0 # with --free_triangles: every 500 iterations delete free triangles within about this many init mean edges of the mesh (0 = off)
        self.mesh_refine = False # conforming longest-edge refinement of the mesh where color gradients are high (needs --no_mesh_densify)
        self.refine_from = 2000
        self.refine_until = 15000
        self.refine_interval = 1000 # multiple of 500
        self.refine_fraction = 0.05 # share of mesh triangles selected per refinement step
        self.refine_min_edge = 0.3 # do not split triangles whose longest edge is below this many init mean edges
        self.mesh_max_vertices = 4000000
        self.refine_min_inradius = 0.0 # only split triangles whose largest on-screen inradius (render pixels) reaches this; the rasterizer skips triangles under 1 (0 = off)
        self.no_mesh_densify = False # never split mesh-group triangles: the mesh keeps the init topology (with --free_triangles only free triangles are split)

        self.sigma_start = 0

        self.splitt_large_triangles = 100
        self.start_upsampling = 20000
        self.upscaling_factor = 2

        self.size_probs_zero = 7.5e-05
        self.size_probs_zero_image_space = 0.0

        self.prune_size = 1400



        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def update_indoor(params):
    params.add_percentage = 1.27
    params.densify_from_iter = 1000
    params.densify_until_iter = 25000
    params.feature_lr = 0.004
    params.size_probs_zero = 0.0
    params.splitt_large_triangles = 500
    params.start_pruning = 3000
    params.weight_lr = 0.05
    params.lambda_weight = 0.0
    params.lambda_normals = 0.025
    params.prune_size = 1300

    return params