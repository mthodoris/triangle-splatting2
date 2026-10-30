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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH, SH2RGB
from utils.graphics_utils import BasicPointCloud
import math
from pytorch3d.ops import knn_points
import triangulation
import trimesh
#import igl



class TriangleModel:

    def setup_functions(self):
        """self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid"""

        self.eps = 1e-6
        self.opacity_floor = 0.0
        self.opacity_activation = lambda x: self.opacity_floor + (1.0 - self.opacity_floor) * torch.sigmoid(x)
        # Matching inverse for any y in [m, 1): logit( (y - m)/(1 - m) )
        self.inverse_opacity_activation = lambda y: inverse_sigmoid(
            ((y.clamp(self.opacity_floor + self.eps, 1.0 - self.eps) - self.opacity_floor) /
            (1.0 - self.opacity_floor + self.eps))
        )

        self.exponential_activation = lambda x:math.exp(x)
        self.inverse_exponential_activation = lambda y: math.log(y)

    def __init__(self, sh_degree : int):

        self._triangles = torch.empty(0) # can be deleted eventually

        self.size_probs_zero = 0.0
        self.size_probs_zero_image_space = 0.0
        self.vertices = torch.empty(0)
        self._triangle_indices = torch.empty(0)
        self.vertex_weight = torch.empty(0)

        self._sigma = 0
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self.optimizer = None
        self.image_size = 0
        self.importance_score = 0
        self.add_percentage = 1.0

        # Per-vertex init position and normal (mesh init only), kept in sync with the
        # vertices through densification and pruning; used by --normal_only_motion and
        # --lambda_tangential_laplacian. init_edge: mean edge length of the init mesh.
        self._anchor = None
        self._anchor_normal = None
        self.init_edge = None

        # Two triangle groups (only with --free_triangles): the mesh group (init mesh and
        # its subdivisions) and free triangles seeded from SfM points the mesh does not
        # explain. Groups never share vertices. _vertex_is_mesh is None without groups.
        # free_opacity_floor is the free group's floor; opacity_floor is the mesh group's
        # (and everyone's, without groups).
        self._vertex_is_mesh = None
        self.free_opacity_floor = None
        self.free_max_triangles = 0 # >0: the free group never grows beyond this many triangles
        self.no_mesh_densify = False # never split mesh-group triangles (keeps the init topology)

        # Per-face texture (--tex_res): a residual RGB colour on a barycentric lattice of level tex_res inside
        # every triangle, K = (n+1)(n+2)/2 samples per face, added to the vertex colours by the rasterizer.
        self.tex_res = 0
        self._face_tex = torch.empty(0)
        self.free_behind = False # --free_behind: render the free group strictly behind the mesh group
        self.free_sigma = None # --free_sigma_init: the free group's own sigma (with --free_behind); None = shared sigma

        self.scaling = 1

        self.setup_functions()

    def save_parameters(self, path):

        mkdir_p(path)

        point_cloud_state_dict = {}

        point_cloud_state_dict["triangles_points"] = self.vertices
        point_cloud_state_dict["_triangle_indices"] = self._triangle_indices
        point_cloud_state_dict["vertex_weight"] = self.vertex_weight
        point_cloud_state_dict["sigma"] = self._sigma
        point_cloud_state_dict["active_sh_degree"] = self.active_sh_degree
        point_cloud_state_dict["features_dc"] = self._features_dc
        point_cloud_state_dict["features_rest"] = self._features_rest
        point_cloud_state_dict["importance_score"] = self.importance_score
        point_cloud_state_dict["image_size"] = self.image_size
        if self.tex_res > 0:
            point_cloud_state_dict["face_tex"] = self._face_tex
            point_cloud_state_dict["tex_res"] = self.tex_res
        point_cloud_state_dict["free_behind"] = self.free_behind
        if self.free_sigma is not None:
            point_cloud_state_dict["free_sigma"] = self.free_sigma
        point_cloud_state_dict["opacity_floor"] = self.opacity_floor
        if self._vertex_is_mesh is not None:
            point_cloud_state_dict["vertex_is_mesh"] = self._vertex_is_mesh
            point_cloud_state_dict["free_opacity_floor"] = self.free_opacity_floor

        torch.save(point_cloud_state_dict, os.path.join(path, 'point_cloud_state_dict.pt'))

   

    def load_parameters(self, path, device="cuda", segment=False, ratio_threshold = 0.25):
        # 1. Load the dict you saved
        state = torch.load(os.path.join(path, "point_cloud_state_dict.pt"), map_location=device)

        # 2. Restore everything you put in there (one line each)
        self.vertices            = state["triangles_points"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._triangle_indices   = state["_triangle_indices"].to(device).to(torch.int32)
        self.vertex_weight       = state["vertex_weight"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._sigma              = state["sigma"]
        self.active_sh_degree    = state["active_sh_degree"]
        self._features_dc        = state["features_dc"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._features_rest      = state["features_rest"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self.importance_score = state["importance_score"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        if state.get("tex_res", 0) > 0:
            self.tex_res = int(state["tex_res"])
            self._face_tex = state["face_tex"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self.free_behind = bool(state.get("free_behind", False))
        self.free_sigma = state.get("free_sigma", None)


        # For object extraction
        if segment:
            base = os.path.dirname(os.path.dirname(path))
            triangle_hits = torch.load(os.path.join(base, 'segmentation/triangle_hits_mask.pt'))
            triangle_hits_total = torch.load(os.path.join(base, 'segmentation/triangle_hits_total.pt'))

            min_hits = 1  

            # Handle division by zero - triangles with no renders get ratio 0
            triangle_ratio = torch.zeros_like(triangle_hits, dtype=torch.float32)
            valid_mask = triangle_hits_total > 0
            triangle_ratio[valid_mask] = triangle_hits[valid_mask].float() / triangle_hits_total[valid_mask].float()

            # Create the keep mask: triangles must meet both ratio and minimum hits criteria
            keep_mask = (triangle_ratio >= ratio_threshold) & (triangle_hits >= min_hits)
            #keep_mask = ~keep_mask

            with torch.no_grad():
                self._triangle_indices = self._triangle_indices[keep_mask]

        ################################################################


        # the floor the model was trained with (vertex_weight logits are relative to it);
        # older checkpoints did not store it and always ended training at 0.9999
        self.opacity_floor = state.get("opacity_floor", 0.9999)
        if "vertex_is_mesh" in state:
            self._vertex_is_mesh = state["vertex_is_mesh"].to(device).bool()
            self.free_opacity_floor = state["free_opacity_floor"]

        # 3. (Re)compute any derived quantities

        self._triangle_indices = self._triangle_indices.to(torch.int32)


        param_groups = [
            {'params': [self._features_dc], 'lr': 0.0, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': 0.0 / 20.0, "name": "f_rest"},
            {'params': [self.vertices], 'lr': 0.0, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': 0.0, "name": "vertex_weight"}
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")


    def capture(self):
        return (
            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self.optimizer.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._features_dc, 
        self._features_rest,
        opt_dict) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    @property
    def get_triangle_indices(self):
        return self._triangle_indices

    @property
    def get_triangles_points_flatten(self):
        return self._triangles.flatten(0)
  
    @property
    def get_triangles_points(self):
        return self._triangles

    @property
    def get_vertices(self):
        return self.vertices
    
    @property
    def get_sigma(self):
        return self.exponential_activation(self._sigma)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_vertex_weight(self):
        f = self.vertex_floor()
        return f + (1.0 - f) * torch.sigmoid(self.vertex_weight)

    def vertex_floor(self, idx=None):
        """Opacity floor per vertex as [V,1] (or for vertices idx), or the scalar floor without groups."""
        if self._vertex_is_mesh is None:
            return self.opacity_floor
        is_mesh = self._vertex_is_mesh if idx is None else self._vertex_is_mesh[idx]
        return torch.where(is_mesh, self.opacity_floor, self.free_opacity_floor).unsqueeze(-1).to(torch.float32)

    def inverse_opacity(self, y, floor):
        """Logit that gives opacity y under floor (scalar or per-vertex [V,1]); same as the scalar inverse_opacity_activation."""
        lo = floor + self.eps if torch.is_tensor(floor) else torch.full_like(y, floor + self.eps)
        y = torch.maximum(y, lo).clamp(max=1.0 - self.eps)
        return inverse_sigmoid((y - floor) / (1.0 - floor + self.eps))

    def triangle_is_mesh(self):
        """[T] bool: triangle belongs to the mesh group (all True without groups)."""
        if self._vertex_is_mesh is None:
            return torch.ones(self._triangle_indices.shape[0], dtype=torch.bool, device=self._triangle_indices.device)
        return self._vertex_is_mesh[self._triangle_indices[:, 0].long()]

    @torch.no_grad()
    def remove_free_near_mesh(self, dist):
        """Delete free triangles whose centroid lies in a voxel (edge dist) touched by the mesh or next to one,
        i.e. roughly within dist..2*dist of the current mesh surface, and their unused vertices. Keeps free
        triangles in the background so the mesh has to carry the appearance itself. Returns the number removed."""
        if self._vertex_is_mesh is None or dist <= 0:
            return 0
        F = self._triangle_indices.long()
        is_mesh = self._vertex_is_mesh[F[:, 0]]
        free = torch.nonzero(~is_mesh, as_tuple=True)[0]
        if free.numel() == 0:
            return 0
        V = self.vertices.detach()
        # mesh surface samples: vertices and face centroids (spacing about one edge, finer than dist)
        mesh_pts = torch.cat([V[self._vertex_is_mesh], V[F[is_mesh]].mean(1)], dim=0)
        lo = mesh_pts.min(0).values - 2 * dist
        def keys(p):
            c = torch.floor((p - lo) / dist).long()
            return (c[:, 0] * 2097152 + c[:, 1]) * 2097152 + c[:, 2]  # 21 bits per axis
        occupied = torch.unique(keys(mesh_pts))
        cen = V[F[free]].mean(1)
        near = torch.zeros(free.numel(), dtype=torch.bool, device=V.device)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    k = keys(cen + torch.tensor([dx, dy, dz], device=V.device, dtype=V.dtype) * dist)
                    pos = torch.searchsorted(occupied, k).clamp(max=occupied.numel() - 1)
                    near |= occupied[pos] == k
        if not near.any():
            return 0
        keep = torch.ones(F.shape[0], dtype=torch.bool, device=V.device)
        keep[free[near]] = False
        self.prune_triangles(keep)
        used = torch.zeros(V.shape[0], dtype=torch.bool, device=V.device)
        used[self._triangle_indices.flatten().long()] = True
        self._prune_vertices(used | self._vertex_is_mesh)
        return int(near.sum().item())

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def create_from_pcd(self, pcd : BasicPointCloud, opacity : float, set_sigma : float):

        # we remove all points that are too close to each other. Otherwise, this somehow gives an oom
        pcd_points = np.asarray(pcd.points)
        pcd_points = np.round(pcd_points, decimals=6)
        _, unique_indices = np.unique(pcd_points, axis=0, return_index=True)
        pcd_points = pcd_points[np.sort(unique_indices)]
        _points = torch.tensor(pcd_points).float().cuda()

        n = _points.shape[0]
        
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)[:n]).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        ################################
        # RUN DELAUNAY TRIANGULATION   #
        ################################
        dt = triangulation.Triangulation(_points)
        perm = dt.permutation().to(torch.long)
        _points = _points[perm]
        features = features[perm]
        
        tets = dt.tets()     
        tets = dt.tets().to(torch.int64)

        faces = torch.cat([
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ], dim=0)  # [4 * num_tets, 3]

        
        # Step 3: Sort to ignore winding order
        faces, _ = torch.sort(faces, dim=1)
        faces = torch.unique(faces, dim=0)

        # finally stash on your module
        self.vertices          = nn.Parameter(_points.requires_grad_(True))
        self._triangle_indices = faces.to(torch.int32)                         # [T,3]
        vert_weight = inverse_sigmoid(opacity * torch.ones((self.vertices.shape[0], 1), dtype=torch.float, device="cuda")) 
        self.vertex_weight = nn.Parameter(vert_weight.requires_grad_(True))

        # Sigma should be very low such that the triangles are solid. No need for soft triangles.
        self._sigma = self.inverse_exponential_activation(set_sigma)


        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")


    def create_from_mesh(self, mesh_path : str, opacity : float, set_sigma : float, free_pcd : BasicPointCloud = None, free_min_dist : float = 3.0):
        """Initialize the triangle model from a predefined mesh instead of running
        Delaunay triangulation on a point cloud. The mesh's own vertices and faces
        are used directly, so its topology is preserved."""

        mesh = trimesh.load(mesh_path, process=False, force="mesh")

        # Drop vertices no face uses. With --free_triangles the "prune unused vertices" step would delete them at the
        # first pruning iteration, changing the number of mesh vertices after the Large Steps operator was built.
        used = np.zeros(len(mesh.vertices), dtype=bool)
        used[np.asarray(mesh.faces).reshape(-1)] = True
        if not used.all():
            remap = np.full(len(used), -1, dtype=np.int64)
            remap[used] = np.arange(int(used.sum()))
            colors = np.asarray(mesh.visual.vertex_colors)[used] if mesh.visual.kind == "vertex" else None
            print("Mesh: dropped {} of {} vertices that no face uses".format(int((~used).sum()), used.size))
            mesh = trimesh.Trimesh(np.asarray(mesh.vertices)[used], remap[np.asarray(mesh.faces)], vertex_colors=colors, process=False)

        _points = torch.tensor(np.asarray(mesh.vertices)).float().cuda()
        faces = torch.tensor(np.asarray(mesh.faces)).to(torch.int64).cuda()

        if mesh.visual is not None and getattr(mesh.visual, "vertex_colors", None) is not None:
            vertex_colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
        else:
            vertex_colors = np.full((_points.shape[0], 3), 0.5, dtype=np.float32)

        num_mesh_vertices = _points.shape[0]
        anchor_normal = torch.nn.functional.normalize(torch.tensor(np.asarray(mesh.vertex_normals)).float().cuda(), dim=1)
        init_edge = float(mesh.edges_unique_length.mean())

        if free_pcd is not None:
            # Free triangles: one small, randomly oriented triangle per SfM point farther than
            # free_min_dist init edges from the mesh (nearest mesh vertex or face centroid).
            # Its circumradius is half the mean distance to the point's 3 nearest SfM
            # neighbours. (Delaunay on these sparse leftover points gives huge triangles
            # spanning the scene.)
            pts = np.round(np.asarray(free_pcd.points), decimals=6)
            _, unique_indices = np.unique(pts, axis=0, return_index=True)
            unique_indices = np.sort(unique_indices)
            pts = torch.tensor(pts[unique_indices]).float().cuda()
            cols = np.asarray(free_pcd.colors)[unique_indices].astype(np.float32)
            spacing = knn_points(pts[None], pts[None], K=4).dists[0, :, 1:].sqrt().mean(1)
            surface = torch.cat([_points, torch.tensor(np.asarray(mesh.triangles_center)).float().cuda()], dim=0)
            dist = knn_points(pts[None], surface[None], K=1).dists[0, :, 0].sqrt()
            keep = dist > free_min_dist * init_edge
            pts, spacing, cols = pts[keep], spacing[keep], cols[keep.cpu().numpy()]
            print("Free triangles: {} of {} SfM points are more than {} init edges from the mesh".format(
                pts.shape[0], keep.numel(), free_min_dist))
            if pts.shape[0] > 0:
                gen = torch.Generator(device="cuda").manual_seed(0)
                n = torch.nn.functional.normalize(torch.randn(pts.shape[0], 3, device="cuda", generator=gen), dim=1)
                helper = torch.where(n[:, :1].abs() < 0.9, torch.tensor([1.0, 0.0, 0.0], device="cuda"), torch.tensor([0.0, 1.0, 0.0], device="cuda"))
                e1 = torch.nn.functional.normalize(torch.cross(n, helper, dim=1), dim=1)
                e2 = torch.cross(n, e1, dim=1)
                r = (0.5 * spacing)[:, None]
                angles = torch.tensor([0.0, 2.0943951, 4.1887902], device="cuda")
                corners = torch.stack([pts + r * (torch.cos(t) * e1 + torch.sin(t) * e2) for t in angles], dim=1)  # [P,3,3]
                free_points = corners.reshape(-1, 3)
                free_faces = torch.arange(free_points.shape[0], device="cuda").reshape(-1, 3)
                print("Free triangles: {} triangles, median circumradius {:.2f} init edges".format(
                    free_faces.shape[0], (r.median() / init_edge).item()))
                _points = torch.cat([_points, free_points], dim=0)
                faces = torch.cat([faces, free_faces + num_mesh_vertices], dim=0)
                vertex_colors = np.concatenate([vertex_colors, np.repeat(cols, 3, axis=0)], axis=0)
                anchor_normal = torch.cat([anchor_normal, torch.zeros_like(free_points)], dim=0)  # unused for free vertices
            self._vertex_is_mesh = torch.arange(_points.shape[0], device="cuda") < num_mesh_vertices
            self.free_opacity_floor = self.opacity_floor

        fused_color = RGB2SH(torch.tensor(vertex_colors).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        self.vertices          = nn.Parameter(_points.requires_grad_(True))
        self._triangle_indices = faces.to(torch.int32)
        vert_weight = inverse_sigmoid(opacity * torch.ones((self.vertices.shape[0], 1), dtype=torch.float, device="cuda"))
        self.vertex_weight = nn.Parameter(vert_weight.requires_grad_(True))

        self._sigma = self.inverse_exponential_activation(set_sigma)

        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")

        self._anchor = _points.detach().clone()
        self._anchor_normal = anchor_normal
        self.init_edge = init_edge


    def project_to_anchor_normals(self, max_offset):
        """Move every vertex back onto the line through its anchor along the anchor normal,
        at most max_offset away from the anchor (normal-only vertex motion)."""
        with torch.no_grad():
            offset = ((self.vertices - self._anchor) * self._anchor_normal).sum(1, keepdim=True).clamp(-max_offset, max_offset)
            projected = self._anchor + offset * self._anchor_normal
            if self._vertex_is_mesh is not None:
                projected = torch.where(self._vertex_is_mesh[:, None], projected, self.vertices)
            self.vertices.copy_(projected)


    def extract_mesh(self, path, iteration):
        """Export the current triangles as a colored mesh, using the trained
        per-vertex SH DC coefficients as vertex colors."""

        mesh_path = os.path.join(path, "mesh", "iteration_{}".format(iteration))
        mkdir_p(mesh_path)

        vertices = self.vertices.detach().cpu().numpy()
        faces = self._triangle_indices.detach().cpu().numpy().astype(np.int64)
        if self._vertex_is_mesh is not None:
            # mesh.ply holds the mesh group only; all_triangles.ply holds both groups
            vertex_colors_all = (SH2RGB(self._features_dc.detach().squeeze(1)).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
            trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors_all, process=False).export(
                os.path.join(mesh_path, "all_triangles.ply"))
            faces = faces[self.triangle_is_mesh().cpu().numpy()]

        vertex_colors = SH2RGB(self._features_dc.detach().squeeze(1)).clamp(0.0, 1.0)
        vertex_colors = (vertex_colors.cpu().numpy() * 255.0).astype(np.uint8)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
        mesh.export(os.path.join(mesh_path, "mesh.ply"))

        if self.tex_res > 0:
            self.export_textured_mesh(os.path.join(mesh_path, "mesh_textured.ply"), faces)

        return os.path.join(mesh_path, "mesh.ply")

    @torch.no_grad()
    def export_textured_mesh(self, path, faces):
        """Write the mesh with its per-face texture baked in: every face split along its texture lattice into
        tex_res^2 sub-triangles, coloured with the vertex colour plus the texture at each lattice sample.
        Lattice points shared by neighbouring faces (on edges and vertices) are merged by topology, so the result
        stays one connected mesh; at merged points the first face's colour and position are kept."""
        n = self.tex_res
        F_all = self._triangle_indices.long()
        keep = self.triangle_is_mesh()
        Fm = F_all[keep]
        tex = self._face_tex.detach()[keep]
        lat = self.tex_lattice(n)  # [K,2]
        w = torch.stack([1 - lat.sum(1) / n, lat[:, 0] / n, lat[:, 1] / n], 1)  # [K,3]
        V = self.vertices.detach()
        col_v = SH2RGB(self._features_dc.detach().squeeze(1))
        pos = torch.einsum("kc,fcd->fkd", w, V[Fm])  # [F,K,3]
        col = (torch.einsum("kc,fcd->fkd", w, col_v[Fm]) + tex).clamp(0, 1)
        # sub-triangles in lattice coordinates, keeping the parent's winding
        index = {(int(i), int(j)): k for k, (i, j) in enumerate(lat.tolist())}
        sub = []
        for i in range(n):
            for j in range(n - i):
                sub.append((index[(i, j)], index[(i + 1, j)], index[(i, j + 1)]))
                if i + j <= n - 2:
                    sub.append((index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]))
        sub = torch.tensor(sub, device=V.device)  # [n^2,3]
        Fcount, K = pos.shape[0], pos.shape[1]
        # merge lattice points by topology, not by position (rounding positions leaves seams at 1/3 points):
        # a corner is keyed by its mesh vertex, an edge point by its edge and step from the lower vertex id,
        # an interior point by its face
        cnt = torch.stack([n - lat.sum(1), lat[:, 0], lat[:, 1]], 1).long()  # [K,3] integer barycentrics
        fid = torch.arange(Fcount, device=V.device)
        keys = torch.zeros((Fcount, K, 4), dtype=torch.long, device=V.device)
        for k in range(K):
            nz = (cnt[k] > 0).nonzero().flatten().tolist()
            if len(nz) == 1:
                keys[:, k, 1] = Fm[:, nz[0]]
            elif len(nz) == 2:
                va, vb = Fm[:, nz[0]], Fm[:, nz[1]]
                keys[:, k, 0] = 1
                keys[:, k, 1] = torch.minimum(va, vb)
                keys[:, k, 2] = torch.maximum(va, vb)
                keys[:, k, 3] = torch.where(va < vb, cnt[k, nz[0]], cnt[k, nz[1]])
            else:
                keys[:, k, 0] = 2
                keys[:, k, 1] = fid
                keys[:, k, 2] = k
        uniq, inv = torch.unique(keys.reshape(-1, 4), dim=0, return_inverse=True)
        first = torch.full((uniq.shape[0],), inv.numel(), dtype=torch.long, device=V.device)
        first.scatter_reduce_(0, inv, torch.arange(inv.numel(), device=V.device), reduce="amin")
        tri = inv.reshape(Fcount, K)[fid[:, None, None], sub[None]].reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=pos.reshape(-1, 3)[first].cpu().numpy(), faces=tri.cpu().numpy(),
                               vertex_colors=(col.reshape(-1, 3)[first].cpu().numpy() * 255).astype(np.uint8), process=False)
        mesh.export(path)
        return path


    def training_setup(self, training_args, lr_mask, lr_features, weight_lr, lr_sigma, lr_triangles_init):

        l = [
            {'params': [self._features_dc], 'lr': lr_features, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr_features / 20.0, "name": "f_rest"},
            {'params': [self.vertices], 'lr': lr_triangles_init, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': weight_lr, "name": "vertex_weight"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.triangle_scheduler_args = get_expon_lr_func(lr_init=lr_triangles_init,
                                                        lr_final=lr_triangles_init/100,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)




    def training_setup(self, training_args, lr_features, weight_lr, lr_triangles_init):

        l = [
            {'params': [self._features_dc], 'lr': lr_features, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr_features / 20.0, "name": "f_rest"},
            {'params': [self.vertices], 'lr': lr_triangles_init, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': weight_lr, "name": "vertex_weight"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.triangle_scheduler_args = get_expon_lr_func(lr_init=lr_triangles_init,
                                                        lr_final=lr_triangles_init/100,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)

    def set_sigma(self, sigma):
        self._sigma = self.inverse_exponential_activation(sigma)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "vertices":
                    if iteration < 1000:
                        lr = 0
                    else:
                        lr = self.triangle_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "face_tex":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # ---------------- per-face texture ----------------
    @staticmethod
    def tex_lattice(n, device="cuda"):
        """[K,2] lattice coordinates (i, j) of the K samples (weights i/n, j/n of vertices 1, 2)."""
        ij = [(i, j) for i in range(n + 1) for j in range(n + 1 - i)]
        return torch.tensor(ij, device=device, dtype=torch.float32)

    @staticmethod
    def tex_lookup(tex, w1, w2, n):
        """Texture value of faces tex [N,K,3] at points with vertex-1/2 weights w1, w2 [N,P] (same as CUDA)."""
        u = w1.clamp(0, 1) * n; v = w2.clamp(0, 1) * n
        i = torch.floor(u).clamp(max=n - 1); j = torch.floor(v).clamp(max=n - 1)
        over = (i + j) > (n - 1)
        fix_j = over & ((u - i) > (v - j))
        j = torch.where(fix_j, n - 1 - i, j); i = torch.where(over & ~fix_j, n - 1 - j, i)
        i = i.clamp(min=0); j = j.clamp(min=0)
        fu, fv = u - i, v - j
        idx = lambda a, b: (a * (n + 1) - (a * (a - 1)) / 2 + b).long()
        lower = (fu + fv) <= 1
        i0 = torch.where(lower, idx(i, j), idx(i + 1, j + 1)); w0 = torch.where(lower, 1 - fu - fv, fu + fv - 1)
        i1 = torch.where(lower, idx(i + 1, j), idx(i, j + 1)); wa = torch.where(lower, fu, 1 - fu)
        i2 = torch.where(lower, idx(i, j + 1), idx(i + 1, j)); wb = torch.where(lower, fv, 1 - fv)
        g = lambda k: torch.gather(tex, 1, k.unsqueeze(-1).expand(-1, -1, 3))
        return w0.unsqueeze(-1) * g(i0) + wa.unsqueeze(-1) * g(i1) + wb.unsqueeze(-1) * g(i2)

    def init_texture(self, tex_res, lr):
        """Add a zero per-face texture of level tex_res to the model and the optimizer."""
        K = (tex_res + 1) * (tex_res + 2) // 2
        self.tex_res = tex_res
        self._face_tex = nn.Parameter(torch.zeros((self._triangle_indices.shape[0], K, 3), device="cuda").requires_grad_(True))
        self.optimizer.add_param_group({"params": [self._face_tex], "lr": lr, "name": "face_tex"})

    def _set_face_tex(self, new_values, state_fn):
        """Replace the texture parameter (pruned/extended faces); state_fn maps an Adam moment the same way."""
        if self.tex_res <= 0:
            return
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if group["name"] != "face_tex":
                    continue
                stored = self.optimizer.state.get(group["params"][0], None)
                param = nn.Parameter(new_values.requires_grad_(True))
                if stored is not None:
                    stored["exp_avg"] = state_fn(stored["exp_avg"])
                    stored["exp_avg_sq"] = state_fn(stored["exp_avg_sq"])
                    del self.optimizer.state[group["params"][0]]
                    self.optimizer.state[param] = stored
                group["params"][0] = param
                self._face_tex = param
                return
        self._face_tex = nn.Parameter(new_values.requires_grad_(True))

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] not in tensors_dict:
                continue  # e.g. the per-face texture, kept in sync separately
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
    

        return optimizable_tensors
    

    # Refinement accumulators (per triangle: inradius stats, per vertex: color-gradient sums) follow every
    # prune/append, so free-group pruning and densification do not reset them (--free_triangles with --mesh_refine).
    _TRI_ACCUM = ("_max_inradius", "_inr_views", "_inr_views_ok")
    _VERT_ACCUM = ("_color_grad", "_color_grad_count")

    def _select_accum(self, names, sel, old_n):
        for name in names:
            t = getattr(self, name, None)
            if t is not None and t.shape[0] == old_n:
                setattr(self, name, t[sel])

    def _grow_accum(self, names, old_n, extra):
        for name in names:
            t = getattr(self, name, None)
            if t is not None and t.shape[0] == old_n:
                setattr(self, name, torch.cat([t, torch.zeros(extra, dtype=t.dtype, device=t.device)]))

    def densification_postfix(self, new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles, new_anchor=None, new_anchor_normal=None, new_face_tex=None):
        # Create dictionary of new tensors to append
        d = {
            "vertices": new_vertices,
            "vertex_weight": new_vertex_weight,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
        }
        
        self._grow_accum(self._VERT_ACCUM, self.vertices.shape[0], new_vertices.shape[0])
        self._grow_accum(self._TRI_ACCUM, self._triangle_indices.shape[0], new_triangles.shape[0])

        # Append new tensors to optimizer
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        # Update model parameters
        self.vertices = optimizable_tensors["vertices"]
        self.vertex_weight = optimizable_tensors["vertex_weight"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        
        if self._anchor is not None:
            self._anchor = torch.cat([self._anchor, new_anchor], dim=0)
            self._anchor_normal = torch.cat([self._anchor_normal, new_anchor_normal], dim=0)
        if self._vertex_is_mesh is not None:
            self._vertex_is_mesh = torch.cat([self._vertex_is_mesh, self._new_vertex_is_mesh], dim=0)

        if self.tex_res > 0:
            if new_face_tex is None:
                new_face_tex = torch.zeros((new_triangles.shape[0],) + tuple(self._face_tex.shape[1:]), device="cuda")
            extra = new_face_tex.shape[0]
            self._set_face_tex(torch.cat([self._face_tex.detach(), new_face_tex], 0),
                               lambda t: torch.cat([t, torch.zeros((extra,) + tuple(t.shape[1:]), device=t.device)], 0))

        # Update triangle indices
        self._triangle_indices = torch.cat([
            self._triangle_indices, 
            new_triangles
        ], dim=0)

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")



    def _update_params_fast(self, selected_indices, iteration):
        selected_indices = torch.unique(selected_indices)
        selected_triangles_indices = self._triangle_indices[selected_indices]  # [S, 3]
        S = selected_triangles_indices.shape[0]
        
        edges = torch.cat([
            selected_triangles_indices[:, [0, 1]],
            selected_triangles_indices[:, [0, 2]],
            selected_triangles_indices[:, [1, 2]]
        ], dim=0) 
        edges_sorted, _ = torch.sort(edges, dim=1)
        
        unique_edges_tensor, unique_indices = torch.unique(
            edges_sorted, return_inverse=True, dim=0
        )  
        M = unique_edges_tensor.shape[0]
        
        v0 = self.vertices[unique_edges_tensor[:, 0]]
        v1 = self.vertices[unique_edges_tensor[:, 1]]
        new_vertices = (v0 + v1) / 2.0
        
        new_vertex_base = self.vertices.shape[0]
        
        unique_edges_cpu = unique_edges_tensor.cpu()
        edge_to_midpoint = {}
        for i in range(M):
            edge_tuple = (unique_edges_cpu[i, 0].item(), unique_edges_cpu[i, 1].item())
            edge_to_midpoint[edge_tuple] = new_vertex_base + i

        new_triangles_list = []
        selected_triangles_cpu = selected_triangles_indices.cpu()
        
        for i in range(S):
            tri = selected_triangles_cpu[i]
            a, b, c = tri[0].item(), tri[1].item(), tri[2].item()
            
            ab = (min(a, b), max(a, b))
            ac = (min(a, c), max(a, c))
            bc = (min(b, c), max(b, c))
            
            m_ab = edge_to_midpoint[ab]
            m_ac = edge_to_midpoint[ac]
            m_bc = edge_to_midpoint[bc]

            new_triangles_list.append([a, m_ab, m_ac])
            new_triangles_list.append([b, m_bc, m_ab])  # keep the parent's winding
            new_triangles_list.append([c, m_ac, m_bc])
            new_triangles_list.append([m_ab, m_bc, m_ac])
        
        subdivided_triangles = torch.tensor(
            new_triangles_list, 
            dtype=torch.int32, 
            device=self._triangle_indices.device
        )

        u, v = unique_edges_tensor[:, 0], unique_edges_tensor[:, 1]
        new_features_dc = (self._features_dc[u] + self._features_dc[v]) / 2.0
        new_features_rest = (self._features_rest[u] + self._features_rest[v]) / 2.0
        
        weights = self.get_vertex_weight
        avg_opacity = (weights[u] + weights[v]) / 2.0
        floor_new = self.vertex_floor(u)  # midpoints join their parents' group
        new_vertex_weight = self.inverse_opacity(avg_opacity, floor_new)

        new_triangles = subdivided_triangles

        if self._vertex_is_mesh is not None:
            self._new_vertex_is_mesh = self._vertex_is_mesh[u]
        new_anchor = new_anchor_normal = None
        if self._anchor is not None:
            new_anchor = (self._anchor[u] + self._anchor[v]) / 2.0
            n = self._anchor_normal[u] + self._anchor_normal[v]
            new_anchor_normal = torch.where(n.norm(dim=1, keepdim=True) > 1e-6, n, self._anchor_normal[u])
            new_anchor_normal = torch.nn.functional.normalize(new_anchor_normal, dim=1)

        return (
            new_vertices,
            new_vertex_weight,
            new_features_dc,
            new_features_rest,
            new_triangles,
            new_anchor,
            new_anchor_normal
        )


    def _prune_vertex_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["vertices", "vertex_weight", "f_dc", "f_rest"]:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    # Prune optimizer state
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    
                    del self.optimizer.state[group['params'][0]]
                    # Update parameter
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group['params'][0]
        
        if self._anchor is not None:
            self._anchor = self._anchor[mask]
            self._anchor_normal = self._anchor_normal[mask]
        if self._vertex_is_mesh is not None:
            self._vertex_is_mesh = self._vertex_is_mesh[mask]

        # Update model parameters
        for name, tensor in optimizable_tensors.items():
            if name == "vertices":
                self.vertices = tensor
            elif name == "vertex_weight":
                self.vertex_weight = tensor
            elif name == "f_dc":
                self._features_dc = tensor
            elif name == "f_rest":
                self._features_rest = tensor


    def _prune_vertices(self, vertex_mask: torch.Tensor):
        device = vertex_mask.device
        oldV = vertex_mask.numel()

        # Create mapping from old vertex IDs to new IDs (-1 for removed vertices)
        new_id = torch.full((oldV,), -1, dtype=torch.long, device=device)
        kept = torch.nonzero(vertex_mask, as_tuple=True)[0]
        new_id[kept] = torch.arange(kept.numel(), device=device, dtype=torch.long)

        # Remap triangle indices and drop triangles with removed vertices
        if self._triangle_indices.numel() > 0:
            remapped = new_id[self._triangle_indices.long()]
            valid_tris = (remapped >= 0).all(dim=1)
            remapped = remapped[valid_tris]
            self._select_accum(self._TRI_ACCUM, valid_tris, valid_tris.numel())
            if self.tex_res > 0 and not valid_tris.all():
                self._set_face_tex(self._face_tex.detach()[valid_tris], lambda t: t[valid_tris])
            self._triangle_indices = remapped.to(torch.int32).contiguous()

            if isinstance(self.image_size, torch.Tensor) and self.image_size.numel() > 0:
                self.image_size = self.image_size[valid_tris]
            if isinstance(self.importance_score, torch.Tensor) and self.importance_score.numel() > 0:
                self.importance_score = self.importance_score[valid_tris]

        # Prune vertex-related parameters using the initial mask
        self._select_accum(self._VERT_ACCUM, vertex_mask, oldV)
        self._prune_vertex_optimizer(vertex_mask)

        # After initial pruning, check for unreferenced vertices
        current_vertex_count = self.vertices.shape[0]
        if current_vertex_count > 0:
            # Identify vertices still referenced by triangles
            if self._triangle_indices.numel() > 0:
                referenced_vertices = torch.unique(self._triangle_indices)
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device)
                mask_referenced[referenced_vertices] = True
            else:
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device)

            # Remove unreferenced vertices
            if not mask_referenced.all():
                # Prune vertex parameters
                self._select_accum(self._VERT_ACCUM, mask_referenced, current_vertex_count)
                self._prune_vertex_optimizer(mask_referenced)

                # Remap triangle indices if triangles exist
                if self._triangle_indices.numel() > 0:
                    new_id2 = torch.full((current_vertex_count,), -1, dtype=torch.long, device=device)
                    kept2 = torch.nonzero(mask_referenced, as_tuple=True)[0]
                    new_id2[kept2] = torch.arange(kept2.numel(), device=device, dtype=torch.long)
                    self._triangle_indices = new_id2[self._triangle_indices.long()].to(torch.int32).contiguous()





    def prune_triangles(self, mask):

        ##################################################################
        # REMOVE ALL TRIANLGES WITH MIN WEIGHT LESS THAN SOME THRESHOLD  #
        ##################################################################
        self._select_accum(self._TRI_ACCUM, mask, self._triangle_indices.shape[0])
        self._triangle_indices = self._triangle_indices[mask]
        if self.tex_res > 0:
            self._set_face_tex(self._face_tex.detach()[mask], lambda t: t[mask])

        #################################################
        # WE SET THE CLASS VARIABLES TO THEIR NEW VALUES#
        #################################################
        self._triangle_indices = self._triangle_indices.to(torch.int32)

        self.image_size = self.image_size[mask]
        self.importance_score = self.importance_score[mask]
        

    def _sample_alives(self, probs, num, alive_indices=None):
        torch.manual_seed(1)  # always same "random" indices
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=False)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        return sampled_idxs        

    def add_new_gs(self, iteration, cap_max, splitt_large_triangles, probs_opacity=True):

        current_num_points = self.vertices.shape[0]
        target_num = min(cap_max, int(self.add_percentage * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        # Find indexes based on proba
        triangle_transp = self.importance_score
        probs = triangle_transp.squeeze()

        areas = self.triangle_areas().squeeze()
        probs = torch.where(areas < self.size_probs_zero, torch.zeros_like(probs), probs)
        probs = torch.where(self.image_size < self.size_probs_zero_image_space, torch.zeros_like(probs), probs) # dont splitt if smaller than 10

        if self.no_mesh_densify:
            # only free triangles may be split (without --free_triangles: none); grow the free
            # group by add_percentage of its own size, as the whole model grows without groups
            splittable = ~self.triangle_is_mesh()
            if not splittable.any():
                return 0
            probs = torch.where(splittable, probs, torch.zeros_like(probs))
            areas = torch.where(splittable, areas, torch.full_like(areas, -1.0))
            num_free = int((~self._vertex_is_mesh).sum().item())
            num_gs = min(int(self.add_percentage * num_free) - num_free, cap_max - current_num_points)
            num_gs = max(0, min(num_gs, int((probs > 0).sum().item())))

        rand_idx = self._sample_alives(probs=probs, num=num_gs) if num_gs > 0 else torch.empty(0, dtype=torch.long, device=probs.device)

        # Split the largest triangles
        split_large = splitt_large_triangles
        k = min(split_large, int((areas >= 0).sum().item()))
        _, top_idx = torch.topk(areas, k, largest=True, sorted=False)

        # 3) combine and deduplicate
        add_idx = torch.unique(torch.cat([rand_idx, top_idx.to(rand_idx.device)]), sorted=False)
        if self.free_max_triangles > 0 and self.no_mesh_densify:
            # every split replaces one free triangle by four (net +3); stop at the cap
            room = (self.free_max_triangles - int((~self.triangle_is_mesh()).sum().item())) // 3
            if room <= 0:
                return 0
            add_idx = add_idx[:room]

        (new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles, new_anchor, new_anchor_normal) = self._update_params_fast(add_idx, iteration)

        self.densification_postfix(new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles, new_anchor, new_anchor_normal)

        mask = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool)
        mask[add_idx] = False
        self.prune_triangles(mask)



    def update_min_weight(self, new_min_weight: float, preserve_outputs: bool = True, new_free_min_weight: float = None):
        """Raise the floor (mesh group, or everyone without groups; free group: new_free_min_weight)
        while keeping every vertex's current opacity unless it is below its new floor."""
        new_m = float(max(0.0, min(new_min_weight, 1.0 - 1e-4)))

        # 1) grab the current realized opacities y (under the old floor)
        with torch.no_grad():
            y = self.get_vertex_weight.detach()
        self.opacity_floor = new_m
        if self._vertex_is_mesh is not None and new_free_min_weight is not None:
            self.free_opacity_floor = float(max(0.0, min(new_free_min_weight, 1.0 - 1e-4)))
        with torch.no_grad():
            self.vertex_weight.data.copy_(self.inverse_opacity(y, self.vertex_floor()))


    # Child triangles for a triangle (v0,v1,v2) with edges e0=(v0,v1), e1=(v1,v2), e2=(v2,v0) and edge
    # midpoints m0,m1,m2, per pattern s0 + 2*s1 + 4*s2 of split edges; slots 0..5 = v0,v1,v2,m0,m1,m2.
    # Every child keeps the parent's winding.
    _REFINE_TEMPLATES = {
        1: [(0, 3, 2), (3, 1, 2)],
        2: [(0, 1, 4), (0, 4, 2)],
        4: [(0, 1, 5), (5, 1, 2)],
        3: [(3, 1, 4), (0, 3, 4), (0, 4, 2)],
        6: [(5, 4, 2), (0, 1, 4), (0, 4, 5)],
        5: [(0, 3, 5), (3, 1, 2), (3, 2, 5)],
        7: [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)],
        # Catmull-Clark on a triangle: 3 quads (v_i, m, c, m') with the face point c = slot 6, each cut along v_i-c
        8: [(0, 3, 6), (0, 6, 5), (1, 4, 6), (1, 6, 3), (2, 5, 6), (2, 6, 4)],
    }

    def accumulate_color_grad(self):
        """Add this step's per-vertex color-gradient norm (the refinement score) to a running sum."""
        g = self._features_dc.grad
        if g is None:
            return
        n = g.shape[0]
        if getattr(self, "_color_grad", None) is None or self._color_grad.shape[0] != n:
            self._color_grad = torch.zeros(n, device=g.device)
            self._color_grad_count = torch.zeros(n, device=g.device)
        norm = g.detach().reshape(n, -1).norm(dim=1)
        self._color_grad += norm
        self._color_grad_count += (norm > 0).float()

    @torch.no_grad()
    def accumulate_vertex_score(self, idx, values):
        """Add per-vertex values (e.g. the multi-view mismatch near each vertex, --refine_by_mv) to the same running
        sums as accumulate_color_grad, so refine_mesh and the Voronoi seed rounds use them as the error."""
        n = self.vertices.shape[0]
        if getattr(self, "_color_grad", None) is None or self._color_grad.shape[0] != n:
            self._color_grad = torch.zeros(n, device="cuda")
            self._color_grad_count = torch.zeros(n, device="cuda")
        keep = idx >= 0
        self._color_grad.index_add_(0, idx[keep], values[keep].float())
        self._color_grad_count.index_add_(0, idx[keep], torch.ones_like(values[keep], dtype=torch.float))

    @torch.no_grad()
    def accumulate_inradius(self, camera, radii, render_scaling, min_inradius=0.0):
        """Keep, per triangle, the largest on-screen inradius (in render pixels) over the views it was drawn in.
        The rasterizer skips triangles whose inradius is below 1 render pixel, so refinement uses this to only
        split triangles whose children will still be drawn."""
        T = self._triangle_indices.shape[0]
        if getattr(self, "_max_inradius", None) is None or self._max_inradius.shape[0] != T:
            self._max_inradius = torch.zeros(T, device="cuda")
            self._inr_views = torch.zeros(T, device="cuda")  # views the triangle was drawn in
            self._inr_views_ok = torch.zeros(T, device="cuda")  # ... with inradius >= min_inradius
        vis = torch.nonzero(radii > 0, as_tuple=True)[0]
        if vis.numel() == 0:
            return
        F = self._triangle_indices[vis].long()
        V = self.vertices.detach()
        p = torch.cat([V, torch.ones_like(V[:, :1])], 1) @ camera.full_proj_transform  # row-vector convention
        w = p[:, 3:4].clamp(min=1e-6)
        W, H = camera.image_width * render_scaling, camera.image_height * render_scaling
        xy = torch.stack([((p[:, 0:1] / w) + 1) * W * 0.5, ((p[:, 1:2] / w) + 1) * H * 0.5], -1).squeeze(1)
        t = xy[F]  # [n,3,2]
        e = torch.stack([(t[:, 1] - t[:, 0]).norm(dim=1), (t[:, 2] - t[:, 1]).norm(dim=1), (t[:, 0] - t[:, 2]).norm(dim=1)], 1)
        d1, d2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        area = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]).abs()
        inr = 2 * area / e.sum(1).clamp(min=1e-9)
        self._max_inradius[vis] = torch.maximum(self._max_inradius[vis], inr)
        self._inr_views[vis] += 1
        self._inr_views_ok[vis] += (inr >= min_inradius).float()

    @torch.no_grad()
    def refine_mesh(self, fraction, min_edge, max_vertices, min_inradius=0.0, view_fraction=0.0, catmull_clark=False, move_old=False, geo_angle=0.0):
        """Conforming refinement of the mesh group: pick the `fraction` of mesh triangles with the highest
        mean color gradient of their vertices (longest edge above min_edge), split their longest edge at its
        midpoint and re-triangulate every triangle touching a split edge (neighbours included), so no
        T-junctions form. With catmull_clark, a selected triangle instead gets all three edges split plus a face
        point (centroid), i.e. one Catmull-Clark step on a triangle: 3 quads, each cut into 2 triangles (6 children).
        New edge points use the CC rule (v1 + v2 + c1 + c2) / 4 with c the centroids of the two adjacent
        triangles (midpoint on boundary / non-manifold edges); old vertices stay fixed (interpolating).
        With move_old (full Catmull-Clark smoothing) the corners of selected triangles also move to
        (Q + 2R + (n-3) P) / n (Q: mean adjacent centroid, R: mean adjacent edge midpoint, n: valence);
        boundary and non-manifold vertices stay fixed. Neighbours of selected triangles are re-triangulated
        as above. Returns the number of new vertices."""
        if getattr(self, "_color_grad", None) is None or self._color_grad.shape[0] != self.vertices.shape[0]:
            return 0
        F = self._triangle_indices.long()
        V = self.vertices.detach()
        T = F.shape[0]
        is_mesh_tri = self.triangle_is_mesh()
        num_mesh_vertices = V.shape[0] if self._vertex_is_mesh is None else int(self._vertex_is_mesh.sum().item())
        budget = max_vertices - num_mesh_vertices
        if budget <= 0:
            return 0

        # unique edges; each triangle's three edge ids
        ends = torch.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1)  # [T,3,2]
        edges, edge_id = torch.unique(torch.sort(ends.reshape(-1, 2), dim=1).values, dim=0, return_inverse=True)
        edge_id = edge_id.reshape(T, 3)

        # score: mean over the triangle's vertices of the average color-gradient norm per step seen
        vertex_score = self._color_grad / self._color_grad_count.clamp(min=1)
        score = vertex_score[F].mean(1)
        tri = V[F]
        edge_len = torch.stack([(tri[:, 1] - tri[:, 0]).norm(dim=1), (tri[:, 2] - tri[:, 1]).norm(dim=1), (tri[:, 0] - tri[:, 2]).norm(dim=1)], 1)
        longest = edge_len.argmax(1)
        eligible = is_mesh_tri & (edge_len.max(1).values > min_edge) & (score > 0)
        if geo_angle > 0:
            n = torch.nn.functional.normalize(torch.nan_to_num(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)), dim=1)
            tid = torch.arange(T, device=V.device).repeat_interleave(3)
            eid = edge_id.reshape(-1)
            first = torch.full((edges.shape[0],), T, dtype=torch.long, device=V.device).scatter_reduce(0, eid, tid, "amin", include_self=True)
            last = torch.full((edges.shape[0],), -1, dtype=torch.long, device=V.device).scatter_reduce(0, eid, tid, "amax", include_self=True)
            two = torch.zeros(edges.shape[0], device=V.device).index_add_(0, eid, torch.ones_like(eid, dtype=torch.float)) == 2
            cos = (n[first.clamp(max=T - 1)] * n[last.clamp(min=0)]).sum(1).abs().clamp(max=1.0)
            edge_angle = torch.where(two, torch.rad2deg(torch.acos(cos)), torch.zeros_like(cos))
            eligible &= edge_angle[edge_id].max(1).values >= geo_angle
        if min_inradius > 0:
            # only triangles big enough on screen that their children (about half the inradius) are still drawn
            inr = getattr(self, "_max_inradius", None)
            if inr is None or inr.shape[0] != T:
                return 0
            if view_fraction > 0:
                # big enough in at least view_fraction of the views it was drawn in, not only its best view
                eligible &= self._inr_views_ok >= view_fraction * self._inr_views.clamp(min=1)
                eligible &= self._inr_views > 0
            else:
                eligible &= inr >= min_inradius
        k = min(int(fraction * int(is_mesh_tri.sum().item())), int(eligible.sum().item()), budget // 4 if catmull_clark else budget)
        if k <= 0:
            return 0
        selected = torch.topk(torch.where(eligible, score, torch.full_like(score, -1.0)), k).indices

        split = torch.zeros(edges.shape[0], dtype=torch.bool, device=V.device)
        if catmull_clark:
            split[edge_id[selected].reshape(-1)] = True
        else:
            split[edge_id[selected, longest[selected]]] = True
        split_ids = torch.nonzero(split, as_tuple=True)[0]
        midpoint = torch.full((edges.shape[0],), -1, dtype=torch.long, device=V.device)
        midpoint[split_ids] = V.shape[0] + torch.arange(split_ids.numel(), device=V.device)

        # new vertices: averages of the edge's endpoints (Catmull-Clark: pulled towards the adjacent centroids)
        u, w = edges[split_ids, 0], edges[split_ids, 1]
        new_vertices = (V[u] + V[w]) / 2.0
        if catmull_clark:
            centroid = V[F].mean(1)  # [T,3] face points of all triangles
            csum = torch.zeros(edges.shape[0], 3, device=V.device).index_add_(0, edge_id.reshape(-1), centroid.repeat_interleave(3, dim=0))
            ccount = torch.zeros(edges.shape[0], device=V.device).index_add_(0, edge_id.reshape(-1), torch.ones(T * 3, device=V.device))
            interior = (ccount[split_ids] == 2)[:, None]
            new_vertices = torch.where(interior, (V[u] + V[w] + csum[split_ids]) / 4.0, new_vertices)
        moved_ids = moved_pos = None
        if catmull_clark and move_old:
            nV = V.shape[0]
            ev = edges.reshape(-1)
            valence = torch.zeros(nV, device=V.device).index_add_(0, ev, torch.ones(ev.numel(), device=V.device))
            rsum = torch.zeros(nV, 3, device=V.device).index_add_(0, ev, ((V[edges[:, 0]] + V[edges[:, 1]]) / 2.0).repeat_interleave(2, dim=0))
            nonmanifold = torch.zeros(nV, device=V.device).index_add_(0, ev, (ccount != 2).float().repeat_interleave(2)) > 0
            qsum = torch.zeros(nV, 3, device=V.device).index_add_(0, F.reshape(-1), centroid.repeat_interleave(3, dim=0))
            fcount = torch.zeros(nV, device=V.device).index_add_(0, F.reshape(-1), torch.ones(T * 3, device=V.device))
            n = valence.clamp(min=1)[:, None]
            cc_pos = (qsum / fcount.clamp(min=1)[:, None] + 2.0 * rsum / n + (n - 3.0) * V) / n
            corner = torch.zeros(nV, dtype=torch.bool, device=V.device)
            corner[F[selected].reshape(-1)] = True
            corner &= ~nonmanifold & (valence >= 3)
            if self._vertex_is_mesh is not None:
                corner &= self._vertex_is_mesh
            moved_ids = torch.nonzero(corner, as_tuple=True)[0]
            moved_pos = cc_pos[moved_ids]
        new_features_dc = (self._features_dc[u] + self._features_dc[w]) / 2.0
        new_features_rest = (self._features_rest[u] + self._features_rest[w]) / 2.0
        weights = self.get_vertex_weight
        new_weight = (weights[u] + weights[w]) / 2.0
        new_anchor = new_anchor_normal = None
        if self._anchor is not None:
            new_anchor = (self._anchor[u] + self._anchor[w]) / 2.0
            n = self._anchor_normal[u] + self._anchor_normal[w]
            new_anchor_normal = torch.nn.functional.normalize(torch.where(n.norm(dim=1, keepdim=True) > 1e-6, n, self._anchor_normal[u]), dim=1)
        floor_idx = u
        face_point = torch.full((T,), -1, dtype=torch.long, device=V.device)
        if catmull_clark:
            # one face point per selected triangle: centroid, attributes averaged over its three vertices
            Fs = F[selected]
            face_point[selected] = V.shape[0] + split_ids.numel() + torch.arange(k, device=V.device)
            new_vertices = torch.cat([new_vertices, V[Fs].mean(1)], 0)
            new_features_dc = torch.cat([new_features_dc, self._features_dc[Fs].mean(1)], 0)
            new_features_rest = torch.cat([new_features_rest, self._features_rest[Fs].mean(1)], 0)
            new_weight = torch.cat([new_weight, weights[Fs].mean(1)], 0)
            floor_idx = torch.cat([u, Fs[:, 0]], 0)
            if self._anchor is not None:
                new_anchor = torch.cat([new_anchor, self._anchor[Fs].mean(1)], 0)
                n = self._anchor_normal[Fs].sum(1)
                n = torch.nn.functional.normalize(torch.where(n.norm(dim=1, keepdim=True) > 1e-6, n, self._anchor_normal[Fs[:, 0]]), dim=1)
                new_anchor_normal = torch.cat([new_anchor_normal, n], 0)
        new_vertex_weight = self.inverse_opacity(new_weight, self.vertex_floor(floor_idx))
        if self._vertex_is_mesh is not None:
            self._new_vertex_is_mesh = torch.ones(new_vertices.shape[0], dtype=torch.bool, device=V.device)

        # children of every triangle touching a split edge
        s = split[edge_id]  # [T,3]
        pattern = s[:, 0].long() + 2 * s[:, 1].long() + 4 * s[:, 2].long()
        if catmull_clark:
            pattern[selected] = 8
        affected = torch.nonzero(pattern > 0, as_tuple=True)[0]
        slots = torch.cat([F, midpoint[edge_id], face_point[:, None]], 1)  # [T,7]: v0,v1,v2,m0,m1,m2,c
        # barycentric position of each slot in the parent triangle (weights of vertices 0, 1, 2)
        slot_bary = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5], [1 / 3, 1 / 3, 1 / 3]], device=V.device)
        children, child_tex = [], []
        if self.tex_res > 0:
            lat = self.tex_lattice(self.tex_res) / self.tex_res  # [K,2] child weights of its vertices 1, 2
        for code, template in self._REFINE_TEMPLATES.items():
            idx = torch.nonzero(pattern == code, as_tuple=True)[0]
            if idx.numel():
                t = torch.tensor(template, device=V.device)
                children.append(slots[idx][:, t].reshape(-1, 3))
                if self.tex_res > 0:
                    for c in range(t.shape[0]):
                        # parent barycentrics of the child's lattice samples, then the parent's texture there
                        cb = slot_bary[t[c]]  # [3 child vertices, 3 parent weights]
                        wc = torch.stack([1 - lat.sum(1), lat[:, 0], lat[:, 1]], 1)  # [K,3] child weights
                        pb = wc @ cb  # [K,3] parent weights
                        tex = self.tex_lookup(self._face_tex.detach()[idx], pb[None, :, 1].expand(idx.numel(), -1),
                                              pb[None, :, 2].expand(idx.numel(), -1), self.tex_res)
                        child_tex.append((c, tex))
                    # keep the same order as children: per parent, its template's children in sequence
                    k = len(t)
                    block = torch.stack([ct for _, ct in child_tex[-k:]], 1).reshape(-1, child_tex[-1][1].shape[1], 3)
                    child_tex = child_tex[:-k] + [(-1, block)]
        children = torch.cat(children, 0).to(torch.int32)
        new_face_tex = torch.cat([b for _, b in child_tex], 0) if self.tex_res > 0 else None

        self.densification_postfix(new_vertices, new_vertex_weight, new_features_dc, new_features_rest, children, new_anchor, new_anchor_normal, new_face_tex)
        if moved_ids is not None:
            self.vertices.data[moved_ids] = moved_pos
        keep = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool, device=V.device)
        keep[affected] = False  # the parents (children were appended after them)
        self.prune_triangles(keep)
        self._color_grad = None
        self._max_inradius = None
        return int(new_vertices.shape[0])

    def triangle_areas(self):
        tri = self.vertices[self._triangle_indices]                    # [T, 3, 3]
        AB  = tri[:, 1] - tri[:, 0]                                    # [T, 3]
        AC  = tri[:, 2] - tri[:, 0]                                    # [T, 3]
        cross_prod = torch.cross(AB, AC, dim=1)                        # [T, 3]
        areas = 0.5 * torch.linalg.norm(cross_prod, dim=1)             # [T]
        areas = torch.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)
        return areas

