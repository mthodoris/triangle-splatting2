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
import math
from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
try:  # per-face texture rasterizer (submodules/diff-triangle2-rasterization-tex), only needed with --tex_res
    from diff_triangle_rasterization_tex import TriangleRasterizationSettings as TexRasterizationSettings
    from diff_triangle_rasterization_tex import TriangleRasterizer as TexRasterizer
except ImportError:
    TexRasterizer = None
from scene.triangle_model import TriangleModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
import torch.nn.functional as F

def render(viewpoint_camera, pc : TriangleModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    
    triangles_indices = pc.get_triangle_indices  # the idx of the 3 vertices of each triangl
    vertices = pc.get_vertices # contains all the vertices of the triangles
    vertex_weights = pc.get_vertex_weight  # contains the weights of the vertices for each vertex in the triangles
    scaling = torch.zeros_like(triangles_indices[:, 0], dtype=pc.get_triangles_points.dtype, requires_grad=True, device="cuda").detach()



    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    H_init = int(viewpoint_camera.image_height)
    W_init = int(viewpoint_camera.image_width)

    upsample = pc.scaling

    H = upsample * H_init
    W = upsample * W_init

    use_tex = getattr(pc, "tex_res", 0) > 0
    if use_tex and TexRasterizer is None:
        raise ImportError("this model has a per-face texture: install submodules/diff-triangle2-rasterization-tex")
    raster_settings = (TexRasterizationSettings if use_tex else TriangleRasterizationSettings)(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = (TexRasterizer if use_tex else TriangleRasterizer)(raster_settings=raster_settings)

    sigma = pc.get_sigma

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features

    else:
        colors_precomp = override_color

    # Rasterize visible triangles to image, obtain their radii (on screen). 
    def rasterize(face_mask=None, rasterizer=rasterizer, sigma=sigma):
        tri_idx = triangles_indices if face_mask is None else triangles_indices[face_mask]
        tex = {}
        if use_tex:
            tex = {"face_tex": pc._face_tex if face_mask is None else pc._face_tex[face_mask], "tex_res": pc.tex_res}
        return rasterizer(
            vertices=vertices,
            triangles_indices=tri_idx,
            vertex_weights=vertex_weights.squeeze(),
            sigma=sigma,
            shs = shs,
            colors_precomp = colors_precomp,
            scaling = scaling if face_mask is None else scaling[face_mask],
            **tex
           )

    mesh_image = None
    if getattr(pc, "free_behind", False) and pc._vertex_is_mesh is not None:
        # --free_behind: the free group is composited strictly behind the mesh group,
        # image = mesh (over black) + (1 - mesh alpha) * free (over bg), with the mesh alpha detached.
        # Free triangles get no gradient where the mesh is opaque, so they can only fill what the mesh leaves empty.
        is_mesh = pc.triangle_is_mesh()
        mesh_rasterizer = (TexRasterizer if use_tex else TriangleRasterizer)(
            raster_settings=raster_settings._replace(bg=torch.zeros_like(bg_color)))
        mesh_image, radii_m, scaling_m, allmap, max_blending_m = rasterize(is_mesh, mesh_rasterizer)
        free_sigma = sigma if pc.free_sigma is None else pc.free_sigma
        free_image, radii_f, scaling_f, _, max_blending_f = rasterize(~is_mesh, sigma=free_sigma)
        rendered_image = mesh_image + (1 - allmap[1:2].detach()) * free_image
        radii = torch.zeros(is_mesh.shape[0], dtype=radii_m.dtype, device=radii_m.device)
        radii[is_mesh], radii[~is_mesh] = radii_m, radii_f
        scaling = torch.zeros(is_mesh.shape[0], dtype=scaling_m.dtype, device=scaling_m.device)
        scaling[is_mesh], scaling[~is_mesh] = scaling_m, scaling_f
        max_blending = torch.zeros(is_mesh.shape[0], dtype=max_blending_m.dtype, device=max_blending_m.device)
        max_blending[is_mesh], max_blending[~is_mesh] = max_blending_m, max_blending_f
        mesh_image = F.interpolate(mesh_image.unsqueeze(0), size=(H_init, W_init), mode="area").squeeze(0)
    else:
        rendered_image, radii, scaling, allmap, max_blending = rasterize()

    img_hr = rendered_image.unsqueeze(0)  # -> [1, 3, H, W]
    img_ds_area = F.interpolate(img_hr, size=(H_init, W_init), mode="area")  # [1, 3, H0, W0]
    rendered_image = img_ds_area.squeeze(0)
    

    rets =  {"render": rendered_image,
            "visibility_filter" : radii > 0,
            "radii": radii, 
            "scaling": scaling,
            "max_blending": max_blending,
            }
    if mesh_image is not None:
        rets["mesh_render"] = mesh_image  # mesh group alone (black where it is empty), --free_behind only

    # additional regularizations
    render_alpha = allmap[1:2]
    img_hr = render_alpha.unsqueeze(0)  # -> [1, 3, H, W]
    img_ds_area = F.interpolate(img_hr, size=(H_init, W_init), mode="area")  # [1, 3, H0, W0]
    render_alpha = img_ds_area.squeeze(0)

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    img_hr = render_normal.unsqueeze(0)  # -> [1, 3, H, W]
    img_ds_area = F.interpolate(img_hr, size=(H_init, W_init), mode="area")  # [1, 3, H0, W0]
    render_normal = img_ds_area.squeeze(0)
    #render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    img_hr = render_depth_median.unsqueeze(0)  # -> [1, 3, H, W]
    img_ds_area = F.interpolate(img_hr, size=(H_init, W_init), mode="area")  # [1, 3, H0, W0]
    render_depth_median = img_ds_area.squeeze(0)
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    rendered_entropy = allmap[0:1]
    
    
    # gets the id per pixel of the triangle influencing it
    render_id = allmap[6:7]
    
    surf_depth = render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_ids': render_id,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            "entropy": rendered_entropy,
            "depth_full": allmap[5:6]
    })

    return rets





 