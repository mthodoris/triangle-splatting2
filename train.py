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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, mesh_topology, crease_weights, laplacian_loss, edge_aware_normal_consistency, edge_aware_smoothness, tangential_laplacian_loss
from triangle_renderer import render
import sys
from scene import Scene, TriangleModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, update_indoor
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import lpips
import torch.nn.functional as F
from utils.train_logger import TrainLogger


def training(
        dataset,   
        opt, 
        pipe,
        testing_iterations,
        checkpoint, 
        debug_from,
        logger,
        ):
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Load parameters, triangles and scene
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    init_num_vertices = triangles.vertices.shape[0]
    init_num_triangles = triangles._triangle_indices.shape[0]
    print("Initial model: {} vertices, {} triangles ({})".format(
        init_num_vertices, init_num_triangles, dataset.mesh_path if dataset.mesh_path else "point cloud"))
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    triangles.add_percentage = opt.add_percentage


    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()

    ema_loss_for_log = 0.0
    # tqdm goes straight to the terminal so its carriage-return updates do not flood logs/stdout.txt
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", file=sys.__stderr__)
    first_iter += 1

    initial_sigma = opt.set_sigma
    final_sigma = 0.0001
    sigma_start = opt.sigma_start
    total_iters = opt.sigma_until

    init_opacity = 0.1
    final_opacity = opt.final_opacity
    total_iters_opacity = opt.final_opacity_iter

    lambda_weight = opt.lambda_weight

    prune_triangles = opt.prune_triangles_threshold

    prune_size = opt.prune_size
    start_upsampling = opt.start_upsampling
    splitt_large_triangles = opt.splitt_large_triangles
    triangles.size_probs_zero = opt.size_probs_zero
    triangles.size_probs_zero_image_space = opt.size_probs_zero_image_space

    use_mesh_reg = opt.lambda_laplacian > 0 or opt.lambda_normal_consistency > 0 or opt.lambda_tangential_laplacian > 0
    if (opt.normal_only_motion or opt.lambda_tangential_laplacian > 0) and triangles._anchor is None:
        raise ValueError("--normal_only_motion and --lambda_tangential_laplacian need --mesh_path")
    topology = None # edge connectivity, rebuilt after every prune/densify step

    for iteration in range(first_iter, opt.iterations + 1):


        if iteration == start_upsampling:
            triangles.scaling = opt.upscaling_factor
        if iteration == 25000:
            triangles.scaling = 4


        iter_start.record()

        triangles.update_learning_rate(iteration)
        if iteration < sigma_start:
            current_sigma = initial_sigma
        else:
            progress = (iteration - sigma_start) / (total_iters - sigma_start)
            progress = min(progress, 1.0)
            current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress

        triangles.set_sigma(current_sigma)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if not viewpoint_stack or len(scene.getTrainCameras()) + iteration == opt.iterations:
            viewpoint_stack = scene.getTrainCameras().copy()
            if len(scene.getTrainCameras()) + iteration == opt.iterations:
                print(iteration)
                triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device="cuda") # reset to 0 to ensure that everything is deleted with an importance score of 0
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_normal = viewpoint_cam.normal_map.cuda()
        seg_hr = gt_normal.unsqueeze(0)  # -> [1, 3, H, W]
        seg_ds_area = F.interpolate(seg_hr, size=(gt_image.shape[1], gt_image.shape[2]), mode="area")  # [1, 3, H0, W0]
        gt_normal = seg_ds_area.squeeze(0)  # -> [3, H0, W0]

        pixel_loss = l1_loss(image, gt_image)

        image_size = render_pkg["scaling"].detach()
        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]

        importance_score = render_pkg["max_blending"].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]

        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # loss normal and distortion
        rend_normal  = render_pkg['rend_normal']
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0 # 0.001

        normal_error = (1 - (rend_normal * gt_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        
        if iteration < opt.start_opacity_floor:
            loss_weight = triangles.get_vertex_weight[triangles._triangle_indices].mean() * lambda_weight
        else:
            loss_weight = 0

        # mesh regularization: Laplacian smoothing and edge-aware normal smoothness
        reg_loss = 0
        if iteration > opt.iteration_mesh:
            if use_mesh_reg and triangles._triangle_indices.numel() > 0:
                if topology is None:
                    reg_faces = triangles._triangle_indices[triangles.triangle_is_mesh()]  # mesh group only
                    topology = mesh_topology(reg_faces)
                if opt.lambda_laplacian > 0:
                    edge_w = crease_weights(triangles.vertices, reg_faces, topology, opt.crease_sigma)
                    reg_loss = reg_loss + opt.lambda_laplacian * laplacian_loss(triangles.vertices, topology[0], edge_w, topology[4])
                if opt.lambda_tangential_laplacian > 0:
                    reg_loss = reg_loss + opt.lambda_tangential_laplacian * tangential_laplacian_loss(
                        triangles.vertices, topology[0], triangles._anchor_normal, topology[4])
                if opt.lambda_normal_consistency > 0:
                    reg_loss = reg_loss + opt.lambda_normal_consistency * edge_aware_normal_consistency(
                        triangles.vertices, reg_faces, topology, opt.crease_sigma)
            if opt.lambda_edge_smooth > 0:
                reg_loss = reg_loss + opt.lambda_edge_smooth * edge_aware_smoothness(rend_normal, gt_image, opt.edge_alpha)

        # FINAL LOSS
        loss = loss_image + loss_weight + normal_loss + reg_loss  # + depth_loss

        loss.backward()
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            
            logger.log_metrics(iteration, loss, ema_loss_for_log, pixel_loss, normal_loss, loss_weight, current_sigma, triangles)
            training_report(tb_writer, logger, iteration, pixel_loss, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            # Handle pruning operations
            if iteration % 500 == 0:
                topology = None # pruning/densification below changes the mesh connectivity

                print(torch.min(triangles.importance_score))

                # --- Build condition masks (all mean "DELETE") ---
                triangle_vertex_weights = triangles.get_vertex_weight[triangles._triangle_indices]  # [T,3,1]
                min_weights = triangle_vertex_weights.min(dim=1).values  # [T]

                mask_opacity     = (min_weights <= prune_triangles).squeeze()              # delete if too low
                mask_importance  = (triangles.importance_score <= prune_triangles).squeeze()  # delete if too low
                mask_size        = (triangles.image_size > prune_size).squeeze()                 # delete if too big

                delete_mask = mask_opacity | mask_importance | mask_size
                if opt.no_prune:
                    # protect the mesh group (everything, without --free_triangles)
                    delete_mask = delete_mask & ~triangles.triangle_is_mesh()
                keep_mask   = ~delete_mask  # prune_triangles keeps True

                # --- Apply pruning (keep=True) ---
                if iteration > opt.start_pruning and delete_mask.any():
                    triangles.prune_triangles(keep_mask)

                if not opt.no_prune or triangles._vertex_is_mesh is not None:
                    # We prune vertices that are no longer used
                    device = triangles.vertices.device
                    used_vertex_mask = torch.zeros(triangles.vertices.shape[0],
                                                dtype=torch.bool,
                                                device=device)
                    if triangles._triangle_indices.numel() > 0:
                        # Flatten indices and mark used vertices
                        flat_indices = triangles._triangle_indices.flatten()
                        used_vertex_mask[flat_indices] = True

                    # Combine conditions: keep vertices if used OR weight above threshold
                    weight_mask = (triangles.get_vertex_weight.squeeze() >= prune_triangles)
                    vertex_mask = weight_mask | used_vertex_mask
                    if opt.no_prune:
                        vertex_mask = vertex_mask | triangles._vertex_is_mesh

                    triangles._prune_vertices(vertex_mask)


                triangle_vertex_weights = triangles.get_vertex_weight[triangles._triangle_indices]  # [T,3,1]
                min_weights = triangle_vertex_weights.min(dim=1).values

                
                needs_densification = (iteration < opt.densify_until_iter and 
                                     iteration % opt.densification_interval == 0 and 
                                     iteration > opt.densify_from_iter)
                
                if needs_densification:
                    # We add new vertices

                    # switch between probability of opacity and 3d triangle size
                    if iteration < opt.start_opacity_floor:
                        probs_opacity = True
                    else:
                        if iteration % 1000 == 0:
                            probs_opacity = True
                        else:
                            probs_opacity = False

                    triangles.add_new_gs(iteration, cap_max=opt.max_points, splitt_large_triangles=splitt_large_triangles, probs_opacity=probs_opacity)
   

                if iteration > opt.start_opacity_floor:
                    start_iter = opt.start_opacity_floor
                    end_iter = total_iters_opacity  # the iteration where you want to reach final_opacity
                    a = min(1.0, max(0.0, (iteration - start_iter) / max(1, end_iter - start_iter)))
                    current_opacity = init_opacity + (final_opacity - init_opacity) * a
                    current_opacity = min(current_opacity, final_opacity)
                    current_free = min(init_opacity + (opt.free_final_opacity - init_opacity) * a, opt.free_final_opacity)
                    triangles.update_min_weight(current_opacity, new_free_min_weight=current_free)

                    prune_triangles += 0.01 

                    triangle_vertex_weights = triangles.get_vertex_weight[triangles._triangle_indices]



            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none = True)
                if opt.normal_only_motion:
                    triangles.project_to_anchor_normals(opt.max_normal_offset * triangles.init_edge)


    # cleaning of triangles that we do not need
    if not opt.no_prune:
        viewpoint_stack = scene.getTrainCameras().copy()
        triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        while viewpoint_stack:
            viewpoint_cam = viewpoint_stack.pop(0)
            render_pkg = render(viewpoint_cam, triangles, pipe, bg)

            importance_score = render_pkg["max_blending"].detach()
            mask = importance_score > triangles.importance_score
            triangles.importance_score[mask] = importance_score[mask]
        mask_importance  = (triangles.importance_score <= 0.5).squeeze()
        triangles.prune_triangles(~mask_importance) # delete all the remaining triangles that do not have an influence
    elif triangles._vertex_is_mesh is not None:
        # --no_prune with free triangles: the same cleanup, for the free group only
        viewpoint_stack = scene.getTrainCameras().copy()
        triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        while viewpoint_stack:
            viewpoint_cam = viewpoint_stack.pop(0)
            render_pkg = render(viewpoint_cam, triangles, pipe, bg)
            importance_score = render_pkg["max_blending"].detach()
            mask = importance_score > triangles.importance_score
            triangles.importance_score[mask] = importance_score[mask]
        mask_importance = (triangles.importance_score <= 0.5).squeeze() & ~triangles.triangle_is_mesh()
        triangles.prune_triangles(~mask_importance)

    scene.save(iteration)
    mesh_file = scene.save_mesh(iteration)
    print("Saved mesh to {}".format(mesh_file))
    logger.write_summary({
        "status": "finished",
        "iterations": iteration,
        "init_source": dataset.mesh_path if dataset.mesh_path else "point cloud",
        "init_num_vertices": init_num_vertices,
        "init_num_triangles": init_num_triangles,
        "final_num_vertices": triangles.vertices.shape[0],
        "final_num_triangles": triangles._triangle_indices.shape[0],
        "final_ema_loss": "{:.6f}".format(ema_loss_for_log),
        "peak_gpu_mem_gb": "{:.2f}".format(torch.cuda.max_memory_allocated() / 1024 ** 3),
        "model_path": scene.model_path,
        "mesh_file": mesh_file,
    })
    print("Training is done")

def resolve_model_path(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

def prepare_output_and_logger(args):    
    resolve_model_path(args)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, logger, iteration, pixel_loss, loss, loss_fn, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/pixel_loss', pixel_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration % 1000 == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                pixel_loss_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                total_time = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    image = torch.clamp(renderFunc(viewpoint, scene.triangles, *renderArgs)["render"], 0.0, 1.0)
                    end_event.record()
                    torch.cuda.synchronize()
                    runtime = start_event.elapsed_time(end_event)
                    total_time += runtime

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    pixel_loss_test += loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                pixel_loss_test /= len(config['cameras'])       
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])  
                total_time /= len(config['cameras'])
                fps = 1000.0 / total_time
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], pixel_loss_test, psnr_test, ssim_test, lpips_test))
                logger.log_eval(iteration, config['name'], pixel_loss_test, psnr_test, ssim_test, lpips_test, fps)

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--indoor", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Resolve the output folder up front so everything from here on is logged to <model_path>/logs/
    resolve_model_path(args)
    logger = TrainLogger(args.model_path)
    logger.write_config(args)

    print("Optimizing " + args.model_path)

    lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    lps = lp.extract(args)
    ops = op.extract(args)
    pps = pp.extract(args)

    if args.indoor:
        ops = update_indoor(ops)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    try:
        training(lps,
                 ops,
                 pps,
                 args.test_iterations,
                 args.start_checkpoint,
                 args.debug_from,
                 logger,
                 )
    except BaseException as e:
        logger.write_summary({"status": "failed", "error": repr(e)})
        raise
    
    # All done
    print("\nTraining complete.")