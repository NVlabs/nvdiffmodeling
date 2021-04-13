# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import time
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

import src.renderutils as ru
from src import obj
from src import util
from src import mesh
from src import texture
from src import render
from src import regularizer
from src.mesh import Mesh

RADIUS = 3.5

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Utility mesh loader
###############################################################################

def load_mesh(filename, mtl_override=None):
    name, ext = os.path.splitext(filename)
    if ext == ".obj":
        return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override)
    assert False, "Invalid mesh file extension"

###############################################################################
# Loss setup
###############################################################################

def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relativel2":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

def optimize_mesh(
    FLAGS,
    out_dir, 
    log_interval=10,
    mesh_scale=2.0
    ):

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mesh"), exist_ok=True)

    # Projection matrix
    proj_mtx = util.projection(x=0.4, f=1000.0)

    # Guess learning rate if not specified
    if FLAGS.learning_rate is None:
        FLAGS.learning_rate = 0.01

    # Reference mesh
    ref_mesh = load_mesh(FLAGS.ref_mesh, FLAGS.mtl_override)
    print("Ref mesh has %d triangles and %d vertices." % (ref_mesh.t_pos_idx.shape[0], ref_mesh.v_pos.shape[0]))

    # Check if the training texture resolution is acceptable
    ref_texture_res = np.maximum(ref_mesh.material['kd'].getRes(), ref_mesh.material['ks'].getRes())
    if 'normal' in ref_mesh.material:
        ref_texture_res = np.maximum(ref_texture_res, ref_mesh.material['normal'].getRes())
    if FLAGS.texture_res[0] < ref_texture_res[0] or FLAGS.texture_res[1] < ref_texture_res[1]:
        print("---> WARNING: Picked a texture resolution lower than the reference mesh [%d, %d] < [%d, %d]" % (FLAGS.texture_res[0], FLAGS.texture_res[1], ref_texture_res[0], ref_texture_res[1]))

    # Base mesh
    base_mesh = load_mesh(FLAGS.base_mesh)
    print("Base mesh has %d triangles and %d vertices." % (base_mesh.t_pos_idx.shape[0], base_mesh.v_pos.shape[0]))
    print("Avg edge length: %f" % regularizer.avg_edge_length(base_mesh))

    # Create normalized size versions of the base and reference meshes. Normalized base_mesh is important as it makes it easier to configure learning rate.
    normalized_base_mesh = mesh.unit_size(base_mesh)
    normalized_ref_mesh = mesh.unit_size(ref_mesh)

    assert not FLAGS.random_train_res or FLAGS.custom_mip, "Random training resolution requires custom mip."

    # ==============================================================================================
    #  Initialize weights / variables for trainable mesh
    # ==============================================================================================
    trainable_list = [] 

    v_pos_opt = normalized_base_mesh.v_pos.clone().detach().requires_grad_(True)

    # Trainable normal map, initialize to (0,0,1) & make sure normals are always in positive hemisphere
    if FLAGS.random_textures:
        normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip)
    else:
        if 'normal' not in ref_mesh.material:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip)
        else:
            normal_map_opt = texture.create_trainable(ref_mesh.material['normal'], FLAGS.texture_res, not FLAGS.custom_mip)

    # Setup Kd, Ks albedo and specular textures
    if FLAGS.random_textures:
        if FLAGS.layers > 1:
            kd_map_opt = texture.create_trainable(np.random.uniform(size=FLAGS.texture_res + [4], low=0.0, high=1.0), FLAGS.texture_res, not FLAGS.custom_mip)
        else:
            kd_map_opt = texture.create_trainable(np.random.uniform(size=FLAGS.texture_res + [3], low=0.0, high=1.0), FLAGS.texture_res, not FLAGS.custom_mip)

        ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
        ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=FLAGS.min_roughness, high=1.0)
        ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=1.0)
        ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip)
    else:
        kd_map_opt = texture.create_trainable(ref_mesh.material['kd'], FLAGS.texture_res, not FLAGS.custom_mip)
        ks_map_opt = texture.create_trainable(ref_mesh.material['ks'], FLAGS.texture_res, not FLAGS.custom_mip)

    # Trainable displacement map
    displacement_map_var = None
    if FLAGS.subdivision > 0:
        displacement_map_var = torch.tensor(np.zeros(FLAGS.texture_res + [1], dtype=np.float32), dtype=torch.float32, device='cuda', requires_grad=True)

    # Add trainable arguments according to config
    if not 'position' in FLAGS.skip_train:
        trainable_list += [v_pos_opt]        
    if not 'normal' in FLAGS.skip_train:
        trainable_list += normal_map_opt.getMips()
    if not 'kd' in FLAGS.skip_train:
        trainable_list += kd_map_opt.getMips()
    if not 'ks' in FLAGS.skip_train:
        trainable_list += ks_map_opt.getMips()
    if not 'displacement' in FLAGS.skip_train and displacement_map_var is not None:
        trainable_list += [displacement_map_var]

    # ==============================================================================================
    #  Setup material for optimized mesh
    # ==============================================================================================

    opt_material = {
        'bsdf'   : ref_mesh.material['bsdf'],
        'kd'     : kd_map_opt,
        'ks'     : ks_map_opt,
        'normal' : normal_map_opt
    }

    # ==============================================================================================
    #  Setup reference mesh. Compute tangentspace and animate with skinning
    # ==============================================================================================

    render_ref_mesh = mesh.compute_tangents(ref_mesh)
    
    # Compute AABB of reference mesh. Used for centering during rendering TODO: Use pre frame AABB?
    ref_mesh_aabb = mesh.aabb(render_ref_mesh.eval())

    # ==============================================================================================
    #  Setup base mesh operation graph, precomputes topology etc.
    # ==============================================================================================

    # Create optimized mesh with trainable positions 
    opt_base_mesh = Mesh(v_pos_opt, normalized_base_mesh.t_pos_idx, material=opt_material, base=normalized_base_mesh)

    # Scale from [-1, 1] local coordinate space to match extents of the reference mesh
    opt_base_mesh = mesh.align_with_reference(opt_base_mesh, ref_mesh)

    # Compute smooth vertex normals
    opt_base_mesh = mesh.auto_normals(opt_base_mesh)

    # Set up tangent space
    opt_base_mesh = mesh.compute_tangents(opt_base_mesh)

    # Subdivide if we're doing displacement mapping
    if FLAGS.subdivision > 0:
        # Subdivide & displace optimized mesh
        subdiv_opt_mesh = mesh.subdivide(opt_base_mesh, steps=FLAGS.subdivision)
        opt_detail_mesh = mesh.displace(subdiv_opt_mesh, displacement_map_var, FLAGS.displacement, keep_connectivity=True)
    else:
        opt_detail_mesh = opt_base_mesh

    # Laplace regularizer
    if FLAGS.relative_laplacian:
        with torch.no_grad():
            orig_opt_base_mesh = opt_base_mesh.eval().clone()
        lap_loss_fn = regularizer.laplace_regularizer_const(opt_detail_mesh, orig_opt_base_mesh)
    else:
        lap_loss_fn = regularizer.laplace_regularizer_const(opt_detail_mesh)

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    optimizer  = torch.optim.Adam(trainable_list, lr=FLAGS.learning_rate)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002))) 

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    # Background color
    if FLAGS.background == 'checker':
        background = torch.tensor(util.checkerboard(FLAGS.display_res, 8), dtype=torch.float32, device='cuda')
    elif FLAGS.background == 'white':
        background = torch.ones((1, FLAGS.display_res, FLAGS.display_res, 3), dtype=torch.float32, device='cuda')
    else:
        background = None

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    ang = 0.0
    img_loss_vec = []
    lap_loss_vec = []
    iter_dur_vec = []
    glctx = dr.RasterizeGLContext()
    for it in range(FLAGS.iter+1):
        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
        save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
        if display_image or save_image:
            eye = np.array(FLAGS.camera_eye)
            up  = np.array(FLAGS.camera_up)
            at  = np.array([0,0,0])
            a_mv =  util.lookAt(eye, at, up)
            a_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]
            a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
            a_campos = np.linalg.inv(a_mv)[None, :3, 3]

            params = {'mvp' : a_mvp, 'lightpos' : a_lightpos, 'campos' : a_campos, 'resolution' : [FLAGS.display_res, FLAGS.display_res], 
            'time' : 0}

            # Render images, don't need to track any gradients
            with torch.no_grad():
                # Center meshes
                _opt_detail = mesh.center_by_reference(opt_detail_mesh.eval(params), ref_mesh_aabb, mesh_scale)
                _opt_ref    = mesh.center_by_reference(render_ref_mesh.eval(params), ref_mesh_aabb, mesh_scale)

                # Render
                if FLAGS.subdivision > 0:
                    _opt_base   = mesh.center_by_reference(opt_base_mesh.eval(params), ref_mesh_aabb, mesh_scale)
                    img_base = render.render_mesh(glctx, _opt_base, a_mvp, a_campos, a_lightpos, FLAGS.light_power, FLAGS.display_res, 
                        num_layers=FLAGS.layers, background=background, min_roughness=FLAGS.min_roughness)
                    img_base = util.scale_img_nhwc(img_base, [FLAGS.display_res, FLAGS.display_res])

                img_opt = render.render_mesh(glctx, _opt_detail, a_mvp, a_campos, a_lightpos, FLAGS.light_power, FLAGS.display_res, 
                    num_layers=FLAGS.layers, background=background, min_roughness=FLAGS.min_roughness)
                img_ref = render.render_mesh(glctx, _opt_ref, a_mvp, a_campos, a_lightpos, FLAGS.light_power, FLAGS.display_res, 
                    num_layers=1, spp=FLAGS.spp, background=background, min_roughness=FLAGS.min_roughness)

                # Rescale
                img_opt  = util.scale_img_nhwc(img_opt,  [FLAGS.display_res, FLAGS.display_res])
                img_ref  = util.scale_img_nhwc(img_ref,  [FLAGS.display_res, FLAGS.display_res])

                if FLAGS.subdivision > 0:
                    img_disp = torch.clamp(torch.abs(displacement_map_var[None, ...]), min=0.0, max=1.0).repeat(1,1,1,3)
                    img_disp = util.scale_img_nhwc(img_disp, [FLAGS.display_res, FLAGS.display_res])
                    result_image = torch.cat([img_base, img_opt, img_ref], axis=2)
                else:
                    result_image = torch.cat([img_opt, img_ref], axis=2)

            result_image[0] = util.tonemap_srgb(result_image[0])
            np_result_image = result_image[0].detach().cpu().numpy()
            if display_image:
                util.display_image(np_result_image, size=FLAGS.display_res, title='%d / %d' % (it, FLAGS.iter))
            if save_image:
                util.save_image(out_dir + '/' + ('img_%06d.png' % img_cnt), np_result_image)
                img_cnt = img_cnt+1

        # ==============================================================================================
        #  Initialize training
        # ==============================================================================================

        iter_start_time = time.time()
        img_loss = torch.zeros([1], dtype=torch.float32, device='cuda')
        lap_loss = torch.zeros([1], dtype=torch.float32, device='cuda')

        iter_res = FLAGS.train_res
        iter_spp = FLAGS.spp
        if FLAGS.random_train_res:
            # Random resolution, 16x16 -> train_res. Scale up sample count so we always land close to train_res*samples_per_pixel samples
            iter_res = np.random.randint(16, FLAGS.train_res+1)
            iter_spp = FLAGS.spp * (FLAGS.train_res // iter_res)

        mvp = np.zeros((FLAGS.batch, 4,4),  dtype=np.float32)
        campos   = np.zeros((FLAGS.batch, 3), dtype=np.float32)
        lightpos = np.zeros((FLAGS.batch, 3), dtype=np.float32)

        # ==============================================================================================
        #  Build transform stack for minibatching
        # ==============================================================================================
        for b in range(FLAGS.batch):
            # Random rotation/translation matrix for optimization.
            r_rot      = util.random_rotation_translation(0.25)
            r_mv       = np.matmul(util.translate(0, 0, -RADIUS), r_rot)
            mvp[b]     = np.matmul(proj_mtx, r_mv).astype(np.float32)
            campos[b]  = np.linalg.inv(r_mv)[:3, 3]
            lightpos[b] = util.cosine_sample(campos[b])*RADIUS


        params = {'mvp' : mvp, 'lightpos' : lightpos, 'campos' : campos, 'resolution' : [iter_res, iter_res], 'time' : 0}

        # Random bg color
        randomBgColor = torch.rand(FLAGS.batch, iter_res, iter_res, 3, dtype=torch.float32, device='cuda')

        # ==============================================================================================
        #  Evaluate all mesh ops (may change when positions are modified etc) and center/align meshes
        # ==============================================================================================
        _opt_ref  = mesh.center_by_reference(render_ref_mesh.eval(params), ref_mesh_aabb, mesh_scale)
        _opt_detail = mesh.center_by_reference(opt_detail_mesh.eval(params), ref_mesh_aabb, mesh_scale)

        # ==============================================================================================
        #  Render reference mesh
        # ==============================================================================================
        with torch.no_grad():
            color_ref = render.render_mesh(glctx, _opt_ref, mvp, campos, lightpos, FLAGS.light_power, iter_res, 
                spp=iter_spp, num_layers=1, background=randomBgColor, min_roughness=FLAGS.min_roughness)

        # ==============================================================================================
        #  Render the trainable mesh
        # ==============================================================================================
        color_opt = render.render_mesh(glctx, _opt_detail, mvp, campos, lightpos, FLAGS.light_power, iter_res, 
            spp=iter_spp, num_layers=FLAGS.layers, msaa=True , background=randomBgColor, 
            min_roughness=FLAGS.min_roughness)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        # Image-space loss
        img_loss = image_loss_fn(color_opt, color_ref)

        # Compute laplace loss
        lap_loss = lap_loss_fn.eval(params)

        # Debug, store every training iteration
        # result_image = torch.cat([color_opt, color_ref], axis=2)
        # np_result_image = result_image[0].detach().cpu().numpy()
        # util.save_image(out_dir + '/' + ('train_%06d.png' % it), np_result_image)

        # Log losses
        img_loss_vec.append(img_loss.item())
        lap_loss_vec.append(lap_loss.item())

        # Schedule for laplacian loss weight
        if it == 0:
            if FLAGS.laplacian_factor is not None:
                lap_fac = FLAGS.laplacian_factor
            else:
                ratio = 0.1 / lap_loss.item() # Hack that assumes RMSE ~= 0.1
                lap_fac = ratio * 0.25
            min_lap_fac = lap_fac * 0.02
        else:
            lap_fac = (lap_fac - min_lap_fac) * 10**(-it*0.000001) + min_lap_fac

        # Compute total aggregate loss
        total_loss = img_loss + lap_loss * lap_fac

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================

        normal_map_opt.clamp_(min=-1, max=1)
        kd_map_opt.clamp_(min=0, max=1)
        ks_map_opt.clamp_rgb_(minR=0, maxR=1, minG=FLAGS.min_roughness, maxG=1.0, minB=0.0, maxB=1.0)

        iter_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Log & save outputs
        # ==============================================================================================

        # Print/save log.
        if log_interval and (it % log_interval == 0):
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            lap_loss_avg = np.mean(np.asarray(lap_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
            remaining_time = (FLAGS.iter-it)*iter_dur_avg
            print("iter=%5d, img_loss=%.6f, lap_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                (it, img_loss_avg, lap_loss_avg*lap_fac, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))

    # Save final mesh to file
    obj.write_obj(os.path.join(out_dir, "mesh/"), opt_base_mesh.eval())

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='diffmodeling')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', type=int, default=512)
    parser.add_argument('-rtr', '--random-train-res', action='store_true', default=False)
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=None)
    parser.add_argument('-lp', '--light-power', type=float, default=5.0)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-sd', '--subdivision', type=int, default=0)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-lf', '--laplacian-factor', type=float, default=None)
    parser.add_argument('-rl', '--relative-laplacian', type=bool, default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relativel2'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str)
    
    FLAGS = parser.parse_args()

    FLAGS.camera_eye = [0.0, 0.0, RADIUS]
    FLAGS.camera_up  = [0.0, 1.0, 0.0]
    FLAGS.skip_train = []
    FLAGS.displacement = 0.15
    FLAGS.mtl_override = None

    if FLAGS.config is not None:
        with open(FLAGS.config) as f:
            data = json.load(f)
            for key in data:
                print(key, data[key])
                FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    if FLAGS.out_dir is None:
        out_dir = 'out/cube_%d' % (FLAGS.train_res)
    else:
        out_dir = 'out/' + FLAGS.out_dir

    optimize_mesh(FLAGS, out_dir)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
