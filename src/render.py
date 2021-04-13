# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from . import util
from . import mesh
from . import renderutils as ru

# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

# ==============================================================================================
#  pixel shader
# ==============================================================================================
def shade(
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        light_pos,
        light_power,
        material,
        min_roughness
    ):

    ################################################################################
    # Texture lookups
    ################################################################################

    kd = material['kd'].sample(gb_texc, gb_texc_deriv)
    ks = material['ks'].sample(gb_texc, gb_texc_deriv)[..., 0:3] # skip alpha
    perturbed_nrm = None
    if 'normal' in material:
        perturbed_nrm = material['normal'].sample(gb_texc, gb_texc_deriv)

    gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True)

    # Separate kd into alpha and color, default alpha = 1
    alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1]) 
    kd = kd[..., 0:3]

    ################################################################################
    # Evaluate BSDF
    ################################################################################

    assert 'bsdf' in material, "Material must specify a BSDF type"
    if material['bsdf'] == 'pbr':
        shaded_col = ru.pbr_bsdf(kd, ks, gb_pos, gb_normal, view_pos, light_pos, min_roughness) * light_power
    elif material['bsdf'] == 'diffuse':
        shaded_col = kd * ru.lambert(gb_normal, util.safe_normalize(light_pos - gb_pos)) * light_power
    elif material['bsdf'] == 'normal':
        shaded_col = (gb_normal + 1.0)*0.5
    elif material['bsdf'] == 'tangent':
        shaded_col = (gb_tangent + 1.0)*0.5
    else:
        assert False, "Invalid BSDF '%s'" % material['bsdf']

    out = torch.cat((shaded_col, alpha), dim=-1)

    return out

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        rast,
        rast_deriv,
        mesh,
        view_pos,
        light_pos,
        light_power,
        resolution,
        min_roughness,
        spp,
        msaa
    ):

    full_res = resolution*spp

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, [resolution, resolution], mag='nearest', min='nearest')
        rast_out_deriv_s = util.scale_img_nhwc(rast_deriv, [resolution, resolution], mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    ################################################################################
    # Interpolate attributes
    ################################################################################

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int())

    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :]
    v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :]
    v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :]
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normal_indices.int())

    # Compute tangent space
    assert mesh.v_nrm is not None and mesh.v_tng is not None
    gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
    gb_tangent, _ = interpolate(mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int()) # Interpolate tangents

    # Texure coordinate
    assert mesh.v_tex is not None
    gb_texc, gb_texc_deriv = interpolate(mesh.v_tex[None, ...], rast_out_s, mesh.t_tex_idx.int(), rast_db=rast_out_deriv_s)

    ################################################################################
    # Shade
    ################################################################################

    color = shade(gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_texc, gb_texc_deriv, 
        view_pos, light_pos, light_power, mesh.material, min_roughness)

    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        color = util.scale_img_nhwc(color, [full_res, full_res], mag='nearest', min='nearest')

    # Return color & raster output for peeling
    return color


# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_mesh(
        ctx,
        mesh,
        mtx_in,
        view_pos,
        light_pos,
        light_power,
        resolution,
        spp                       = 1,
        num_layers                = 1,
        msaa                      = False,
        background                = None,
        antialias                 = True,
        min_roughness             = 0.08
    ):

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    full_res = resolution*spp

    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    light_pos   = prepare_input_vector(light_pos)
    light_power = prepare_input_vector(light_power)
    view_pos    = prepare_input_vector(view_pos)

    # clip space transform
    v_pos_clip = ru.xfm_points(mesh.v_pos[None, ...], mtx_in)

    # Render all layers front-to-back
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), [resolution*spp, resolution*spp]) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [(render_layer(rast, db, mesh, view_pos, light_pos, light_power, resolution, min_roughness, spp, msaa), rast)]

    # Clear to background layer
    if background is not None:
        assert background.shape[1] == resolution and background.shape[2] == resolution
        if spp > 1:
            background = util.scale_img_nhwc(background, [full_res, full_res], mag='nearest', min='nearest')
        accum_col = background
    else:
        accum_col = torch.zeros(size=(1, full_res, full_res, 3), dtype=torch.float32, device='cuda')

    # Composite BACK-TO-FRONT
    for color, rast in reversed(layers):
        alpha     = (rast[..., -1:] > 0) * color[..., 3:4]
        accum_col = torch.lerp(accum_col, color[..., 0:3], alpha)
        if antialias:
            accum_col = dr.antialias(accum_col.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int()) # TODO: need to support bfloat16

    # Downscale to framebuffer resolution. Use avg pooling 
    out = util.avg_pool_nhwc(accum_col, spp) if spp > 1 else accum_col

    return out


