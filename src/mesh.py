# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import torch

from . import util
from . import texture

######################################################################################
# Base mesh class
######################################################################################
class Mesh:
    def __init__(self, v_pos=None, t_pos_idx=None, v_nrm=None, t_nrm_idx=None, v_tex=None, t_tex_idx=None, v_tng=None, t_tng_idx=None, 
    v_weights=None, bone_mtx=None, material=None, base=None):
        self.v_pos = v_pos
        self.v_weights = v_weights
        self.v_nrm = v_nrm
        self.v_tex = v_tex
        self.v_tng = v_tng
        self.t_pos_idx = t_pos_idx
        self.t_nrm_idx = t_nrm_idx
        self.t_tex_idx = t_tex_idx
        self.t_tng_idx = t_tng_idx
        self.material = material
        self.bone_mtx = bone_mtx

        if base is not None:
            self.copy_none(base)

    def copy_none(self, other):
        if self.v_pos is None:
            self.v_pos = other.v_pos
        if self.v_weights is None:
            self.v_weights = other.v_weights
        if self.t_pos_idx is None:
            self.t_pos_idx = other.t_pos_idx
        if self.v_nrm is None:
            self.v_nrm = other.v_nrm
        if self.t_nrm_idx is None:
            self.t_nrm_idx = other.t_nrm_idx
        if self.v_tex is None:
            self.v_tex = other.v_tex
        if self.t_tex_idx is None:
            self.t_tex_idx = other.t_tex_idx
        if self.v_tng is None:
            self.v_tng = other.v_tng
        if self.t_tng_idx is None:
            self.t_tng_idx = other.t_tng_idx
        if self.material is None:
            self.material = other.material
        if self.bone_mtx is None:
            self.bone_mtx = other.bone_mtx

    def get_frames(self):
        return self.bone_mtx.shape[0] if self.bone_mtx is not None else 1

    def clone(self):
        out = Mesh(base=self)
        if out.v_pos is not None:
            out.v_pos = out.v_pos.clone()
        if out.v_weights is not None:
            out.v_weights = out.v_weights.clone()
        if out.t_pos_idx is not None:
            out.t_pos_idx = out.t_pos_idx.clone()
        if out.v_nrm is not None:
            out.v_nrm = out.v_nrm.clone()
        if out.t_nrm_idx is not None:
            out.t_nrm_idx = out.t_nrm_idx.clone()
        if out.v_tex is not None:
            out.v_tex = out.v_tex.clone()
        if out.t_tex_idx is not None:
            out.t_tex_idx = out.t_tex_idx.clone()
        if out.v_tng is not None:
            out.v_tng = out.v_tng.clone()
        if out.t_tng_idx is not None:
            out.t_tng_idx = out.t_tng_idx.clone()
        if out.bone_mtx is not None:
            out.bone_mtx = out.bone_mtx.clone()
        return out

    def eval(self, params={}):
        return self

######################################################################################
# Compute AABB
######################################################################################
def aabb(mesh):
    return torch.min(mesh.v_pos, dim=0).values, torch.max(mesh.v_pos, dim=0).values

######################################################################################
# Align base mesh to reference mesh:move & rescale to match bounding boxes.
######################################################################################
def unit_size(mesh):
    with torch.no_grad():
        vmin, vmax = aabb(mesh)
        scale = 2 / torch.max(vmax - vmin).item()
        v_pos = mesh.v_pos - (vmax + vmin) / 2 # Center mesh on origin
        v_pos = v_pos * scale                  # Rescale to unit size

        return Mesh(v_pos, base=mesh)

######################################################################################
# Center & scale mesh for rendering
#
# TODO: It should be better to compute camera position from animated reference mesh 
# instead of centering and scaling all meshes
######################################################################################
def center_by_reference(base_mesh, ref_aabb, scale):
    center = (ref_aabb[0] + ref_aabb[1]) * 0.5
    scale = scale / torch.max(ref_aabb[1] - ref_aabb[0]).item()
    v_pos = (base_mesh.v_pos - center[None, ...]) * scale
    return Mesh(v_pos, base=base_mesh)

######################################################################################
# Rescale base-mesh from NDC [-1, 1] space to same dimensions as reference mesh
######################################################################################
def align_with_reference(base_mesh, ref_mesh): # TODO: Fix normals?
    class mesh_op_align:
        def __init__(self, base_mesh, ref_mesh):
            self.base_mesh = base_mesh
            with torch.no_grad():
                b_vmin, b_vmax = aabb(base_mesh.eval())
                r_vmin, r_vmax = aabb(ref_mesh.eval())
                b_size = (b_vmax - b_vmin)
                self.offset = (r_vmax + r_vmin) / 2
                self.scale = (r_vmax - r_vmin) / torch.where(b_size > 1e-6, b_size, torch.ones_like(b_size))

        def eval(self, params={}):
            base_mesh = self.base_mesh.eval(params)
            v_pos = base_mesh.v_pos * self.scale[None, ...] + self.offset[None, ...]
            return Mesh(v_pos, base=base_mesh)

    return mesh_op_align(base_mesh, ref_mesh)

######################################################################################
# Skinning
######################################################################################

# Helper function to skin homogeneous vectors
def _skin_hvec(bone_mtx, weights, attr):
    attr_out = torch.matmul(attr[None, ...], bone_mtx) * torch.transpose(weights, 0, 1)[..., None]
    return attr_out.sum(dim=0)[:, :3]

def skinning(mesh):
    class mesh_op_skinning:
        def __init__(self, input):
            self.input = input
            
            mesh = self.input.eval()
            t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy() 
            if mesh.t_nrm_idx is not None:
                self.nrm_remap = self._compute_remap(t_pos_idx, mesh.v_nrm.shape[0], mesh.t_nrm_idx.detach().cpu().numpy())
            if mesh.t_tng_idx is not None:
                self.tng_remap = self._compute_remap(t_pos_idx, mesh.v_tng.shape[0], mesh.t_tng_idx.detach().cpu().numpy())

        # Compute an index list with corresponding vertex index for each normal/tangent. Vertices may have multiple normals/tangents, but not the other way around
        def _compute_remap(self, t_pos_idx, n_attrs, t_attr_idx):
            assert len(t_pos_idx) == len(t_attr_idx)

            attr_vtx_idx = [None] * n_attrs
            for ti in range(0, len(t_pos_idx)):
                for vi in range(0, 3):
                    assert attr_vtx_idx[t_attr_idx[ti][vi]] is None or attr_vtx_idx[t_attr_idx[ti][vi]] == t_pos_idx[ti][vi], "Trying to skin a mesh with shared normals (normal with 2 sets of skinning weights)"
                    attr_vtx_idx[t_attr_idx[ti][vi]] = t_pos_idx[ti][vi]

            return torch.tensor(attr_vtx_idx, dtype=torch.int64, device='cuda')

        def eval(self, params={}):
            imesh = self.input.eval(params)

            if imesh.v_weights is None or imesh.bone_mtx is None:
                return imesh

            # Compute frame (assume looping animation). Note, bone_mtx is stored [Frame, Bone, ...]
            t_idx = params['time'] if 'time' in params else 0
            t_idx = (t_idx % imesh.bone_mtx.shape[0]) # Loop animation
            bone_mtx    = imesh.bone_mtx[t_idx, ...]
            bone_mtx_it = torch.transpose(torch.inverse(bone_mtx), -2, -1)

            weights = imesh.v_weights
            assert weights.shape[1] == bone_mtx.shape[0]

            # Normalize weights
            weights = torch.abs(weights) # TODO: This stabilizes training, but I don't know why. All weights are already clamped to >0
            weights = weights / torch.sum(weights, dim=1, keepdim=True)

            # Skin position
            v_pos_out = _skin_hvec(bone_mtx, weights, util.to_hvec(imesh.v_pos, 1))
            
            # Skin normal
            v_nrm_out = None
            if imesh.v_nrm is not None:
                v_nrm_out = _skin_hvec(bone_mtx_it, weights[self.nrm_remap, ...], util.to_hvec(imesh.v_nrm, 0))
                v_nrm_out = util.safe_normalize(v_nrm_out)
            
            # Skin tangent
            v_tng_out = None
            if imesh.v_tng is not None:
                v_tng_out = _skin_hvec(bone_mtx, weights[self.tng_remap, ...], util.to_hvec(imesh.v_tng, 0))
                v_tng_out = util.safe_normalize(v_tng_out)

            if torch.is_anomaly_enabled():
                assert torch.all(torch.isfinite(v_pos_out))
                assert v_nrm_out is None or torch.all(torch.isfinite(v_nrm_out))
                assert v_tng_out is None or torch.all(torch.isfinite(v_tng_out))

            return Mesh(v_pos=v_pos_out[:, :3], v_nrm=v_nrm_out, v_tng=v_tng_out, base=imesh)

    return mesh_op_skinning(mesh)

# Skinning helper functions
def guess_weights(base_mesh, ref_mesh, N=10):
    base_v_pos = base_mesh.v_pos.detach().cpu().numpy()
    ref_v_pos = ref_mesh.v_pos.detach().cpu().numpy()
    ref_v_weights = ref_mesh.v_weights.detach().cpu().numpy()
    base_v_weights = np.zeros((base_v_pos.shape[0], ref_v_weights.shape[1]), dtype=np.float32)
    
    for v_idx, vtx in enumerate(base_v_pos):
        # Compute distance from current vertex to vertices in ref_mesh
        diff = ref_v_pos - vtx[None, ...]
        dist = np.sum(diff * diff, axis=-1)
        idxs = np.argpartition(dist, N)        

        # Get the N nearest vertices
        sum_w = 0.0
        sum_vtx_w = np.zeros_like(ref_v_weights[0,...])
        for i in idxs[:N]:
            sum_w += 1.0 / max(dist[i], 0.001)
            sum_vtx_w += ref_v_weights[i, ...] / max(dist[i], 0.001)
        base_v_weights[v_idx, ...] = sum_vtx_w / sum_w
    
    return base_v_weights

def random_weights(base_mesh, ref_mesh):
    init = np.random.uniform(size=(base_mesh.v_pos.shape[0], ref_mesh.v_weights.shape[1]), low=0.0, high=1.0)
    return init / np.sum(init, axis=1, keepdims=True)


######################################################################################
# Simple smooth vertex normal computation
######################################################################################
def auto_normals(mesh):
    class mesh_op_auto_normals:
        def __init__(self, input):
            self.input = input

        def eval(self, params={}):
            imesh = self.input.eval(params)

            i0 = imesh.t_pos_idx[:, 0]
            i1 = imesh.t_pos_idx[:, 1]
            i2 = imesh.t_pos_idx[:, 2]

            v0 = imesh.v_pos[i0, :]
            v1 = imesh.v_pos[i1, :]
            v2 = imesh.v_pos[i2, :]

            face_normals = torch.cross(v1 - v0, v2 - v0)

            # Splat face normals to vertices
            v_nrm = torch.zeros_like(imesh.v_pos)
            v_nrm.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
            v_nrm.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
            v_nrm.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

            # Normalize, replace zero (degenerated) normals with some default value
            v_nrm = torch.where(util.dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))

            self.v_nrm = util.safe_normalize(v_nrm)

            if torch.is_anomaly_enabled():
                assert torch.all(torch.isfinite(self.v_nrm))

            return Mesh(v_nrm = self.v_nrm, t_nrm_idx=imesh.t_pos_idx, base = imesh)

    return mesh_op_auto_normals(mesh)

######################################################################################
# Compute tangent space from texture map coordinates
# Follows http://www.mikktspace.com/ conventions
######################################################################################
def compute_tangents(mesh):
    class mesh_op_compute_tangents:
        def __init__(self, input):
            self.input = input

        def eval(self, params={}):
            imesh = self.input.eval(params)

            vn_idx = [None] * 3
            pos = [None] * 3
            tex = [None] * 3
            for i in range(0,3):
                pos[i] = imesh.v_pos[imesh.t_pos_idx[:, i]]
                tex[i] = imesh.v_tex[imesh.t_tex_idx[:, i]]
                vn_idx[i] = imesh.t_nrm_idx[:, i]

            tangents = torch.zeros_like(imesh.v_nrm)
            tansum   = torch.zeros_like(imesh.v_nrm)

            # Compute tangent space for each triangle
            uve1 = tex[1] - tex[0]
            uve2 = tex[2] - tex[0]
            pe1  = pos[1] - pos[0]
            pe2  = pos[2] - pos[0]
            
            nom   = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
            denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])
            
            # Avoid division by zero for degenerated texture coordinates
            tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

            # Update all 3 vertices
            for i in range(0,3):
                idx = vn_idx[i][:, None].repeat(1,3)
                tangents.scatter_add_(0, idx, tang)                # tangents[n_i] = tangents[n_i] + tang
                tansum.scatter_add_(0, idx, torch.ones_like(tang)) # tansum[n_i] = tansum[n_i] + 1
            tangents = tangents / tansum

            # Normalize and make sure tangent is perpendicular to normal
            tangents = util.safe_normalize(tangents)
            tangents = util.safe_normalize(tangents - util.dot(tangents, imesh.v_nrm) * imesh.v_nrm)

            self.v_tng = tangents

            if torch.is_anomaly_enabled():
                assert torch.all(torch.isfinite(tangents))

            return Mesh(v_tng=self.v_tng, t_tng_idx=imesh.t_nrm_idx, base=imesh)

    return mesh_op_compute_tangents(mesh)

######################################################################################
# Subdivide each triangle into 4 new ones. Edge midpoint subdivision
######################################################################################
def subdivide(mesh, steps=1):
    class mesh_op_subdivide:
        def __init__(self, input):
            self.input = input
            self.new_vtx_idx = [None] * 4
            self.new_tri_idx = [None] * 4

            imesh = self.input.eval()
            v_attr = v_attr_orig = [imesh.v_pos, imesh.v_nrm, imesh.v_tex, imesh.v_tng]
            v_idx = v_idx_orig = [imesh.t_pos_idx, imesh.t_nrm_idx, imesh.t_tex_idx, imesh.t_tng_idx]
            
            for i, attr in enumerate(v_attr):
                if attr is not None:
                    tri_idx = v_idx[i].cpu().numpy()

                    # Find unique edges
                    edge_fetch_a = []
                    edge_fetch_b = []
                    edge_verts = {}
                    for tri in tri_idx:
                        for e_idx in range(0, 3):
                            v0 = tri[e_idx]
                            v1 = tri[(e_idx + 1) % 3]
                            if (v1, v0) not in edge_verts.keys():
                                edge_verts[(v0, v1)] = [len(edge_fetch_a), v0, v1]
                                edge_fetch_a += [v0]
                                edge_fetch_b += [v1]

                    # Create vertex fetch lists for computing midpoint vertices
                    self.new_vtx_idx[i] = [torch.tensor(edge_fetch_a, dtype=torch.int64, device='cuda'), torch.tensor(edge_fetch_b, dtype=torch.int64, device='cuda')]

                    # Create subdivided triangles
                    new_tri_idx = []
                    for tri in tri_idx:
                        v0, v1, v2= tri
                        h0 = (edge_verts[(v0, v1)][0] if (v0, v1) in edge_verts.keys() else edge_verts[(v1, v0)][0]) + attr.shape[0]
                        h1 = (edge_verts[(v1, v2)][0] if (v1, v2) in edge_verts.keys() else edge_verts[(v2, v1)][0]) + attr.shape[0]
                        h2 = (edge_verts[(v2, v0)][0] if (v2, v0) in edge_verts.keys() else edge_verts[(v0, v2)][0]) + attr.shape[0]
                        new_tri_idx += [[v0, h0, h2], [h0, v1, h1], [h1, v2, h2], [h0, h1, h2]]
                    self.new_tri_idx[i] = torch.tensor(new_tri_idx, dtype=torch.int64, device='cuda')

        def eval(self, params={}):
            imesh = self.input.eval(params)

            v_attr = v_attr_orig = [imesh.v_pos, imesh.v_nrm, imesh.v_tex, imesh.v_tng]
            v_idx = v_idx_orig = [imesh.t_pos_idx, imesh.t_nrm_idx, imesh.t_tex_idx, imesh.t_tng_idx]

            for i, attr in enumerate(v_attr):
                if attr is not None:
                    # Create new edge midpoint attributes
                    edge_attr = (attr[self.new_vtx_idx[i][0], :] + attr[self.new_vtx_idx[i][1], :]) * 0.5
                    v_attr[i] = torch.cat([attr, edge_attr], dim=0)
                    
                    # Copy new triangle lists
                    v_idx[i] = self.new_tri_idx[i]

            return Mesh(v_attr[0], v_idx[0], v_attr[1], v_idx[1], v_attr[2], v_idx[2], v_attr[3], v_idx[3], base=imesh)

    x = mesh
    for i in range(steps):
        x = mesh_op_subdivide(x)

    bm = mesh.eval()
    sm = x.eval()
    v_attr_orig = [bm.v_pos, bm.v_nrm, bm.v_tex, bm.v_tng]
    v_attr = [sm.v_pos, sm.v_nrm, sm.v_tex, sm.v_tng]
    v_idx_orig = [bm.t_pos_idx, bm.t_nrm_idx, bm.t_tex_idx, bm.t_tng_idx]
    v_idx = [sm.t_pos_idx, sm.t_nrm_idx, sm.t_tex_idx, sm.t_tng_idx]
    print("Subdivided mesh:")
    print("    Attrs:   [%6d, %6d, %6d, %6d] -> [%6d, %6d, %6d, %6d]" % tuple(list((a.shape[0] if a is not None else 0) for a in v_attr_orig) + list((a.shape[0] if a is not None else 0) for a in v_attr)))
    print("    Indices: [%6d, %6d, %6d, %6d] -> [%6d, %6d, %6d, %6d]" % tuple(list((a.shape[0] if a is not None else 0) for a in v_idx_orig) + list((a.shape[0] if a is not None else 0) for a in v_idx)))

    return x

######################################################################################
# Displacement mapping
######################################################################################
def displace(mesh, displacement_map, scale=1.0, keep_connectivity=True):
    class mesh_op_displace:
        def __init__(self, input, displacement_map, scale, keep_connectivity):
            self.input = input
            self.displacement_map = displacement_map
            self.scale = scale
            self.keep_connectivity = keep_connectivity
        
        def eval(self, params={}):
            imesh = self.input.eval(params)

            if self.keep_connectivity:
                vd   = torch.zeros_like(imesh.v_pos)
                vd_n = torch.zeros_like(imesh.v_pos)
                for i in range(0, 3):
                    v = imesh.v_pos[imesh.t_pos_idx[:, i], :]
                    n = imesh.v_nrm[imesh.t_nrm_idx[:, i], :]
                    t = imesh.v_tex[imesh.t_tex_idx[:, i], :]
                    v_displ = v + n * self.scale * util.tex_2d(self.displacement_map, t)
            
                    splat_idx = imesh.t_pos_idx[:, i, None].repeat(1,3)
                    vd.scatter_add_(0, splat_idx, v_displ)
                    vd_n.scatter_add_(0, splat_idx, torch.ones_like(v_displ))

                return Mesh(vd / vd_n, base=imesh)
            else:
                vd   = torch.zeros([imesh.v_tex.shape[0], 3], dtype=torch.float32, device='cuda')
                vd_n = torch.zeros([imesh.v_tex.shape[0], 3], dtype=torch.float32, device='cuda')
                for i in range(0, 3):
                    v = imesh.v_pos[imesh.t_pos_idx[:, i], :]
                    n = imesh.v_nrm[imesh.t_nrm_idx[:, i], :]
                    t = imesh.v_tex[imesh.t_tex_idx[:, i], :]
                    v_displ = v + n * self.scale * util.tex_2d(self.displacement_map, t)
            
                    splat_idx = imesh.t_tex_idx[:, i, None].repeat(1, 3)
                    vd.scatter_add_(0, splat_idx, v_displ)
                    vd_n.scatter_add_(0, splat_idx, torch.ones_like(v_displ))

                return Mesh(vd / vd_n, mesh.t_tex_idx, base=imesh)

    return mesh_op_displace(mesh, displacement_map, scale, keep_connectivity)


######################################################################################
# Utilities to merge meshes / materials. No mesh-ops or differentiable stuff here.
######################################################################################

def merge(mesh_a, mesh_b):
    def _merge_attr_idx(a, b, a_idx, b_idx):
        if a is None and b is None:
            return None, None
        elif a is not None and b is None:
            return a, a_idx
        elif a is None and b is not None:
            return b, b_idx
        else:
            return torch.cat((a, b), dim=0), torch.cat((a_idx, b_idx + a.shape[0]), dim=0)

    v_pos, t_pos_idx = _merge_attr_idx(mesh_a.v_pos, mesh_b.v_pos, mesh_a.t_pos_idx, mesh_b.t_pos_idx)
    v_nrm, t_nrm_idx = _merge_attr_idx(mesh_a.v_nrm, mesh_b.v_nrm, mesh_a.t_nrm_idx, mesh_b.t_nrm_idx)
    v_tng, t_tng_idx = _merge_attr_idx(mesh_a.v_tng, mesh_b.v_tng, mesh_a.t_tng_idx, mesh_b.t_tng_idx)
    v_tex, t_tex_idx = _merge_attr_idx(mesh_a.v_tex, mesh_b.v_tex, mesh_a.t_tex_idx, mesh_b.t_tex_idx)

    if mesh_a.v_weights is None and mesh_b.v_weights is None:
        v_weights, bone_mtx = None, None
    elif mesh_a.v_weights is not None and mesh_b.v_weights is None:
        v_weights, bone_mtx = mesh_a.v_weights, mesh_a.bone_mtx
    elif mesh_a.v_weights is None and mesh_b.v_weights is not None:
        v_weights, bone_mtx = mesh_b.v_weights, mesh_b.bone_mtx
    else:
        if torch.all(mesh_a.bone_mtx == mesh_b.bone_mtx): # TODO: Wanted to test if same pointer
            bone_mtx = mesh_a.bone_mtx
            v_weights = torch.cat((mesh_a.v_weights, mesh_b.v_weights), dim=0)
        else:
            bone_mtx = torch.cat((mesh_a.bone_mtx, mesh_b.bone_mtx), dim=1) # Frame, Bone, ...
        
            # Weights need to be increased to account for all bones
            v_wa = torch.nn.functional.pad(mesh_a.v_weights, [0, mesh_b.v_weights.shape[1]]) #Pad weights_a with shape of weights_b
            v_wb = torch.nn.functional.pad(mesh_b.v_weights, [mesh_a.v_weights.shape[1], 0]) #Pad weights_b with shape of weights_a
            v_weights = torch.cat((v_wa, v_wb), dim=0)

    return Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx, v_nrm=v_nrm, t_nrm_idx=t_nrm_idx, v_tng=v_tng, t_tng_idx=t_tng_idx, v_tex=v_tex, t_tex_idx=t_tex_idx, v_weights=v_weights, bone_mtx=bone_mtx, base=mesh_a)
