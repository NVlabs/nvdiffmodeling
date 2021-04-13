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
# Computes the avergage edge length of a mesh. 
# Rough estimate of the tessellation of a mesh. Can be used e.g. to clamp gradients
######################################################################################
def avg_edge_length(opt_mesh):
    with torch.no_grad():
        opt_mesh = opt_mesh.eval()
        nVerts = opt_mesh.v_pos.shape[0]
        t_pos_idx = opt_mesh.t_pos_idx.detach().cpu().numpy() 

        # Find unique edges
        ix_i = []
        ix_j = []
        edge_verts = {}
        for tri in t_pos_idx:
            for (i0, i1) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                if (i1, i0) not in edge_verts.keys():
                    edge_verts[(i0, i1)] = True
                    ix_i += [i0]
                    ix_j += [i1]

        # Setup torch tensors
        ix_i = torch.tensor(ix_i, dtype=torch.int64, device='cuda')
        ix_j = torch.tensor(ix_j, dtype=torch.int64, device='cuda')

        # Gather edge vertex pairs
        x_i = opt_mesh.v_pos[ix_i, :]
        x_j = opt_mesh.v_pos[ix_j, :]

        # Compute edge length
        term = torch.sqrt((x_j - x_i)**2)

        # Compute avg edge length
        return (torch.sum(term) / len(x_i)).item()

######################################################################################
# Edge length regularizer 
######################################################################################
def edge_length_regularizer(mesh):
    class mesh_op_edge_length_regularizer:
        def __init__(self, mesh):
            self.mesh = mesh
            
            mesh = mesh.eval()
            nVerts = mesh.v_pos.shape[0]
            t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy() 

            # Find unique edges
            ix_i = []
            ix_j = []
            edge_verts = {}
            for tri in t_pos_idx:
                for (i0, i1) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                    if (i1, i0) not in edge_verts.keys():
                        edge_verts[(i0, i1)] = True
                        ix_i += [i0]
                        ix_j += [i1]

            # Setup torch tensors
            self.ix_i = torch.tensor(ix_i, dtype=torch.int64, device='cuda')
            self.ix_j = torch.tensor(ix_j, dtype=torch.int64, device='cuda')

        def eval(self, params={}):
            mesh = self.mesh.eval(params)

            # Gather edge vertex pairs
            x_i = mesh.v_pos[self.ix_i, :]
            x_j = mesh.v_pos[self.ix_j, :]

            # Compute edge length
            term = torch.sqrt((x_j - x_i)**2 + 1e-20)

            # Compute avg edge length
            return torch.var(term)

    return mesh_op_edge_length_regularizer(mesh)

######################################################################################
# Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
# https://mgarland.org/class/geom04/material/smoothing.pdf
######################################################################################
def laplace_regularizer_const(opt_mesh, base_mesh=None):
    class mesh_op_laplace_regularizer_const:
        def __init__(self, opt_mesh, base_mesh):
            self.inputs = [opt_mesh, base_mesh]

            opt_mesh = opt_mesh.eval()
            self.nVerts = opt_mesh.v_pos.shape[0]
            t_pos_idx = opt_mesh.t_pos_idx.detach().cpu().numpy() 

            # Build vertex neighbor rings
            vtx_n = [[] for _ in range(self.nVerts)]
            for tri in t_pos_idx:
                for (i0, i1) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                    vtx_n[i0].append(i1)

            # Collect index/weight pairs to compute each Laplacian vector for each vertex.
            # Similar notation to https://mgarland.org/class/geom04/material/smoothing.pdf
            ix_j, ix_i, w_ij = [], [], []
            for i in range(self.nVerts):
                m = len(vtx_n[i])
                ix_i += [i] * m
                ix_j += vtx_n[i]
                w_ij += [1.0 / m] * m

            # Setup torch tensors
            self.ix_i = torch.tensor(ix_i, dtype=torch.int64, device='cuda')
            self.ix_j = torch.tensor(ix_j, dtype=torch.int64, device='cuda')
            self.w_ij = torch.tensor(w_ij, dtype=torch.float32, device='cuda')[:, None]

        def eval(self, params={}):
            opt_mesh = self.inputs[0].eval(params)
            base_mesh = self.inputs[1].eval(params) if self.inputs[1] is not None else None

            # differences or absolute version (see paper)
            if base_mesh is not None:
                v_pos = opt_mesh.v_pos - base_mesh.v_pos
            else:
                v_pos = opt_mesh.v_pos

            # Gather edge vertex pairs
            x_i = v_pos[self.ix_i, :]
            x_j = v_pos[self.ix_j, :]

            # Compute Laplacian differences: (x_j - x_i) * w_ij
            term = (x_j - x_i) * self.w_ij

            # Sum everyhing
            term = util.segment_sum(term, self.ix_i)
            
            return torch.mean(term**2)
    
    return mesh_op_laplace_regularizer_const(opt_mesh, base_mesh)

######################################################################################
# Curvature based regularizer
######################################################################################
def face_normal_regularizer(opt_mesh):
    class mesh_op_face_normal_regularizer:
        def __init__(self, opt_mesh):
            self.input = opt_mesh

            imesh = opt_mesh.eval()
            self.nVerts = imesh.v_pos.shape[0]
            t_pos_idx = imesh.t_pos_idx.detach().cpu().numpy() 

            # Generate edge lists
            edge_tris = {}
            for tri_idx, tri in enumerate(t_pos_idx):
                for (i0, i1) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                    if (i1, i0) in edge_tris.keys():
                        edge_tris[(i1, i0)] += [tri_idx]
                    else:
                        edge_tris[(i0, i1)] = [tri_idx]

            # Get all good edges with 2 incident triangles
            shared_edge_idx = []
            for edge in edge_tris.values():
                if len(edge) == 2:
                    shared_edge_idx += [edge]
            self.edge_tri_idx = torch.tensor(shared_edge_idx, dtype=torch.int64, device='cuda')

        def eval(self, params={}):
            imesh = self.input.eval(params)

            # Compute face normals
            v0 = imesh.v_pos[imesh.t_pos_idx[:, 0], :]
            v1 = imesh.v_pos[imesh.t_pos_idx[:, 1], :]
            v2 = imesh.v_pos[imesh.t_pos_idx[:, 2], :]
            face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))

            # Fetch normals for both faces sharind an edge
            n0 = face_normals[self.edge_tri_idx[:, 0], :]
            n1 = face_normals[self.edge_tri_idx[:, 1], :]

            # Compute error metric based on normal difference
            term = torch.clamp(util.dot(n0, n1), min=-1.0, max=1.0)
            term = (1.0 - term) * 0.5

            return torch.mean(torch.abs(term))
    
    return mesh_op_face_normal_regularizer(opt_mesh)
