// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include "common.h"

struct PrepareShadingNormalKernelParams
{
    Tensor  pos;
    Tensor  view_pos;
    Tensor  perturbed_nrm;
    Tensor  smooth_nrm;
    Tensor  smooth_tng;
    Tensor  geom_nrm;
    Tensor  out;
    dim3    gridSize;
    bool    two_sided_shading, opengl;
};
