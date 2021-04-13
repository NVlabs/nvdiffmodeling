// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include "common.h"

enum TonemapperType
{
    TONEMAPPER_NONE = 0,
    TONEMAPPER_LOG_SRGB = 1
};

enum LossType
{
    LOSS_L1 = 0,
    LOSS_MSE = 1,
    LOSS_RELMSE = 2,
    LOSS_SMAPE = 3
};

struct LossKernelParams
{
    Tensor          img;
    Tensor          target;
    Tensor          out;
    dim3            gridSize;
    TonemapperType  tonemapper;
    LossType        loss;
};
