# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:21.02-py3
FROM $BASE_IMAGE

RUN apt-get update -yq --fix-missing \
 && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV PATH /tmp/usd/bin:$PATH
ENV PYTHONPATH /tmp/usd/lib/python:$PYTHONPATH

# install nvdiffrast
RUN curl -L  https://github.com/NVlabs/nvdiffrast/archive/main.zip > /tmp/nvdiffrast.zip
RUN mkdir -p /tmp/nvdiffrast && \
    cd /tmp/nvdiffrast && unzip /tmp/nvdiffrast.zip
RUN cd /tmp/nvdiffrast/nvdiffrast-main && pip install .

# additional libraries
RUN pip install imageio imageio-ffmpeg


