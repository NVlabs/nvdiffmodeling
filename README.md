# nvdiffmodeling 

![Teaser image](https://research.nvidia.com/sites/default/files/publications/teaser_7.png "Teaser image")

Differentiable rasterization applied to 3D model simplification tasks, as described in the paper:

**Appearance-Driven Automatic 3D Model Simplification**<br>
Jon Hasselgren, Jacob Munkberg, Jaakko Lehtinen, Miika Aittala and Samuli Laine<br>
https://research.nvidia.com/publication/2021-04_Appearance-Driven-Automatic-3D <br>
https://arxiv.org/abs/2104.03989 <br>

# License

Copyright © 2021, NVIDIA Corporation. All rights reserved.

This work is made available under the [Nvidia Source Code License](https://github.com/NVlabs/nvdiffmodeling/blob/main/LICENSE.txt).

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)<br>

# Citation

```
@inproceedings{Hasselgren2021,
  title     = {Appearance-Driven Automatic 3D Model Simplification},
  author    = {Jon Hasselgren and Jacob Munkberg and Jaakko Lehtinen and Miika Aittala and Samuli Laine},
  booktitle = {Eurographics Symposium on Rendering},
  year      = {2021}
}
```

# Installation

Requirements:
 - [Microsoft Visual Studio](https://visualstudio.microsoft.com/) 2019+ with Microsoft Visual C++ installed
 - [Cuda](https://developer.nvidia.com/cuda-toolkit) 10.2+
 - [Pytorch](https://pytorch.org/) 1.6+

Tested in Anaconda3 with Python 3.6 and PyTorch 1.8.

## One time setup (Windows)

1. Install [Microsoft Visual Studio](https://visualstudio.microsoft.com/) 2019+ with Microsoft Visual C++. 
2. Install [Cuda](https://developer.nvidia.com/cuda-toolkit) 10.2 or above. **Note:** Install CUDA toolkit from https://developer.nvidia.com/cuda-toolkit (not through anaconda)
3. Install the appropriate version of PyTorch compatible with the installed Cuda toolkit.
Below is an example with Cuda 11.1

```
conda create -n dmodel python=3.6
activate dmodel
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
conda install imageio
pip install PyOpenGL glfw
```

4. Install [nvdiffrast](https://github.com/NVlabs/nvdiffrast) in the `dmodel` conda env. Follow the [installation instructions](https://nvlabs.github.io/nvdiffrast/#windows).

### Every new command prompt
`activate dmodel`

Examples
========

Sphere to cow example:
```
python train.py --config configs/spot.json
```
The results will be stored in the `out` folder.
The [Spot](http://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/index.html#spot) model was 
created and released into the public domain by [Keenan Crane](http://www.cs.cmu.edu/~kmcrane/index.html).

[Additional assets can be downloaded here](https://github.com/NVlabs/nvdiffmodeling/releases/download/v1.0/assets.zip) [205MB].
Unzip and place the subfolders in the project `data` folder, e.g., `data\skull`.
All assets are copyright of their respective authors, see included license files for further details.

Included examples
- `skull.json` - Joint normal map and shape optimization on a skull
- `ewer.json`  - Ewer model from a reduced mesh as initial guess
- `gardenina.json` - Aggregate geometry example
- `hibiscus.json` - Aggregate geometry example
- `figure_brushed_gold_64.json` - LOD example, trained against a supersampled reference
- `figure_displacement.json` - Joint shape, normal map, and displacement map example

The json files that end in `_paper.json` are configs with the settings used
for the results in the paper. They take longer and require a GPU with sufficient memory.

Server usage (through Docker)
=============================

- Build docker image (run the command from the code root folder).
`docker build -f docker/Dockerfile -t diffmod:v1 .`
Requires a driver that supports Cuda 10.1 or newer.

- Start an interactive docker container:
`docker run --gpus device=0 -it --rm -v /raid:/raid -it diffmod:v1 bash`

- Detached docker:
`docker run --gpus device=1 -d -v /raid:/raid -w=[path to the code] diffmod:v1 python train.py --config configs/spot.json`
