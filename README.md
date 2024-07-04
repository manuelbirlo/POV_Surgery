# GrabNet for Ultrasound Probe Pose Generation

<img src="assets/images/GrabNet_Voluson_grasps.gif" width="250" height="250" alt="Description"> <img src="assets/images/GrabNet_single_Voluson_grasp1.gif" width="250" height="250" alt="Description"> <img src="assets/images/GrabNet_single_Voluson_grasp2.gif" width="250" height="250" alt="Description">

This repo was forked from the already existing [POV-Surgery repo](https://github.com/BatFaceWayne/POV_Surgery) by Rui Wang et al. 

# Installation 

# Optional: Use a Docker container

This repo is laid out for a Linux-based operating system. However, if you would like to use this repo with your Windows machine, a Docker containe is a suitable solution. Please follows our instructions here: [Docker container setup](./Docker/README.md)
Please note: It is also possible to use a Docker container on Mac, but we have not tested this repo on a Mac system with an Apple Silicon chip. We are only referring to the grab generation part of the forked POV-Surgery repo, which generates grab images using GrabNet, but the entire POV-Surgery pipeline renders grab images using Blender.
Newer Apple computers using the Silicon chip may be incompatible with Blender's OpenGL-related functionality.

# Cloning this repo (download the source code)
```sh
git clone https://github.com/manuelbirlo/US_GrabNet_grasp_generation.git
```

## Project structure
Please register yourself at [SMPL-X](https://smpl-x.is.tue.mpg.de/login.php) and [MANO](https://mano.is.tue.mpg.de/login.php) to use their dependencies. Please read and accept their liscenses to use SMPL-X and MANO models. There are different versions of manopth. We have included the implementation of [mano](https://github.com/otaheri/MANO) in our repo already.

Then please download the data.zip from the google drive folder of the original POV-Surgery project ([POV-Surgery](https://drive.google.com/drive/folders/1nSDig2cEHscCPgG10-VcSW3Q1zKge4tP?usp=drive_link)), unzip it and put in the US_GrabNet_grasp_generation folder. 
The final repo structure should look like this:

```bash
    US_GrabNet_grasp_generation
    ├── data
    │    │
    │    ├── sim_room
    │          └── room_sim.obj
    │          └── room_sim.obj.mtl
    │          └── textured_output.jpg
    │    │
    │    └── bodymodel
    │          │
    │          └── smplx_to_smpl.pkl
    │          └── ...
    │          └── mano
    │                └── MANO_RIGHT.pkl
    │          └── body_models
    │                └── smpl
    │                └── smplx
    ├── grasp_generation
    │         │
    │         └── ...
    │         └── grabnet
    │                └── ...
    │                └── tests
    │                      └── ...
    │                      └── **grab_new_tools.py**
    ├── grasp_refinement
    ├── pose_fusion
    ├── pre_rendering
    ├── blender_rendering
    ├── HandOccNet_ft
    └── vis_data

```
The focus of this repo is the use of the grab_new_tools.py (shown and highlighted in the file hierarchy above).

## Please note: 
We only consider the “grab_generation” part in our method, with optional “grab_refinement” and “pose_fusion” following.
We do not consider the Blender-related rendering steps “pre_rendering” and “blender_rendering” as well as the “HandOccNet_ft” and “vis_data” steps for neural network evaluation and result visualization. After the “pose_generation” step, we proceed with our Blender-based “grab_rendering” method [HUP-3D_renderer](https://github.com/manuelbirlo/HUP-3D_renderer). However, we encourage users to try out the “grab_rendering” concept of POV-Surgery and contribute new methods in this area.

## Recommendation: Using a Conda Environment
We recommend create a python 3.8 environment with conda. Install [pytorch](https://pytorch.org) and [torchvision](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjR4K2m8NmBAxVNSfEDHeMhCNAQFnoECBgQAQ&url=https%3A%2F%2Fpytorch.org%2Fvision%2F&usg=AOvVaw1cAB7MRIgRgtMiD3UKEL-9&opi=89978449) that suits you operation system. For example, if you are using cuda 11.8 version, you could use:

```Shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Then you should install [pytorch3d](https://github.com/facebookresearch/pytorch3d/tree/main) that suits your python and cuda version. An example could be found here: 

```Shell
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```
Then install the dependencies to finish the environment set up following the requiremesnts.sh. 
```Shell
sh requirements.sh
```

If you encounter a problem during the dependencies installation with 'sh requirements.sh' that says "fatal error: boost/config.hpp: "No such file or directory", you have to install the boost libraties with: 

```Shell
(sudo) apt-get install libboost-all-dev
```


# Generate grasps using GrabNet

Follow the instructions here: [How to setup grasp generation using GrabNet](grasp_generation/README.md)



