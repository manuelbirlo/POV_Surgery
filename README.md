# GrabNet for Pose Generation

<img src="assets/images/GrabNet_Voluson_grasps.gif" width="250" height="250" alt="Description"> <img src="assets/images/GrabNet_single_Voluson_grasp1.gif" width="250" height="250" alt="Description"> <img src="assets/images/GrabNet_single_Voluson_grasp2.gif" width="250" height="250" alt="Description">

This repo was forked from the already existing [POV-Surgery repo](https://github.com/BatFaceWayne/POV_Surgery) by Rui Wang et al. 

# Installation 

# Optional: Use a Docker container

This repo is laid out for a Linux-based operating system. However, if you would like to use this repo with your Windows machine, a Docker containe is a suitable solution. Please follows our instructions here: [Docker container setup](./Docker/README.md)
Please note: It is also possible to use a Docker container on Mac, but we have not tested this repo on a Mac system with an Apple Silicon chip. We are only referring to the grab generation part of the forked POV-Surgery repo, which generates grab images using GrabNet, but the entire POV-Surgery pipeline renders grab images using Blender.
Newer Apple computers using the Silicon chip may be incompatible with Blender's OpenGL-related functionality.

# Cloning this repo (download the source code)
```sh
git clone https://github.com/manuelbirlo/POV_Surgery (TODO: adjust url as soon as repo has been renamed!!!)
```




