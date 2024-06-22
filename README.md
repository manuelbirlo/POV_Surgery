# Grasp Generation via a generative model: A 'POV-Surgery' Derivative

This repo was forked from the already existing [POV-Surgery repo](https://github.com/BatFaceWayne/POV_Surgery) by Rui Wang et al. 

# Installation 

## Docker Container

The original installation instructions are laid out for a Unix-like OS, for example the Linux-based Ubuntu OS. 
If you don't have a Linux OS and would like to install this repo on a Windows computer, a solution is to use a Docker container. 
A possible docker container installation could look as follows: 

### Docker Desktop Installation and Setup under Windows

1. **Install Docker Desktop** from the [official Docker website](https://www.docker.com/products/docker-desktop/) and launch the application.

2. **Verify Docker installation** by running the following command in a Command Prompt, Terminal, or PowerShell:
    ```sh
    docker --version
    ```

3. **Get the Dockerfile**: In your browser within this repo website, click on 'Dockerfile' in this repo's root directory (same directory as this README file), navigate to the top right corner of the displayed Dockerfile content, and click on the Download symbol.

4. **Place the Dockerfile into your desired project directory**:
    ```sh
    cd your\project\directory
    ```

5. **Within your project directory, build your Docker image via the `docker build` command**:
    ```sh
    docker build -t <your-docker-image-name> .
   ```sh
    Replace `<your-docker-image-name>` with a suitable name, for example, `nvidia_cuda_118`.

6. **Create your docker container based on the newly created docker image `<your-docker-image-name>`
7. ```sh
   docker run --name `<your-docker-container-name>` --gpus all --net=host --env="DISPLAY" -it `<your-docker-image-name>`
   ```



