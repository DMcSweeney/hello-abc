# Hello-ABC

Repository for setting up the ABC-toolkit.

## Requirements
- [Docker](https://www.docker.com/get-started/)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

On Linux distributions, you will need to add the current user to the `docker` group: `sudo usermod -a -G docker USERNAME`

## Getting started
1. Make a copy of `.env-default` and name it `.env`. Update parameters if needed. For testing, default values should work, just create the following directories: `./data/inputs` & `./data/outputs`.
    - `INPUT_DIR` is mounted to `/data/inputs` in all the containers. Your data should be (anywhere) in `INPUT_DIR`.
    - `OUTPUT_DIR` will contain all container outputs.

2. Build the base image: `./build_base_image.sh`


3. Run `docker compose up`. You should see output similar to: ![hello-abc-output](images/hello-abc-output.png "Hello ABC output")



