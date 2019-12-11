# Getting started with Docker

In order to replicate the results smoothly and avoid dependency errors (aka CUDA installation hell) you can use Docker combined with NVIDIA-Docker. Docker will install all the packages in an isolated environment.

Note: You will need an NVIDIA GPU and a Linux OS to use NVIDIA-Docker.
 
## Installing Docker
* [Docker](https://gist.github.com/enric1994/3b5c20ddb2b4033c4498b92a71d909da)
* [Docker-Compose](https://gist.github.com/enric1994/3b5c20ddb2b4033c4498b92a71d909da)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster)

## Downloading the required files
* Download the `frames` and `masks` folders from [here](https://drive.google.com/drive/folders/13aMItboZBxPnbjlOCbKLg7nxZgBWQt9P) and place them on the `demo` folder.

* Download the files `FlowNet2_checkpoint.pth.tar`, `imagenet_deepfill.pth` and `resnet101_movie.pth` from [here](https://drive.google.com/drive/folders/1Nh6eJsue2IkP_bsN02SRPvWzkIi6cNbE) and place them in `pretrained_models`.

```
├── demo
│   ├── frames
│   └── masks
├── pretrained_models
│   ├── FlowNet2_checkpoint.pth.tar
│   ├── imagenet_deepfill.pth
│   └── resnet101_movie.pth
```


## Usage
1. From the `docker` folder run: `docker-compose up -d` 

2. Access the conatiner: `docker exec -it inpainting bash` 

That will open a CLI on the Docker container. Now you can run the demo scripts, for example:

`python3 tools/video_inpaint.py --frame_dir ./demo/frames --MASK_ROOT ./demo/masks --img_size 512 832 --FlowNet2 --DFC --ResNet101 --Propagation`


Tested on Ubuntu 18.04 with a GTX 1060 GPU (drivers 410.104). Not working on higher architectures such as sm_75 (Turing), e.g. RTX 2080 Ti