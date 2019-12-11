# Deep Flow-Guided Video Inpainting
[CVPR 2019 Paper](https://arxiv.org/abs/1905.02884) | [Project Page](https://nbei.github.io/video-inpainting.html) | [YouTube](https://www.youtube.com/watch?v=LIJPUsrwx5E) | [BibeTex](#citation)

<img src="https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting/blob/master/gif/captain.gif" width="860"/>

## Install & Requirements
The code has been tested on pytorch=0.4.0 and python3.6. Please refer to `requirements.txt` for detailed information. 

Alternatively, you can run it with the provided [Docker image](docker/README.md).

**To Install python packages**
```
pip install -r requirements.txt
```
**To Install flownet2 modules**
```
bash install_scripts.sh
```
## Componets
There exist three components in this repo:
* Video Inpainting Tool: DFVI
* Extract Flow: FlowNet2(modified by [Nvidia official version](https://github.com/NVIDIA/flownet2-pytorch/tree/python36-PyTorch0.4))
* Image Inpainting(reimplemented from [Deepfillv1](https://github.com/JiahuiYu/generative_inpainting))

## Usage
* To use our video inpainting tool for object removing, we recommend that the frames should be put into `xxx/video_name/frames`
and the mask of each frame should be put into `xxx/video_name/masks`. 
And please download the resources of the demo and model weights from [here](https://drive.google.com/drive/folders/1a2FrHIQGExJTHXxSIibZOGMukNrypr_g?usp=sharing).
An example demo containing frames and masks has been put into the demo and running the following command will get the result:
```
python tools/video_inpaint.py --frame_dir ./demo/frames --MASK_ROOT ./demo/masks --img_size 512 832 --FlowNet2 --DFC --ResNet101 --Propagation 
```
<img src="https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting/blob/master/gif/flamingo.gif" width="850"/>

We provide the original model weight used in our movie demo which use ResNet101 as backbone and other related weights pls download from [here](https://drive.google.com/drive/folders/1a2FrHIQGExJTHXxSIibZOGMukNrypr_g?usp=sharing). 
Please refer to [tools](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting/tree/master/tools) for detailed use and training settings. 

* For fixed region inpainting, we provide the model weights of refined stages in DAVIS. Please download the lady-running resources [link](https://drive.google.com/drive/folders/1GHV1g1IkpGa2qhRnZE2Fv30RXrbHPH0O?usp=sharing) and 
model weights [link](https://drive.google.com/drive/folders/1zIamN-DzvknZLf5QAGCfvWs7a6qUqaaC?usp=sharing). The following command can help you to get the result:
```
CUDA_VISIBLE_DEVICES=0 python tools/video_inpaint.py --frame_dir ./demo/lady-running/frames \
--MASK_ROOT ./demo/lady-running/mask_bbox.png \
--img_size 448 896 --DFC --FlowNet2 --Propagation \
--PRETRAINED_MODEL_1 ./pretrained_models/resnet50_stage1.pth \
--PRETRAINED_MODEL_2 ./pretrained_models/DAVIS_model/davis_stage2.pth \
--PRETRAINED_MODEL_3 ./pretrained_models/DAVIS_model/davis_stage3.pth \
--MS --th_warp 3 --FIX_MASK
```
<img src="https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting/blob/master/gif/lady-running-res.gif" width="850"/>
You can just change the **th_warp** param for getting better results in your video. 

* To extract flow for videos:
```
python tools/infer_flownet2.py --frame_dir xxx/video_name/frames
```

* To use the Deepfillv1-Pytorch model for image inpainting,
```
python tools/frame_inpaint.py --test_img xxx.png --test_mask xxx.png --image_shape 512 512
```

## Update
* More results can be found and downloaded [here](https://www.dropbox.com/sh/jxcl4he5bgsmk7t/AADxnfqHj-PGcjxd02Bil56ya?dl=0). 
* **Support for PyTorch>1.0:** Sorry for the late update and the pre-release verison for supporting PyTorch>1.0 has been integrated into our new [v1.1 branch](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting/tree/v1.1).

* The frames and masks of our movie demo have been put into [Google Drive](https://drive.google.com/drive/folders/1z2n1LzVY8gjvy7ezF_tuuMgVouR_pFcz?usp=sharing).
* The weights of DAVIS's refined stages have been released and you can download from [here](https://drive.google.com/drive/folders/1zIamN-DzvknZLf5QAGCfvWs7a6qUqaaC?usp=sharing).
Please refer to [Usage](#Usage) for using the Multi-Scale models.
## FAQ
* Errors when running install_scripts.sh
if you meet some problem about gcc when compiling, pls check if the following commands will help:
```
export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"
```

## Citation
```
@InProceedings{Xu_2019_CVPR,
author = {Xu, Rui and Li, Xiaoxiao and Zhou, Bolei and Loy, Chen Change},
title = {Deep Flow-Guided Video Inpainting},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
