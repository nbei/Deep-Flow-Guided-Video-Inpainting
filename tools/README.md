# Tools

## Args in Video Inpainting
* enlarge_kernel: In object removing, we recommend the tight masks fo the object should be enlarged.
* th_warp: threshold in the propagation process. Modifying the th_warp will lead to different results.
If you do not satisfy the result, pls try to modify this argument.

## Training Guidance
To get the better results, you can use the stacked multi-scale(MS) DFC-Net.
When training third stage, you'd better use the second stage's weights as initialization.

For fixed-region(mid-bbox) inpainting, we give further guidance for your reference.

1. For initial stage(stage 1). Firstly, you should generate the data list which is much more similar with the testing list.
However, you should remove the output dir in the data list which is the second to last item. And then you can use the following command:
```
CUDA_VISIBLE_DEVICES=xxxxx python tools/train_initial.py --model_name stage1 --FIX_MASK --MASK_MODE mid-bbox --TRAIN_LIST path_to_train_datalist --DATA_ROOT path_to_input_flow --INITIAL_HOLE
```
2. For Refine stage(stage2 and stage3). For convenient and quick loading data, we first extract the flow by using stage1 network and then use 
these results as the input of the next stage.
```
CUDA_VISIBLE_DEVICES=xxxxx python tools/train_refine.py --model_name stagex --FIX_MASK --MASK_MODE mid-bbox --TRAIN_LIST path_to_train_datalist --DATA_ROOT path_to_input_flow --GT_FLOW_ROOT path_to_gt_flow
```
Some tips in training refinement stage:
* The image shape, result shape and (MASK_HEIGHT, MASK_WIDTH) should be set according to your need and MASK_HEIGHT = img_height // 4, MASK_WIDTH = img_width // 4. 
In our experiment, we set `--IMAGE_SHAPE 320 600 --RES_SAHPE 320 600` in stage 2 and `--IMAGE_SHAPE 480 840 --RES_SAHPE 480 840` in stage 3.
* We highly recommend you to keep the batch_size of 32. However, if you do not have enough gpu memory, you can decrease the learning rate and the weight of HFEM ('--LAMBDA_HARD').
* If you cannot train the following stage with that resolution, you can pre-train the refinement stage with low resolution and large batch_size and then finetune in the large resolution.
 
## Image Inpainting Model
The contextual attention module can be substituted with better implementation such as no-local module.
We provide this version just for testing DeepFillv1 with the same result as the original Tensorflow version.

## Pytorch-FlowNet2: Extract Flow
In the original experiments of our paper, we use the Caffe version of FlowNet2.
However, it may cause inefficiency for you to compile Caffe. Thus, we provide a
modified Pytorch-FlowNet2 for extracting flow conveniently.

However, the image size of the input image must be **divided by 64**.
Otherwise, you will get errors when running this code.

