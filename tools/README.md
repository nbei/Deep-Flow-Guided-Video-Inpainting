# Tools

## Args in Video Inpainting
* enlarge_kernel: In object removing, we recommend the tight masks fo the object should be enlarged.
* th_warp: threshold in the propagation process. Modifying the th_warp will lead to different results.
If you do not satisfy the result, pls try to modify this argument.

## Training the Refined Stage
To get the better results, you can use the stacked multi-scale(MS) DFC-Net.
When training third stage, you'd better use the second stage's weights as initialization.

## Image Inpainting Model
The contextual attention module can be substituted with better implementation such as no-local module.
We provide this version just for testing DeepFillv1 with the same result as the original Tensorflow version.

## Pytorch-FlowNet2: Extract Flow
In the original experiments of our paper, we use the Caffe version of FlowNet2.
However, it may cause inefficiency for you to compile Caffe. Thus, we provide a
modified Pytorch-FlowNet2 for extracting flow conveniently.

However, the image size of the input image must be **divided by 64**.
Otherwise, you will get errors when running this code.

