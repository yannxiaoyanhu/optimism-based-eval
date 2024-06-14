# optimisim-based-eval: An Optimism-based Approach to Online Evaluation of Generative Models -- Pytorch Implementation

Authors: [Xiaoyan Hu](https://yannxiaoyanhu.github.io), [Ho-fung Leung](http://www.cse.cuhk.edu.hk/~lhf/), [Farzan Farnia](https://www.cse.cuhk.edu.hk/~farnia/Home.html) [[Paper](https://arxiv.org/abs/2406.07451)]

## Reproducibility

In this repository, we use ImageNet as an example:

1. Download the pretrained weights for the generative models from the official repository of [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) and its Hugging Face Hub page: [https://huggingface.co/Mingguksky/PyTorch-StudioGAN/tree/main](https://huggingface.co/Mingguksky/PyTorch-StudioGAN/tree/main). Save them under the path **./gen_models/pretrain_weights/**. (Links are provided in the './gen_models/get_models.py' file.) 
2. For FID-based evaluation, download the weights for the embedding pretrained models and save them under the path **./feat_models/weights**. 
3. Download the statistics via [[Google Drive](https://drive.google.com/drive/folders/1lrNfbp9MjDAMKcTOtVNPcAwAZqlMXWVS?usp=share_link)] and save under the path **./stats/**. Specifically, files named 'imagenet_fid_stat_{}.npz' save the mean vectors and the covariance matrices for the real images under different embedding models, and files named 'imagenet_FID_Eval_{}.npy' or 'imagenet_IS_Eval.npy' save the FID/Inception scores for the generators based on 50k generated images.
4. For FID-based evaluation, run [FID-eval.py](https://github.com/yannxiaoyanhu/optimism-based-eval/blob/main/FID-eval.py). To reproduce the results, parameter *ths_ratio* at line 24, which is used to truncate the covariance matrix for the generated images, should be 0.005 for InceptionV3.Net and 0.05 for both dinov2 and CLIP, respectively. For IS-based evaluation, run [IS-eval.py](https://github.com/yannxiaoyanhu/optimism-based-eval/blob/main/IS-eval.py).

### Results for FID-InceptionV3.Net
![FID-InceptionV3.Net](https://github.com/yannxiaoyanhu/optimism-based-eval/blob/main/results/FID-InceptionV3.Net.png)

### Results for FID-dinov2
![FID-dinov2](https://github.com/yannxiaoyanhu/optimism-based-eval/blob/main/results/FID-dinov2.png)

### Results for FID-CLIP
![FID-CLIP](https://github.com/yannxiaoyanhu/optimism-based-eval/blob/main/results/FID-CLIP.png)

### Results for IS
![IS](https://github.com/yannxiaoyanhu/optimism-based-eval/blob/main/results/IS.png)

## Acknowledgements

The authors would like to acknowledge the following repositories, which have been utilized in the implementation and testing of the proposed FID-UCB and IS-UCB:

1. [[MIT License](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/LICENSE), [Nvidia Source Code License](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/LICENSE-NVIDIA)] StudioGAN: [https://github.com/POSTECH-CVLab/PyTorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN).
2. [[Apache License 2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE)] DINOv2: [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).
3. [[License](https://github.com/mlfoundations/open_clip/blob/main/LICENSE)] OpenCLIP: [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip).


## License

The optimisim-based-eval library is released under the MIT License. See [LICENSE.txt](https://github.com/yannxiaoyanhu/optimism-based-eval/blob/main/LICENSE.txt) for additional details. Part of the code are avaiiable under distinct license terms. Specifically, the code based on the implementation in the official repository of OpenCLIP is under the [LICENSE-OpenCLIP.txt](https://github.com/yannxiaoyanhu/optimism-based-eval/blob/main/LICENSE-OpenCLIP), the code based on the implementation in the official repository of dinov2 is under the [Apache License 2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE), and the implementation of StyleGAN3 are licensed under [NVIDIA source code license](https://github.com/yannxiaoyanhu/optimism-based-eval/blob/main/LICENSE-NVIDIA.txt).

## Citation
```
@misc{hu2024optimismbased,
      title={An Optimism-based Approach to Online Evaluation of Generative Models}, 
      author={Xiaoyan Hu and Ho-fung Leung and Farzan Farnia},
      year={2024},
      eprint={2406.07451},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
