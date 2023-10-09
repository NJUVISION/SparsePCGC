# Sparse Tensor-based Multiscale Representation for Point Cloud Geometry Compression


Abstract — This study develops a unified Point Cloud Geometry (PCG) compression method through the processing of multiscale sparse tensor-based voxelized PCG. We call this compression method SparsePCGC. The proposed SparsePCGC is a low complexity solution because it only performs the convolutions on sparsely-distributed Most-Probable Positively-Occupied Voxels (MP-POV). The multiscale representation also allows us to compress scale-wise MP-POVs by exploiting cross-scale and same-scale correlations extensively and flexibly. The overall compression efficiency highly depends on the accuracy of estimated occupancy probability for each MP-POV. Thus, we first design the Sparse Convolution-based Neural Network (SparseCNN) which stacks sparse convolutions and voxel sampling to best characterize and embed spatial correlations. We then develop the SparseCNN-based Occupancy Probability Approximation (SOPA) model to estimate the occupancy probability either in a single-stage manner only using the cross-scale correlation, or in a multi-stage manner by exploiting stage-wise correlation among same-scale neighbors. Besides, we also suggest the SparseCNN based Local Neighborhood Embedding (SLNE) to aggregate local variations as spatial priors in feature attribute to improve the SOPA. Our unified approach not only shows state-of-the-art performance in both lossless and lossy compression modes across a variety of datasets including the dense object PCGs (8iVFB, Owlii, MUVB) and sparse LiDAR PCGs (KITTI, Ford) when compared with standardized MPEG G-PCC and other prevalent learning-based schemes, but also has low complexity which is attractive to practical applications.



## News
- 2022.11.25 The paper was accpeted by TPAMI. (J. Wang, D. Ding, Z. Li, X. Feng, C. Cao and Z. Ma, "Sparse Tensor-Based Multiscale Representation for Point Cloud Geometry Compression," in IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022, doi: 10.1109/TPAMI.2022.3225816.)
- 2022.09.09 Upload supplementary material, which includes comparison details. see `Supplementary_Material.pdf`
- 2022.06.16 Source codes will be released soon to the public after the approval from the funding agency. Now We make the testing results, testing conditions, pretrained models, and other relevant materials publicly accessible.
- 2022.06.16 We simplify the implementation and reduce the computational complexity significantly. (e.g., almost 6∼8×). At the same time, we slightly adjust model parameters and achieve better performance on sparse LiDAR point clouds.
- 2022.01.13 We participate in MPEG AI-3DGC. 
- 2021.11.23 We have posted the manuscript on arxiv (https://arxiv.org/abs/2111.10633).


## Requirments
- pytorch **1.10**
- MinkowskiEngine 0.5 
- **docker pull jianqiang1995/pytorch:1.10.0-cuda11.1-cudnn8-devel**
- **[Testdata]**: https://box.nju.edu.cn/d/80f52336bbf04ab1a9a6/
- **[Results]**: https://box.nju.edu.cn/d/b74e418c1dd94ddb847a/
- **[Pretrained Models]**: https://box.nju.edu.cn/f/3bf11a6700fd4466be5c/
- Training Dataset: ShapeNet (https://box.nju.edu.cn/f/5c5e4568bb614f54b813/);  KITTI



## Authors
These files are provided by Nanjing University  [Vision Lab](https://vision.nju.edu.cn/). Thanks for the help of Prof. Dandan Ding from Hangzhou Normal University, Prof. Zhu Li from University of Missouri at Kansas. Please contact us (wangjq@smail.nju.edu.cn and mazhan@nju.edu.cn) if you have any questions.

