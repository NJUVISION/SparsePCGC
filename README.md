# Sparse Tensor-based Multiscale Representation for Point Cloud Geometry Compression

This study develops a unified Point Cloud Geometry (PCG) compression method through Sparse Tensor Processing (STP) based multiscale representation of voxelized PCG, dubbed as the SparsePCGC. Applying the STP reduces the complexity significantly because it only  performs the convolutions centered at Most-Probable Positively-Occupied Voxels (MP-POV). And the multiscale representation facilitates us to compress scale-wise MP-POVs progressively. The overall compression efficiency highly depends on the approximation accuracy of occupancy probability of each MP-POV. Thus, we design the Sparse Convolution based Neural Networks (SparseCNN) consisting of sparse convolutions and voxel re-sampling to extensively exploit priors. We then develop the SparseCNN based Occupancy Probability Approximation (SOPA) model to estimate the probability in a single-stage manner only using the cross-scale prior or in multi-stage by step-wisely utilizing autoregressive neighbors. Besides, we also suggest the SparseCNN based Local Neighborhood Embedding (SLNE) to characterize the local spatial variations as the feature attribute to improve the SOPA. Our unified approach shows the state-of-art performance in both lossless and lossy  compression modes across a variety of datasets including the dense PCGs (8iVFB, Owlii) and the sparse LiDAR PCGs (KITTI, Ford) when compared with the MPEG G-PCC and other popular learning-based compression schemes. Furthermore, the proposed method presents lightweight complexity due to point-wise computation, and tiny storage desire because of model sharing across all scales.


## News

- 2021.11.23 We have posted the manuscript on arxiv (https://arxiv.org/abs/2111.10633).
- 2022.01.13 We participate in MPEG AI-3DGC (EE. 13.54). 

**I am very sorry that now we can not publish the source code for publicly accessible,  due to the potential contradiction with our collaborators' security policy. But the necessary testing dataset and results are accessible. 
The code can be open to the cross-checkers in standardization organizations like MPEG, and other authorized experts.**

## Requirments
- pytorch1.8
- MinkowskiEngine 0.5 
- **[Testdata]**: https://box.nju.edu.cn/f/8f6774a4ba53482a9435/
- extension(tmc3,pc_error_d): https://box.nju.edu.cn/f/7fba08c849b84f60a4f1/
- **[Results]**: https://box.nju.edu.cn/f/c318817312df4d52bb25/
- Training Dataset: TODO



## Authors
These files are provided by Nanjing University  [Vision Lab](https://vision.nju.edu.cn/). And thanks for the help of Prof. Dandan Ding from Hangzhou Normal University, Prof. Zhu Li from University of Missouri at Kansas. Please contact us (mazhan@nju.edu.cn and wangjq@smail.nju.edu.cn) if you have any questions.

