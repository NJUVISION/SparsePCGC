# Sparse Tensor-based Multiscale Representation for Point Cloud Geometry Compression


Abstract — This study develops a unified Point Cloud Geometry (PCG) compression method through the processing of multiscale sparse tensor-based voxelized PCG. We call this compression method SparsePCGC. The proposed SparsePCGC is a low complexity solution because it only performs the convolutions on sparsely-distributed Most-Probable Positively-Occupied Voxels (MP-POV). The multiscale representation also allows us to compress scale-wise MP-POVs by exploiting cross-scale and same-scale correlations extensively and flexibly. The overall compression efficiency highly depends on the accuracy of estimated occupancy probability for each MP-POV. Thus, we first design the Sparse Convolution-based Neural Network (SparseCNN) which stacks sparse convolutions and voxel sampling to best characterize and embed spatial correlations. We then develop the SparseCNN-based Occupancy Probability Approximation (SOPA) model to estimate the occupancy probability either in a single-stage manner only using the cross-scale correlation, or in a multi-stage manner by exploiting stage-wise correlation among same-scale neighbors. Besides, we also suggest the SparseCNN based Local Neighborhood Embedding (SLNE) to aggregate local variations as spatial priors in feature attribute to improve the SOPA. Our unified approach not only shows state-of-the-art performance in both lossless and lossy compression modes across a variety of datasets including the dense object PCGs (8iVFB, Owlii, MUVB) and sparse LiDAR PCGs (KITTI, Ford) when compared with standardized MPEG G-PCC and other prevalent learning-based schemes, but also has low complexity which is attractive to practical applications.


## News
- 2024.12.06 We released the SparsePCGC source code, which also serves as a preview of [Unicorn](https://njuvision.github.io/Unicorn/).
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
- **[Pretrained Models]**: https://box.nju.edu.cn/f/3bf11a6700fd4466be5c/
- Training Dataset: ShapeNet (https://box.nju.edu.cn/f/5c5e4568bb614f54b813/);  KITTI (Coming soon)


## Usage

### Testing

The following example commands are provided to illustrate the general testing process.

For dense point clouds:

```bash
# dense lossless
python test_ours_dense.py --mode='lossless' \
--ckptdir='../ckpts/dense/epoch_last.pth' \
--filedir='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' \
--prefix='ours_8i'
```

```bash
# dense lossy
python test_ours_dense.py --mode='lossy' \
--ckptdir='../ckpts/dense/epoch_last.pth' \
--ckptdir_sr='../ckpts/dense_1stage/epoch_last.pth' \
--ckptdir_ae='../ckpts/dense_slne/epoch_last.pth' \
--filedir='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' \
--psnr_resolution=1023 --prefix='ours_8i_lossy'
```

For sparse LiDAR point clouds:

```bash
# sparse lossless
python test_ours_sparse.py --mode='lossless' \
--ckptdir_low='../ckpts/sparse_low/epoch_last.pth' \
--ckptdir_high='../ckpts/sparse_high/epoch_last.pth' \
--filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/' \
--voxel_size=1 --prefix='ours_kitti_q1mm'
```

```bash
# sparse lossy
python test_ours_sparse.py --mode='lossy_gpcc' \
--ckptdir_low='../ckpts/sparse_low/epoch_last.pth' \
--ckptdir_high='../ckpts/sparse_high/epoch_last.pth' \
--filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/' \
--voxel_size=1 --prefix='ours_lossy_kitti'
```

```bash
# sparse lossy w/ offset
python test_ours_sparse.py --mode='lossy_gpcc' \
--ckptdir_low='../ckpts/sparse_low/epoch_last.pth' \
--ckptdir_high='../ckpts/sparse_high/epoch_last.pth' \
--ckptdir_offset='../ckpts/sparse_offset/epoch_last.pth' \
--offset --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/' \
--voxel_size=1 --prefix='ours_lossy_kitti_offset'
```

Please refer to `./test/README_test.md` for other testing examples, including commands for testing other datasets such as [Owlii](https://mpeg-pcc.org/index.php/pcc-content-database/owlii-dynamic-human-textured-mesh-sequence-dataset/) and [Ford](https://robots.engin.umich.edu/SoftwareData/InfoFord). Detailed testing results are available in the `./results` directory.

### Training

We provide training script in the `./train` directory. Please refer to `./train/README_train.md` for training examples.

## Authors
These files are provided by Nanjing University  [Vision Lab](https://vision.nju.edu.cn/). Thanks for the help of Prof. Dandan Ding from Hangzhou Normal University, Prof. Zhu Li from University of Missouri at Kansas. Please contact us (wangjq@smail.nju.edu.cn and mazhan@nju.edu.cn) if you have any questions.

