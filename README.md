# Sparse Tensor-based Multiscale Representation for Point Cloud Geometry Compression

This study develops a unified Point Cloud Geometry (PCG) compression method through Sparse Tensor Processing (STP) based multiscale representation of voxelized PCG, dubbed as the SparsePCGC. Applying the STP reduces the complexity significantly because it only  performs the convolutions centered at Most-Probable Positively-Occupied Voxels (MP-POV). And the multiscale representation facilitates us to compress scale-wise MP-POVs progressively. The overall compression efficiency highly depends on the approximation accuracy of occupancy probability of each MP-POV. Thus, we design the Sparse Convolution based Neural Networks (SparseCNN) consisting of sparse convolutions and voxel re-sampling to extensively exploit priors. We then develop the SparseCNN based Occupancy Probability Approximation (SOPA) model to estimate the probability in a single-stage manner only using the cross-scale prior or in multi-stage by step-wisely utilizing autoregressive neighbors. Besides, we also suggest the SparseCNN based Local Neighborhood Embedding (SLNE) to characterize the local spatial variations as the feature attribute to improve the SOPA. Our unified approach shows the state-of-art performance in both lossless and lossy  compression modes across a variety of datasets including the dense PCGs (8iVFB, Owlii) and the sparse LiDAR PCGs (KITTI, Ford) when compared with the MPEG G-PCC and other popular learning-based compression schemes. Furthermore, the proposed method presents lightweight complexity due to point-wise computation, and tiny storage desire because of model sharing across all scales. We make all materials publicly accessible at https://github.com/NJUVISION/SparsePCGC for reproducible research.

## News

- 2021.11.23 We have published the paper on arixiv.

## Requirments
- pytorch1.8
- MinkowskiEngine 0.5 
- Pretrained Models: https://box.nju.edu.cn/f/46d9206c6565471fb256/
- Testdata: https://box.nju.edu.cn/f/e7a4578decf24cfa8e09/
- extension(tmc3): https://box.nju.edu.cn/f/b17a3932ac0843c8ad46/
-  extension(pc_error_d): https://box.nju.edu.cn/f/5e66ad4057144438b5a4/
- Results: (TODO)
- Training Dataset: (TODO)

## Usage

### Testing
Please download the pretrained models, testdata, tmc3, pc_error_d mentioned above first.
```shell
sudo chmod 777 ./*
python test.py --data='8i' --mode='lossless'
python test.py --data='8i' --mode='lossy'
python test.py --data='ford' --mode='lossless'
python test.py --data='ford' --mode='lossless' --voxel_size=20
python test.py --data='ford' --mode='lossy'
```
The testing rusults of 8iVFB can be found in `./results`

### Training
```shell
python train.py --stage=1 --scale=4 --dataset '../dataset/ShapeNet/pc_vox8/' --dataset_test '../dataset/testdata/8iVFB/' --batch_size=4 --init_ckpt='ckpts/shapenet/1stage.pth'

```

## Authors
These files are provided by Nanjing University  [Vision Lab](https://vision.nju.edu.cn/). And thanks for the help from Prof. Dandan Ding from Hangzhou Normal University, and Prof. Zhu Li from University of Missouri at Kansas . Please contact us (mazhan@nju.edu.cn and wangjq@smail.nju.edu.cn) if you have any questions.

