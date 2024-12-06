# dense
python test_ours_dense.py --mode='lossless' --ckptdir='../ckpts/dense/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' --prefix='ours_8i'

python test_ours_dense.py --mode='lossless' --filedir='../../dataset/testdata/testdata_sparsepcgc/Owlii/' --ckptdir='../ckpts/dense/epoch_last.pth' --prefix='ours_owlii'

# dense lossy
python test_ours_dense.py --mode='lossy' --ckptdir='../ckpts/dense/epoch_last.pth' --ckptdir_sr='../ckpts/dense_1stage/epoch_last.pth' --ckptdir_ae='../ckpts/dense_slne/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' --psnr_resolution=1023 --prefix='ours_8i_lossy'

python test_ours_dense.py --mode='lossy' --ckptdir='../ckpts/dense/epoch_last.pth' --ckptdir_sr='../ckpts/dense_1stage/epoch_last.pth' --ckptdir_ae='../ckpts/dense_slne/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/Owlii/' --psnr_resolution=2047 --prefix='ours_owlii_lossy'

## gpcc
python test_gpcc_dense.py --mode='lossless' --filedir='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' --prefix='gpcc_8i'

python test_gpcc_dense.py --mode='lossless' --filedir='../../dataset/testdata/testdata_sparsepcgc/Owlii/' --prefix='gpcc_owlii'

python test_gpcc_dense.py --mode='lossy' --filedir='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' --prefix='gpcc_8i_lossy' --resolution=1023

python test_gpcc_dense.py --mode='lossy' --filedir='../../dataset/testdata/testdata_sparsepcgc/Owlii/' --prefix='gpcc_owlii_lossy' --resolution=2047


## ablation study
python test_ours_dense_ablation.py --stage=1 --ckptdir='../ckpts/dense_1stage/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' --prefix='ours_1stage'


# test sparse
python test_ours_sparse.py --mode='lossless' --ckptdir_low='../ckpts/sparse_low/epoch_last.pth' --ckptdir_high='../ckpts/sparse_high/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q2cm/'  --voxel_size=1 --prefix='ours_kitti_q2cm'

python test_ours_sparse.py --mode='lossless' --ckptdir_low='../ckpts/sparse_low/epoch_last.pth' --ckptdir_high='../ckpts/sparse_high/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/'  --voxel_size=1 --prefix='ours_kitti_q1mm'

python test_ours_sparse.py --mode='lossless' --ckptdir_low='../ckpts/sparse_low/epoch_last.pth' --ckptdir_high='../ckpts/sparse_high/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/Ford_q2cm/'  --voxel_size=1 --prefix='ours_ford_q2cm'

python test_ours_sparse.py --mode='lossless' --ckptdir_low='../ckpts/sparse_low/epoch_last.pth' --ckptdir_high='../ckpts/sparse_high/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/Ford/'  --voxel_size=1 --prefix='ours_ford_q1mm'

<!-- 
python test_ours_sparse.py --mode='lossless' --ckptdir_low='../ckpts/sparse_low/epoch_last.pth' --ckptdir_high='../ckpts/sparse_high/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI/'  --voxel_size=0.001

python test_ours_sparse.py --mode='lossless' --ckptdir_low='../ckpts/sparse_low/epoch_last.pth' --ckptdir_high='../ckpts/sparse_high/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI/'  --voxel_size=0.02 -->


## test sparse lossy

python test_ours_sparse.py --mode='lossy_gpcc' --ckptdir_low='../ckpts/sparse_low/epoch_last.pth' --ckptdir_high='../ckpts/sparse_high/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/' --voxel_size=1 --prefix='ours_lossy_kitti'

python test_ours_sparse.py --mode='lossy_gpcc' --ckptdir_low='../ckpts/sparse_low/epoch_last.pth' --ckptdir_high='../ckpts/sparse_high/epoch_last.pth' --ckptdir_offset='../ckpts/sparse_offset/epoch_last.pth' --offset --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/' --voxel_size=1 --prefix='ours_lossy_kitti_offset'

python test_ours_sparse.py --mode='lossy_gpcc' --ckptdir_low='../ckpts/sparse_low/epoch_last.pth' --ckptdir_high='../ckpts/sparse_high/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/Ford/' --voxel_size=1 --prefix='ours_lossy_ford'

python test_ours_sparse.py --mode='lossy_gpcc' --ckptdir_low='../ckpts/sparse_low/epoch_last.pth' --ckptdir_high='../ckpts/sparse_high/epoch_last.pth' --ckptdir_offset='../ckpts/sparse_offset/epoch_last.pth' --offset --filedir='../../dataset/testdata/testdata_sparsepcgc/Ford/' --voxel_size=1 --prefix='ours_lossy_ford_offset'



## gpcc
python test_gpcc_sparse.py --mode='lossless' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/' --voxel_size=1 --prefix='gpcc_kitti_q1mm'

python test_gpcc_sparse.py --mode='lossless' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q2cm/' --voxel_size=1 --prefix='gpcc_kitti_q2cm'

<!-- python test_gpcc_sparse.py --mode='lossless' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/' --voxel_size=20 --prefix='gpcc_kitti_q2cm' -->


python test_gpcc_sparse.py --mode='lossy' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/' --prefix='gpcc_kitti_lossy'

python test_gpcc_sparse.py --mode='lossy' --filedir='../../dataset/testdata/testdata_sparsepcgc/Ford/' --prefix='gpcc_ford_lossy'




## Comparison to VCN/OctAttention

python test_ours_sparse.py --mode='lossy_vcn' --ckptdir_low='../ckpts/sparse_low/
epoch_last.pth' --ckptdir_high='../ckpts/sparse_high/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI/' --prefix='ours_vcn'

python test_ours_sparse.py --mode='lossy_vcn' --ckptdir_low='../ckpts/sparse_low/epoch_last.pth' --ckptdir_high='../ckpts/sparse_high/epoch_last.pth' --ckptdir_offset='../ckpts/sparse_offset/epoch_last.pth' --offset --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI' --voxel_size=1 --prefix='ours_lossy_kitti_offset'

python test_ours_sparse.py --mode='lossy_attn' --ckptdir_low='../ckpts/sparse_low/epoch_last.pth' --ckptdir_high='../ckpts/sparse_high/epoch_last.pth' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI/' --prefix='ours_attn'


python test_gpcc_vcn.py --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI/'

python test_gpcc_attn.py --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI/'
