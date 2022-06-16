# test G-PCC
## test solid point clouds
python test_gpcc_dense.py --mode='lossless' --filedir='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' --prefix='gpcc_8i'

python test_gpcc_dense.py --mode='lossless' --filedir='../../dataset/testdata/testdata_sparsepcgc/Owlii/' --prefix='gpcc_owlii'

python test_gpcc_dense.py --mode='lossy' --filedir='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' --prefix='gpcc_8i_lossy' --resolution=1023

python test_gpcc_dense.py --mode='lossy' --filedir='../../dataset/testdata/testdata_sparsepcgc/Owlii/' --prefix='gpcc_owlii_lossy' --resolution=2047



## test sparse point clouds
python test_gpcc_sparse.py --mode='lossless' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/' --voxel_size=1 --prefix='gpcc_kitti_q1mm'

python test_gpcc_sparse.py --mode='lossless' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q2cm/' --voxel_size=1 --prefix='gpcc_kitti_q2cm'

<!-- python test_gpcc_sparse.py --mode='lossless' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/' --voxel_size=20 --prefix='gpcc_kitti_q2cm' -->

python test_gpcc_sparse.py --mode='lossy' --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/' --prefix='gpcc_kitti_lossy'

python test_gpcc_sparse.py --mode='lossy' --filedir='../../dataset/testdata/testdata_sparsepcgc/Ford/' --prefix='gpcc_ford_lossy'

python test_gpcc_vcn.py --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI/'

python test_gpcc_attn.py --filedir='../../dataset/testdata/testdata_sparsepcgc/KITTI/'


# test SparsePCGC (TODO)