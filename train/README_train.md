# train dense
python train.py --dataset='../../dataset/shared/wjq/dataset/ShapeNet/pc_vox8_n100k/'  --dataset_test='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' --stage=8 --channels=32 --kernel_size=3 --scale=4 --enc_type='pooling' --epoch=30 --batch_size=4 --augment --init_ckpt='../ckpts/dense/epoch_last.pth' --only_test

python train.py --dataset='../../dataset/shared/wjq/dataset/ShapeNet/pc_vox8_n100k/'  --dataset_test='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' --stage=1 --channels=32 --kernel_size=3 --scale=4 --enc_type='pooling' --epoch=30 --batch_size=4 --augment --init_ckpt='../ckpts/dense_1stage/epoch_last.pth' --only_test

python train.py --dataset='../../dataset/shared/wjq/dataset/ShapeNet/pc_vox8_n100k/'  --dataset_test='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' --stage=1 --channels=32 --kernel_size=3 --scale=4 --enc_type='ae' --epoch=30 --batch_size=4 --augment --init_ckpt='../ckpts/dense_slne/epoch_last.pth' --only_test


# train sparse
python train.py --dataset='../../dataset/KITTI/pc_q1mm/train_part/'  --dataset_test='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/' --stage=8 --kernel_size=5 --channels=32 --enc_type='pooling' --epoch=20 --batch_size=2 --augment --voxel_size=20 --scale=8 --only_test --init_ckpt='../ckpts/sparse_low/epoch_last.pth'

python train.py --dataset='../../dataset/KITTI/pc_q1mm/train_part/'  --dataset_test='../../dataset/testdata/testdata_sparsepcgc/KITTI_q1mm/' --stage=8 --kernel_size=5 --channels=32 --enc_type='pooling' --epoch=20 --batch_size=2 --augment --voxel_size=1 --scale=5 --only_test --init_ckpt='../ckpts/sparse_high/epoch_last.pth'

python train.py --dataset='../../dataset/KITTI/dataset/sequences/train/'  --dataset_test='../../dataset/testdata/testdata_sparsepcgc/KITTI/' --stage=8 --kernel_size=5 --channels=32 --enc_type='pooling' --epoch=20 --batch_size=2 --scale=8 --only_test --init_ckpt='../ckpts/sparse_low/epoch_last.pth'

python train_offset.py --posQuantscale_list 16 32 64 128 256 --dataset='../../dataset/KITTI/pc_q1mm/train_part/' --voxel_size=1 --augment --batch_size=4 --prefix='offset' --only_test

python train.py --dataset='../../dataset/Ford/origin/Ford_01_q_1mm/'  --dataset_test='../../dataset/testdata/testdata_sparsepcgc/Ford23/' --stage=8 --enc_type='pooling' --kernel_size=5 --epoch=20 --batch_size=1 --augment --voxel_size=20 --scale=8 --init_ckpt='../ckpts/anchor/sparse_low/epoch_last.pth' --prefix='ford_low'

