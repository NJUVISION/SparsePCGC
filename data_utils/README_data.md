
1. sample points on raw meshes (ShapeNet) to generate point clouds, and then randomly rotated and quantized them with 8-bit.


```shell
python dataset.py --process='mesh2pc' --input_rootdir='/home/temp/wjq/dataset/shared/ShapeNet/mesh/' --output_rootdir='/home/temp/wjq/dataset/shared/ShapeNet/pc_vox8/' --num_points=800000 --resolution=255 --input_format='obj' --output_format='h5'
```



2. partition point clouds into blocks of up to 100000 points (optional)

```shell
python dataset.py --process='partition' --input_rootdir='/home/temp/wjq/dataset/shared/ShapeNet/pc_vox8/' --output_rootdir='/home/temp/wjq/dataset/shared/ShapeNet/pc_vox8_n100k/' --num_points=100000 --input_format='h5' --output_format='h5'
```
