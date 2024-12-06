"""
python dataset.py --process='partition' --input_rootdir='/home/temp/wjq/dataset/Microsoft/origin/phil10/plyfile/' \
    --output_rootdir='/home/temp/wjq/dataset/testdata_voxeldnn/phil10' --num_points=1200000 --input_format='ply' --output_format='ply'

python dataset.py --process='quantize' --precision=0.001 --input_rootdir='../../dataset/sparsepcgc_testdata/KITTI/' \ 
    --output_rootdir='../../dataset/sparsepcgc_testdata/KITTI_q1mm/' --input_format='bin' --output_format='ply'

python dataset.py --process='quantize' --precision=0.02 --input_rootdir='../../dataset/sparsepcgc_testdata/KITTI/' \
    --output_rootdir='../../dataset/sparsepcgc_testdata/KITTI_q2cm/' --input_format='bin' --output_format='ply'
"""
import os; rootdir = os.path.split(__file__)[0]
import sys; sys.path.append(rootdir)
import open3d as o3d
import numpy as np
import argparse
import glob
import random
from tqdm import tqdm
from inout import read_coords, write_h5, write_ply_o3d
from quantize import quantize_resolution
from partition import kdtree_partition



###################################### mesh2points #############################################
def mesh2points(mesh, num_points):
    """
    sample points uniformly
    !pip install open3d
    """
    try:
        pcd = mesh.sample_points_uniformly(number_of_points=int(num_points))
    except:
        print("ERROR mesh2points !")
        return np.asarray([[0,0,0]])
    points = np.asarray(pcd.points)

    return points

def random_rotate(points):
    # get_rotate_matrix
    matrix = np.eye(3,dtype='float32')
    matrix[0,0] *= np.random.randint(0,2)*2-1
    matrix = np.dot(matrix, np.linalg.qr(np.random.randn(3,3))[0])
    # random_rotate
    points = np.dot(points, matrix)

    return points

def main_mesh2pc(input_rootdir, output_rootdir, input_format, output_format, input_length, output_length, num_points, resolution=255):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+input_format), recursive=True))
    random.shuffle(input_filedirs)
    input_filedirs=input_filedirs[:input_length]
    print("input length:\t", len(input_filedirs))
    for idx, input_filedir in enumerate(tqdm(input_filedirs)):
        # mesh2points
        mesh = o3d.io.read_triangle_mesh(input_filedir)
        points = mesh2points(mesh, num_points)
        if len(points)==1: continue
        # random rotate
        points = random_rotate(points)
        # quantize
        points = quantize_resolution(points, resolution=resolution, eturn_offset=False)
        points = np.unique(points, axis=0)
        print("DBG!!! nums:\t", len(points))
        # save
        output_filedir = os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0])
        output_folder, _ = os.path.split(output_filedir)
        os.makedirs(output_folder, exist_ok=True)
        if output_format == 'ply': write_ply_o3d(output_filedir+'.ply', points)
        if output_format == 'h5': write_h5(output_filedir+'.h5', points)
        if idx >= output_length: break

    return 


###################################### partition #############################################
def main_partition(input_rootdir, output_rootdir, input_format, output_format, input_length, output_length, num_points):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+input_format), recursive=True))[:input_length]
    print("input length:\t", len(input_filedirs))
    np.random.shuffle(input_filedirs)
    count = 0
    for _, input_filedir in enumerate(tqdm(input_filedirs)):
        # load
        points = read_coords(input_filedir)
        # partition
        max_num = num_points
        # max_num = 10+points.shape[0]//2
        kdtree_parts = kdtree_partition(points, max_num)
        # kdtree_parts = random.sample(kdtree_parts, 4)
        for idx_part, points_part in enumerate(kdtree_parts):
            points_part = points_part - np.min(points_part, axis=0)
            # save
            output_filedir = os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0]+'_'+str(idx_part))
            output_folder, _ = os.path.split(output_filedir)
            os.makedirs(output_folder, exist_ok=True)
            if output_format == 'ply': write_ply_o3d(output_filedir+'.ply', points_part)
            if output_format == 'h5': write_h5(output_filedir+'.h5', points_part)
            count += 1
        if count >= output_length: break
    
    return


###################################### quantize #############################################
from quantize import quantize_precision
def main_quantize(input_rootdir, output_rootdir, input_format, output_format, input_length, output_length, precision):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+input_format), recursive=True))[:input_length]
    print("input length:\t", len(input_filedirs))
    np.random.shuffle(input_filedirs)
    count = 0
    for _, input_filedir in enumerate(tqdm(input_filedirs)):
        # load
        points = read_coords(input_filedir)
        # quantize
        points = quantize_precision(points, precision=precision, quant_mode='round', return_offset=False)
        points = np.unique(points.astype('int32'), axis=0).astype('int32')
       # save
        output_filedir = os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0])
        output_folder, _ = os.path.split(output_filedir)
        os.makedirs(output_folder, exist_ok=True)
        if output_format == 'ply': write_ply_o3d(output_filedir+'.ply', points, dtype='float32')
        if output_format == 'h5': write_h5(output_filedir+'.h5', points)
        print('pre quantize', input_filedir, points.max() - points.min())

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--process", default='mesh2pc')
    parser.add_argument("--input_rootdir", default='')
    parser.add_argument("--output_rootdir", default='')
    parser.add_argument("--input_format", default='h5')
    parser.add_argument("--output_format", default='h5')
    parser.add_argument("--input_length", type=int, default=int(1e6))
    parser.add_argument("--output_length", type=int, default=int(1e6))
    parser.add_argument("--num_points", type=int, default=8e5)
    parser.add_argument("--resolution", type=int, default=255)
    parser.add_argument("--precision", type=float, default=0.001)
    # parser.add_argument("--voxel_size", type=int, default=2)
    args = parser.parse_args()

    # mesh2pc
    if args.process=='mesh2pc':
        main_mesh2pc(input_rootdir=args.input_rootdir, output_rootdir=args.output_rootdir, 
                    input_format=args.input_format, output_format=args.output_format, 
                    input_length=args.input_length, output_length=args.output_length, 
                    num_points=args.num_points, resolution=args.resolution)

    # partition
    if args.process=='partition':
        main_partition(input_rootdir=args.input_rootdir, output_rootdir=args.output_rootdir, 
                    input_format=args.input_format, output_format=args.output_format, 
                    input_length=args.input_length, output_length=args.output_length, 
                    num_points=args.num_points)

    # quantize
    if args.process=='quantize':
        main_quantize(input_rootdir=args.input_rootdir, output_rootdir=args.output_rootdir, 
                    input_format=args.input_format, output_format=args.output_format, 
                    input_length=args.input_length, output_length=args.output_length,
                    precision=args.precision)


    # filedir_list_ford = sorted(glob.glob(os.path.join('/home/temp/wjq/dataset/sparsepcgc_testdata/Ford/','**', f'*.ply'), recursive=True))
    # input_rootdir = '/home/temp/wjq/dataset/Ford/origin/'
    # filedir_list_ford = sorted(glob.glob(os.path.join(input_rootdir,'**', f'*.ply'), recursive=True))[::50]
    # print(filedir_list_ford)
    # output_rootdir = '/home/temp/wjq/dataset/Ford/test/test90/'
    # import shutil
    # for idx, input_filedir in enumerate(filedir_list_ford):
    #     output_filedir = os.path.join(output_rootdir, input_filedir[len(input_rootdir):])
    #     output_folder, _ = os.path.split(output_filedir)
    #     print('debug:\t', idx, input_filedir, '\t:', output_filedir)
    #     os.makedirs(output_folder, exist_ok=True)
    #     shutil.copyfile(input_filedir, output_filedir)


    # filedir_list_kitti = sorted(glob.glob(os.path.join('/home/temp/wjq/dataset/sparsepcgc_testdata/KITTI/','**', f'*.bin'), recursive=True))
    # output_rootdir = '/home/temp/wjq/dataset/KITTI_110'
    # import shutil
    # for idx, input_filedir in enumerate(filedir_list_kitti):
    #     filename = os.path.split(input_filedir)[-1]
    #     output_filedir = os.path.join(output_rootdir, str(idx)+'_'+filename)
    #     output_folder, _ = os.path.split(output_filedir)
    #     print('debug:\t', idx, input_filedir, '\t:', output_filedir)
    #     os.makedirs(output_folder, exist_ok=True)
    #     shutil.copyfile(input_filedir, output_filedir)