from logging import root
import open3d as o3d
import os, time
import numpy as np
import h5py

def read_h5(filedir, dtype="int32"):
    pc = h5py.File(filedir, 'r')['data'][:]
    coords = pc[:,0:3].astype(dtype)

    return coords

def write_h5(filedir, coords, dtype="int32"):
    data = coords.astype(dtype)
    with h5py.File(filedir, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)

    return

def read_ply_ascii(filedir, dtype="int32"):
    files = open(filedir, 'r')
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:,0:3].astype(dtype)

    return coords

def write_ply_ascii(filedir, coords, dtype='int32'):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    coords = coords.astype(dtype)
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ',str(p[2]), '\n'])
    f.close() 

    return

def read_ply_o3d(filedir, dtype='int32'):
    pcd = o3d.io.read_point_cloud(filedir)
    coords = np.asarray(pcd.points).astype(dtype)

    return coords

def write_ply_o3d(filedir, coords, dtype='int32', normal=False, knn=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(dtype))
    if normal:
        assert knn is not None
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    o3d.io.write_point_cloud(filedir, pcd, write_ascii=True)
    f = open(filedir)
    lines = f.readlines()
    lines[4] = 'property float x\n'
    lines[5] = 'property float y\n'
    lines[6] = 'property float z\n'
    if normal:
        lines[7] = 'property float nx\n'
        lines[8] = 'property float ny\n'
        lines[9] = 'property float nz\n'
    fo = open(filedir, "w")
    fo.writelines(lines)
    
    return

# from plyfile import PlyData
# def read_plyfile(path):
#     plydata = PlyData.read(path)
#     data = plydata.elements[0].data
#     points = np.asarray([data['x'],data['y'],data['z']]).T 

#     return points

def read_bin(filedir, dtype="float32"):
    """kitti
    """
    data = np.fromfile(filedir, dtype=dtype).reshape(-1, 4)
    coords = data[:,:3]
    
    return coords

def read_coords(filedir):
    if filedir.endswith('h5'): coords = read_h5(filedir)
    if filedir.endswith('ply'): coords = read_ply_o3d(filedir)
    if filedir.endswith('bin'): coords = read_bin(filedir)

    return coords

# def read_points(filedir, voxel_size=1, quant_mode='floor'):
#     if filedir.endswith('h5'): coords = read_h5_geo(filedir)
#     if filedir.endswith('ply'): coords = read_ply_o3d_geo(filedir)
#     if filedir.endswith('bin'): coords = quantize(read_bin(filedir), precision=0.001)
#     if voxel_size>1: 
#         if quant_mode=='round': coords = np.round(coords/voxel_size).astype('int')
#         if quant_mode=='floor': coords = np.floor(coords/voxel_size).astype('int')
#         coords = np.unique(coords, axis=0).astype('int')  
        # 
#     return coords


# import torch
# import MinkowskiEngine as ME

# import os; rootdir = os.path.split(__file__)[0]
# import sys; sys.path.append(rootdir)

# from quantize import quantize_precision
# from partition import kdtree_partition

# def load_sparse_tensor(filedir, voxel_size=1, quant_mode='floor', max_num=1e7, device='cuda'):
#     coords = read_coords(filedir)
#     # quantize:TODO
#     if filedir.endswith('bin'):
#         coords = quantize_precision(coords, precision=0.001)
#     if voxel_size>1: 
#         if quant_mode=='round': 
#             coords = np.round(coords/voxel_size).astype('int')
#         if quant_mode=='floor': 
#             coords = np.floor(coords/voxel_size).astype('int')
#         coords = np.unique(coords, axis=0).astype('int')
#     # partition
#     if coords.shape[0] <= max_num:      
#         coords = torch.tensor(coords).int()
#         feats = torch.ones((len(coords),1)).float()
#         coords, feats = ME.utils.sparse_collate([coords], [feats])
#         x = ME.SparseTensor(
#             features=feats, 
#             coordinates=coords, 
#             tensor_stride=1, 
#             device=device)
        
#         return x
#     else:
#         coords_list = kdtree_partition(coords, max_num=max_num)
#         x_list = []
#         for coords_part in coords_list:
#             coords_part = torch.tensor(coords_part).int()
#             feats_part = torch.ones((len(coords_part),1)).float()
#             coords_part, feats_part = ME.utils.sparse_collate([coords_part], [feats_part])
#             x_part = ME.SparseTensor(
#                 features=feats_part, 
#                 coordinates=coords_part, 
#                 tensor_stride=1, 
#                 device=device)
#             x_list.append(x_part)
            
#         return x_list
