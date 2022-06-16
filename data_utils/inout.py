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