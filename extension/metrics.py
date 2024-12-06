import os, sys
rootdir = os.path.split(__file__)[0]
sys.path.append(rootdir)
rootdir = os.path.split(rootdir)[0]
sys.path.append(rootdir)
import subprocess
import time
import numpy as np
# import os; rootdir = os.path.split(__file__)[0]
# import sys; sys.path.append(os.path.abspath('..'))
from data_utils.inout import read_ply_o3d, write_ply_o3d
import open3d as o3d

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try: number = float(item) 
        except ValueError: continue
        
    return number

def pc_error(infile1, infile2, resolution, normal=False, show=False):
    start_time = time.time()
    # headersF = ["mse1      (p2point)", "mse1,PSNR (p2point)", 
    #            "h.       1(p2point)", "h.,PSNR  1(p2point)",
    #            "mse2      (p2point)", "mse2,PSNR (p2point)", 
    #            "h.       2(p2point)", "h.,PSNR  2(p2point)" ,
    #            "mseF      (p2point)", "mseF,PSNR (p2point)", 
    #            "h.        (p2point)", "h.,PSNR   (p2point)" ]
    # headersF_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
    #                   "mse2      (p2plane)", "mse2,PSNR (p2plane)",
    #                   "mseF      (p2plane)", "mseF,PSNR (p2plane)"]             
    headers = ["mseF      (p2point)", "mseF,PSNR (p2point)"]
    rootdir = os.path.split(__file__)[0]
    command = str(rootdir+'/pc_error_d' + 
                          ' -a '+infile1+ 
                          ' -b '+infile2+ 
                          ' --hausdorff=1 '+ 
                          ' --resolution='+str(resolution))
    if normal:
        headers +=["mseF      (p2plane)", "mseF,PSNR (p2plane)"]
        command = str(command + ' -n ' + infile1)
    results = {}   
    subp=subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')# python3.
        if show: print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value
        c=subp.stdout.readline() 

    return results

def chamfer_dist(a, b):
    pcdA = o3d.geometry.PointCloud()
    pcdA.points = o3d.utility.Vector3dVector(a.astype('float32'))
    pcdB = o3d.geometry.PointCloud()
    pcdB.points = o3d.utility.Vector3dVector(b.astype('float32'))
    distA = pcdA.compute_point_cloud_distance(pcdB)
    distB = pcdB.compute_point_cloud_distance(pcdA)
    distA = np.array(distA)**2
    distB = np.array(distB)**2

    return distA, distB

def get_PSNR_VCN(f1, f2):
    """pc0: origin data;    pc1: decded data
    """
    pc0 = read_ply_o3d(f1, dtype='float32')
    pc1 = read_ply_o3d(f2, dtype='float32')
    centroid = pc0.mean(axis=0)
    # print('centroid:\t', centroid)  # almost 0
    pc0 -= centroid
    pc1 -= centroid
    m = np.abs(pc0).max()
    pc0 /= m
    pc1 /= m
    mse0, mse1 = chamfer_dist(pc0, pc1)
    mse0, mse1 = mse0.mean(), mse1.mean()
    psnr0 = mse0.clip(1e-15, 1e10)
    psnr1 = mse1.clip(1e-15, 1e10)
    psnr0 = 10 * (np.log(1 * 1 / psnr0) / np.log(10))
    psnr1 = 10 * (np.log(1 * 1 / psnr1) / np.log(10))
    mse0 = round(mse0, 14)
    mse1 = round(mse1, 14)
    psnr0 = round(psnr0, 4)
    psnr1 = round(psnr0, 4)

    return {'mse':max(mse0 ,mse1), 'psnr':min(psnr0, psnr1)}
    
def get_PSNR_attn(f1, f2, resolution=1, test_d2=False):
    """pc0: origin data;    pc1: decded data
    """
    points1 = read_ply_o3d(f1, dtype='float32')
    points2 = read_ply_o3d(f2, dtype='float32')
    centroid = points1.mean(axis=0)
    points1 -= centroid
    points2 -= centroid
    max_value = np.max(np.abs(points1))
    points1 /= max_value
    points2 /= max_value
    filename = os.path.split(f1)[-1].split('.')[0]
    outdir = os.path.join(os.path.abspath('..'), 'output')
    os.makedirs(outdir, exist_ok=True)
    outfile1 = os.path.join(outdir, filename+'_ori.ply')
    outfile2 = os.path.join(outdir, filename+'_dec.ply')
    write_ply_o3d(outfile1, points1, dtype='float32')
    write_ply_o3d(outfile2, points2, dtype='float32')

    results = pc_error(outfile1, outfile2, resolution=1, normal=test_d2)

    return results
