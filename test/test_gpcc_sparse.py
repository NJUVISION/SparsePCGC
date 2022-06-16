import os; rootdir=os.path.abspath('..')
import sys; sys.path.append(rootdir)
import numpy as np
import glob, argparse
from tqdm import tqdm
import pandas as pd
import time
from extension.metrics import pc_error
from extension.gpcc import gpcc_encode, gpcc_decode, number_in_line
from data_utils.inout import read_coords, write_ply_o3d
from data_utils.quantize import quantize_precision



def get_points_number(filedir):
    if filedir.endswith('ply'):
        plyfile = open(filedir)
        line = plyfile.readline()
        while line.find("element vertex") == -1:
            line = plyfile.readline()
        number = int(line.split(' ')[-1][:-1])
    elif filedir.endswith("bin"):
        number = len(np.fromfile(filedir, dtype='float32').reshape(-1, 4))

    return number

def print_pipline(p, DBG=False):
    headers = ['Processing time (user)', 'Processing time (wall)']
    results = {}
    c=p.stdout.readline()
    while c:
        if DBG: print(c)
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1: 
                value = number_in_line(line)
                results[key] = value
        c=p.stdout.readline()

    return results

def test_one(filedir, posQuantscale, resolution, test_psnr=False):
    filename = os.path.split(filedir)[-1].split('.')[0]
    bin_dir = os.path.join(args.outdir, filename + '.bin')
    dec_dir = os.path.join(args.outdir, filename + '_rec.ply')
    start = time.time()
    log_enc = gpcc_encode(filedir, bin_dir, posQuantscale=posQuantscale, 
                        tmc3dir='tmc3_v14', cfgdir='sparse.cfg')
    enc_time = round(time.time() - start, 3)
    start = time.time()
    log_dec = gpcc_decode(bin_dir, dec_dir, 
                        tmc3dir='tmc3_v14')
    dec_time = round(time.time() - start, 3)
    # bpp
    num_points_raw = get_points_number(filedir)
    num_points = get_points_number(dec_dir)
    # bpp = round(8*results_enc['Total bitstream size']/get_points_number(filedir), 4)
    file_size = os.path.getsize(bin_dir) * 8
    bpp = round(file_size/num_points_raw, 6)
    # results
    results_gpcc_enc = print_pipline(log_enc, False)
    results_gpcc_dec = print_pipline(log_dec, False)

    results={'filedir':filedir, 'posQuantscale':posQuantscale,  
            'num_points_raw':num_points_raw, 'num_points':num_points, 
            'file_size':file_size, 'bpp':bpp, 
            'enc_time':enc_time, 
            'enc_time_user':results_gpcc_enc['Processing time (user)'],
            'enc_time_wall':results_gpcc_enc['Processing time (wall)'],
            'dec_time':dec_time, 
            'dec_time_user':results_gpcc_dec['Processing time (user)'],
            'dec_time_wall':results_gpcc_dec['Processing time (wall)']}
    if test_psnr: 
        psnr_results = pc_error(filedir, dec_dir, resolution=resolution, normal=True, show=False)
        results.update(psnr_results)

    return results

def test_all(filedir, posQuantscaleList, resolution, test_d2=True):
    if test_d2:
        points = read_coords(filedir)
        filename = os.path.split(filedir)[-1].split('.')[0]
        filedir = os.path.join(args.outdir, filename+'.ply')
        write_ply_o3d(filedir, points, normal=True, knn=16)# test d2
    results_list = []
    for idx_rate, posQuantscale in enumerate(posQuantscaleList):
        results = test_one(filedir, posQuantscale, resolution=resolution, test_psnr=True)
        print('DBG!!! results:\t', results)
        results_list.append(results)
    results_all = {'filedir':results_list[0]['filedir'], 'num_points_raw':results_list[0]['num_points_raw']}
    for idx, results in enumerate(results_list):
        for k, v in results.items(): 
            if k in ['filedir', 'num_points_raw']: continue
            results_all['R'+str(idx)+'_'+k] = v
    
    return results_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir",type=str, default='')
    parser.add_argument("--mode", default="lossless")
    parser.add_argument("--voxel_size", type=float, default=1)
    parser.add_argument("--resolution", type=int, default=30000)
    parser.add_argument("--outdir",type=str, default='../output')
    parser.add_argument("--resultdir",type=str, default='../results')
    parser.add_argument("--prefix",type=str, default='gpcc')
    parser.add_argument("--knn", type=int, default=20)
    args = parser.parse_args()
    args.outdir = os.path.join(args.outdir, args.prefix)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resultdir, exist_ok=True)
    

    ################# test dataset ################# 
    filedir_list = sorted(glob.glob(os.path.join(args.filedir,'**', f'*.*'), recursive=True))
    filedir_list = [f for f in filedir_list if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]
    
    ################# test #################      
    for idx_file, filedir in enumerate(tqdm(filedir_list)):
        print('='*8, idx_file, filedir, '='*8)
        assert os.path.exists(filedir)
        if filedir.endswith('bin'):
            coords = read_coords(filedir)
            coords = quantize_precision(coords, precision=0.001, quant_mode='round', return_offset=False)
            coords = np.unique(coords.astype('int32'), axis=0).astype('int32')
            filename = os.path.split(filedir)[-1].split('.')[0]
            filedir = os.path.join(args.outdir, filename + '_q1mm.ply')
            write_ply_o3d(filedir, coords, dtype='int32')
            print('pre quantize', filedir, coords.max() - coords.min())

        if args.mode=='lossless':
            if args.voxel_size>1:
                coords = read_coords(filedir)
                coords = quantize_precision(coords, precision=args.voxel_size, quant_mode='round', return_offset=False)
                coords = np.unique(coords.astype('int32'), axis=0).astype('int32')
                filename = os.path.split(filedir)[-1].split('.')[0]
                filedir = os.path.join(args.outdir, filename + '_q'+str(round(args.voxel_size))+'mm.ply')
                write_ply_o3d(filedir, coords, dtype='int32')
                print('quantize', filedir, args.voxel_size, coords.max() - coords.min())
            results = test_one(filedir, posQuantscale=1, resolution=args.resolution, test_psnr=False)
            results['filedir'] = filedir
        elif args.mode=='lossy':
            posQuantscaleList = np.array([1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512])
            results = test_all(filedir, posQuantscaleList, resolution=args.resolution, test_d2=True)
        print('DBG!!! results:\t', results)
        results = pd.DataFrame([results])
        if idx_file==0: all_results = results.copy(deep=True)
        else: all_results = all_results.append(results, ignore_index=True)
        csvfile = os.path.join(args.resultdir, args.prefix+'_data'+str(len(filedir_list))+'.csv')   
        all_results.to_csv(csvfile, index=False)
    print('save results to ', csvfile)
    print(all_results.mean())
