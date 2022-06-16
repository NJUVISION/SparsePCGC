"""In this script, We follow the testing methods of VoxelContextNet.
"""
import os, sys
rootdir = os.path.split(__file__)[0]
sys.path.append(rootdir)
rootdir = os.path.split(rootdir)[0]
sys.path.append(rootdir)
import numpy as np
import subprocess
from data_utils.inout import read_ply_o3d, write_ply_o3d, read_bin
from extension.gpcc import number_in_line
from extension.metrics import get_PSNR_VCN
tmc3_rootdir = os.path.join(rootdir, 'extension', 'tmc3_v81')
import time


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

def test_one(filedir, Qstep, posQuantScale, DBG=False):
    start = time.time()
    pc = read_bin(filedir)
    num_points_raw = pc.shape[0]
    # normalize & quantize & write data
    centroid = np.mean(pc, axis=0)
    pc -= centroid
    pc *= Qstep
    filename = os.path.split(filedir)[-1][:-4]+'_n'+str(Qstep)
    ori_file = os.path.join(args.outdir, filename+'.ply')
    bin_file = os.path.join(args.outdir, filename + '.bin')
    rec_file = os.path.join(args.outdir, filename + '_rec.ply')
    write_ply_o3d(ori_file, pc, dtype='float32')
    # encode
    cmd = tmc3_rootdir + ' --mode=0 ' \
        +' --uncompressedDataPath=' + ori_file \
        + ' --compressedStreamPath=' + bin_file \
        + ' --mergeDuplicatedPoints=1' \
        + ' --positionQuantizationScale='+str(posQuantScale) \
        + ' --positionBaseQp=4'
    start_enc = time.time()
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    p.wait()
    enc_time = round(time.time() - start_enc, 3)
    all_enc_time = round(time.time() - start, 3)
    results_gpcc_enc = print_pipline(p, DBG)
    # decode
    start = time.time()
    cmd = tmc3_rootdir + ' --mode=1 --outputBinaryPly=0 ' \
        + ' --reconstructedDataPath=' + rec_file \
        + ' --compressedStreamPath=' + bin_file
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    p.wait()
    dec_time = round(time.time() - start, 3)
    all_dec_time = round(time.time() - start, 3)
    results_gpcc_dec = print_pipline(p, DBG)
    # bpp & psnr
    fsize = os.path.getsize(bin_file)
    fsize *= 8
    bpp = fsize / num_points_raw
    num_points = read_ply_o3d(rec_file, dtype='float32').shape[0]
    results = {'filedir':filedir, 'num_points_raw':num_points_raw, 'bpp':bpp, 'file_size':fsize, 'num_points':num_points,
            'enc_time':enc_time, 'all_enc_time':all_enc_time, 
            'enc_time_user':results_gpcc_enc['Processing time (user)'],
            'enc_time_wall':results_gpcc_enc['Processing time (wall)'],
            'dec_time':dec_time, 'all_dec_time':all_dec_time,
            'dec_time_user':results_gpcc_dec['Processing time (user)'],
            'dec_time_wall':results_gpcc_dec['Processing time (wall)']}

    psnr_results = get_PSNR_VCN(ori_file, rec_file)
    results.update(psnr_results)

    return results

def test_all(filedir, Qstep=100, posQuantScaleList=[0.3,0.1,0.05,0.03]):
    results_list = []
    for i, posQuantScale in enumerate(posQuantScaleList): 
        results = test_one(filedir, Qstep=Qstep, posQuantScale=posQuantScale)
        print('DBG!!! results:\t', results)
        results_list.append(results)
    results_all = {'filedir':results_list[0]['filedir'], 'num_points_raw':results_list[0]['num_points_raw']}
    for idx, results in enumerate(results_list):
        for k, v in results.items(): 
            if k in ['filedir', 'num_points_raw']: continue
            results_all['R'+str(idx)+'_'+k] = v
    
    return results_all

def test_all_sequences(filedir_list):
    for idx_file, filedir in enumerate(tqdm(filedir_list)):
        assert os.path.exists(filedir)
        results = test_all(filedir, Qstep=100, posQuantScaleList=[0.3,0.1,0.05,0.03])
        results = pd.DataFrame([results])
        if idx_file==0: all_results = results.copy(deep=True)
        else: all_results = all_results.append(results, ignore_index=True)
        csvfile = os.path.join(args.resultdir, args.prefix+'_data'+str(len(filedir_list))+'.csv')
        all_results.to_csv(csvfile, index=False)
    print('save results to ', csvfile)
    print(all_results.mean())



if __name__ == "__main__":
    import argparse, glob
    from tqdm import tqdm
    import pandas as pd
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir",type=str, default='')
    parser.add_argument("--outdir",type=str, default='output')
    parser.add_argument("--resultdir",type=str, default='results')
    parser.add_argument("--prefix",type=str, default='gpcc_vcn')
    args = parser.parse_args()
    args.outdir = os.path.join(rootdir, args.outdir, args.prefix)
    args.resultdir = os.path.join(rootdir, args.resultdir)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resultdir, exist_ok=True)

    ################# test dataset ################# 
    filedir_list = sorted(glob.glob(os.path.join(args.filedir,'**', f'*.*'), recursive=True))
    filedir_list = [f for f in filedir_list if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]
    
    test_all_sequences(filedir_list)


        

