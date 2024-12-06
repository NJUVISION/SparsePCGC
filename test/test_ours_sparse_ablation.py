import os, sys
rootdir = os.path.split(__file__)[0]
sys.path.append(rootdir)
rootdir = os.path.split(rootdir)[0]
sys.path.append(rootdir)
import time
import numpy as np
import os, glob, tqdm
import torch
import pandas as pd

from models.model import PCCModel
from coder import BasicCoder2


def test_sparse_lossless(ckptdir_low, ckptdir_high, filedir_list, voxel_size=1):
    model_low = PCCModel(stage=8, kernel_size=args.kernel_size, 
                        channels=args.channels, enc_type='pooling').to(device)
    assert os.path.exists(ckptdir_low)
    ckpt = torch.load(ckptdir_low)
    model_low.load_state_dict(ckpt['model'])

    model_high = PCCModel(stage=8, kernel_size=args.kernel_size, 
                        channels=args.channels, enc_type='pooling').to(device)
    assert os.path.exists(ckptdir_high)
    ckpt = torch.load(ckptdir_high)
    model_high.load_state_dict(ckpt['model'])

    basic_coder = BasicCoder2(model_low, model_high, device=device)

    for idx_file, filedir in enumerate(tqdm.tqdm(filedir_list)):
        filename = os.path.split(filedir)[-1].split('.')[0]
        bin_dir = os.path.join(args.outdir, filename+'.bin')
        dec_dir = os.path.join(args.outdir, filename+'_dec.ply')
        results = basic_coder.test(filedir, bin_dir, dec_dir, voxel_size=voxel_size, posQuantscale=1)
        print('DBG!!! results:\t', results)
        results = pd.DataFrame([results])
        if idx_file==0: all_results = results.copy(deep=True)
        else: all_results = all_results.append(results, ignore_index=True)
        csvfile = os.path.join(args.resultdir, args.prefix+'_data'+str(len(filedir_list))+'.csv')   
        all_results.to_csv(csvfile, index=False)
    print('save results to ', csvfile)
    print(all_results.mean())

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", default='')# lossless
    parser.add_argument("--stage", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--filedir",type=str, default='')
    parser.add_argument("--outdir",type=str, default='output')
    parser.add_argument("--ckptdir_low",type=str, default='../ckpts/sparse_low/epoch_last.pth')
    parser.add_argument("--ckptdir_high",type=str, default='../ckpts/sparse_high/epoch_last.pth')
    parser.add_argument("--ckptdir_offset",type=str, default='../ckpts/sparse_offset/epoch_last.pth')
    parser.add_argument("--voxel_size",type=float, default=1)
    parser.add_argument("--resultdir",type=str, default='results')
    parser.add_argument("--offset", action="store_true", help="offset or not.")
    parser.add_argument("--prefix",type=str, default='ours_sparse')
    args = parser.parse_args()
    args.outdir = os.path.join(rootdir, args.outdir, args.prefix)
    args.resultdir = os.path.join(rootdir, args.resultdir)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resultdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ################# test dataset ################# 
    filedir_list = sorted(glob.glob(os.path.join(args.filedir,'**', f'*.*'), recursive=True))
    filedir_list = [f for f in filedir_list if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]

    all_results = test_sparse_lossless(ckptdir_low=args.ckptdir_low, ckptdir_high=args.ckptdir_low,
                                        filedir_list=filedir_list, voxel_size=args.voxel_size)
