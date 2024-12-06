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
from models.model_offset import OffsetModel
from coder import BasicCoder2, LossyCoderSparse


def test_sparse_lossless(ckptdir_low, ckptdir_high, filedir_list, voxel_size=1):
    model_low = PCCModel(stage=8, kernel_size=5, enc_type='pooling').to(device)
    assert os.path.exists(ckptdir_low)
    ckpt = torch.load(ckptdir_low)
    model_low.load_state_dict(ckpt['model'])

    model_high = PCCModel(stage=8, kernel_size=5, enc_type='pooling').to(device)
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


def test_sparse_lossy(ckptdir_low, ckptdir_high, ckptdir_offset, filedir_list, mode='gpcc', offset=False):
    model_low = PCCModel(stage=8, kernel_size=5, enc_type='pooling').to(device)
    assert os.path.exists(ckptdir_low)
    ckpt = torch.load(ckptdir_low)
    model_low.load_state_dict(ckpt['model'])

    model_high = PCCModel(stage=8, kernel_size=5, enc_type='pooling').to(device)
    assert os.path.exists(ckptdir_high)
    ckpt = torch.load(ckptdir_high)
    model_high.load_state_dict(ckpt['model'])

    basic_coder = BasicCoder2(model_low, model_high, device=device)

    if offset:
        model_offset = OffsetModel(kernel_size=5).to(device)
        assert os.path.exists(ckptdir_offset)
        ckpt = torch.load(ckptdir_offset)
        model_offset.load_state_dict(ckpt['model'])
    else:
        model_offset = None
    lossy_coder = LossyCoderSparse(basic_coder, model_offset=model_offset, device=device)

    for idx_file, filedir in enumerate(tqdm.tqdm(filedir_list)):
        filename = os.path.split(filedir)[-1].split('.')[0]
        # mode1: input Ford_q1mm/or KITTI raw, adjust posQuantScale
        if mode=='lossy_gpcc':
            # preprocessing: quantize to 1mm
            if filedir.endswith('bin'):
                from data_utils.inout import read_coords, write_ply_o3d
                from data_utils.quantize import quantize_precision
                coords = read_coords(filedir)
                coords = quantize_precision(coords, precision=0.001, quant_mode='round', return_offset=False)
                coords = np.unique(coords.astype('int32'), axis=0).astype('int32')
                filename = os.path.split(filedir)[-1].split('.')[0]
                filedir = os.path.join(args.outdir, filename + '_q1mm.ply')
                write_ply_o3d(filedir, coords, dtype='int32')
                print('pre quantize', filedir, coords.max() - coords.min())
            #
            # posQuantscale_list = [4, 8, 32, 64, 256, 512]
            posQuantscale_list = [4, 8, 16, 32, 64, 128, 256, 512]
            results_list = []
            for idx_rate, posQuantscale in enumerate(posQuantscale_list):
                bin_dir = os.path.join(args.outdir, filename+'_R'+str(idx_rate)+'.bin')
                dec_dir = os.path.join(args.outdir, filename+'_R'+str(idx_rate)+'.ply')
                results = lossy_coder.test(filedir, bin_dir, dec_dir, posQuantscale=posQuantscale, 
                                        quant_mode='precision', quant_factor=1, psnr_mode='gpcc', test_d2=True)
                print('DBG!!! results:\t', results)
                results_list.append(results)
        # mode2: input kitti raw, adjust resolution
        if mode=='lossy_vcn':
            resolution_list = [2**12-1, 2**11-1, 2**10-1, 2**9-1]
            results_list = []
            for i, resolution in enumerate(resolution_list):
                bin_dir = os.path.join(args.outdir, filename+'_R'+str(i)+'.bin')
                dec_dir = os.path.join(args.outdir, filename+'_R'+str(i)+'.ply')
                results = lossy_coder.test(filedir, bin_dir, dec_dir, posQuantscale=1, 
                                        quant_mode='resolution', quant_factor=resolution, psnr_mode='vcn')
                print('DBG!!! results:\t', results)
                results_list.append(results)
        # mode3: input kitti raw, adjust octree level
        if mode=='lossy_attn':
            qlevel_list = [12,11,10,9]
            results_list = []
            for i, qlevel in enumerate(qlevel_list):
                bin_dir = os.path.join(args.outdir, filename+'_R'+str(i)+'.bin')
                dec_dir = os.path.join(args.outdir, filename+'_R'+str(i)+'.ply')
                results = lossy_coder.test(filedir, bin_dir, dec_dir, posQuantscale=1, 
                                        quant_mode='octree', quant_factor=qlevel, psnr_mode='attn', test_d2=True)
                print('DBG!!! results:\t', results)
                results_list.append(results)

        # collect results
        results = {'filedir':results_list[0]['filedir'], 'num_points_raw':results_list[0]['num_points_raw']}
        for idx, results_one in enumerate(results_list):
            for k, v in results_one.items(): 
                if k in ['filedir', 'num_points_raw']: continue
                results['R'+str(idx)+'_'+k] = v
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
    parser.add_argument("--filedir",type=str, default='')
    parser.add_argument("--outdir",type=str, default='output')
    parser.add_argument("--ckptdir_low",type=str, default='../ckpts/anchor/sparse_low/epoch_last.pth')
    parser.add_argument("--ckptdir_high",type=str, default='../ckpts/anchor/sparse_high/epoch_last.pth')
    parser.add_argument("--ckptdir_offset",type=str, default='../ckpts/anchor/sparse_offset/epoch_last.pth')
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

    ################# test #################

    if args.mode=='lossless':
        all_results = test_sparse_lossless(ckptdir_low=args.ckptdir_low, ckptdir_high=args.ckptdir_high,
                                            filedir_list=filedir_list, voxel_size=args.voxel_size)
    if args.mode[:5]=='lossy':
        all_results = test_sparse_lossy(ckptdir_low=args.ckptdir_low, ckptdir_high=args.ckptdir_high, 
                                        ckptdir_offset =args.ckptdir_offset,
                                        filedir_list=filedir_list, mode=args.mode, offset=args.offset)
    