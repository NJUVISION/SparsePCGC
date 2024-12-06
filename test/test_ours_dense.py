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
from coder import BasicCoder, LossyCoderDense

# coder1

def test_dense_lossless(ckptdir, filedir_list):
    model = PCCModel(stage=8, kernel_size=3, enc_type='pooling').to(device)
    assert os.path.exists(ckptdir)
    ckpt = torch.load(ckptdir)
    model.load_state_dict(ckpt['model'])
    basic_coder = BasicCoder(model, device=device)

    for idx_file, filedir in enumerate(tqdm.tqdm(filedir_list)):
        filename = os.path.split(filedir)[-1].split('.')[0]
        bin_dir = os.path.join(args.outdir, filename+'.bin')
        dec_dir = os.path.join(args.outdir, filename+'_dec.ply')
        results = basic_coder.test(filedir, bin_dir, dec_dir, voxel_size=1, posQuantscale=1)
        print('DBG!!! results:\t', results)
        results = pd.DataFrame([results])
        if idx_file==0: all_results = results.copy(deep=True)
        else: all_results = all_results.append(results, ignore_index=True)
        csvfile = os.path.join(args.resultdir, args.prefix+'_data'+str(len(filedir_list))+'.csv')   
        all_results.to_csv(csvfile, index=False)
    print('save results to ', csvfile)
    print(all_results.mean())

    return all_results

def test_dense_lossy(ckptdir, ckptdir_sr, ckptdir_ae, filedir_list):
    model = PCCModel(stage=8, kernel_size=3, enc_type='pooling').to(device)
    assert os.path.exists(ckptdir)
    ckpt = torch.load(ckptdir)
    model.load_state_dict(ckpt['model'])
    basic_coder = BasicCoder(model, device=device)

    model_SR = PCCModel(stage=1, kernel_size=3, enc_type='pooling').to(device)
    assert os.path.exists(ckptdir_sr)
    ckpt = torch.load(ckptdir_sr)
    model_SR.load_state_dict(ckpt['model'])

    model_AE = PCCModel(stage=1, kernel_size=3, enc_type='ae').to(device)
    assert os.path.exists(ckptdir_ae)
    ckpt = torch.load(ckptdir_ae)
    model_AE.load_state_dict(ckpt['model'])

    lossy_coder = LossyCoderDense(basic_coder, model_AE, model_SR, device=device)
    
    for idx_file, filedir in enumerate(tqdm.tqdm(filedir_list)):
        filename = os.path.split(filedir)[-1].split('.')[0]
        scale_AE_list = [1,0,1,0,1,0]
        scale_SR_list = [0,1,1,2,2,3]
        results_list = []
        idx_rate = 0
        for scale_AE, scale_SR in zip(scale_AE_list, scale_SR_list):
            bin_dir = os.path.join(args.outdir, filename+'_R'+str(idx_rate)+'.bin')
            dec_dir = os.path.join(args.outdir, filename+'_R'+str(idx_rate)+'.ply')
            idx_rate += 1
            results = lossy_coder.test(filedir, bin_dir, dec_dir,
                                scale_AE=scale_AE, scale_SR=scale_SR, psnr_resolution=args.psnr_resolution)
            print('DBG!!! results', results)
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
    # data
    parser.add_argument("--data", default='')# 8i, owlii, kitti, ford, etc
    parser.add_argument("--filedir",type=str, default='')
    parser.add_argument("--mode", default='lossless')# lossy, lossless
    parser.add_argument("--psnr_resolution",type=int, default=1023)
    parser.add_argument("--outdir",type=str, default='output')
    parser.add_argument("--ckptdir",type=str, default='../ckpts/dense/epoch_last.pth')
    parser.add_argument("--ckptdir_sr",type=str, default='../ckpts/dense_1stage/epoch_last.pth')
    parser.add_argument("--ckptdir_ae",type=str, default='../ckpts/dense_slne/epoch_last.pth')
    parser.add_argument("--resultdir",type=str, default='results')
    parser.add_argument("--prefix",type=str, default='ours_lossless')
    args = parser.parse_args()
    args.outdir = os.path.join(rootdir, args.outdir, args.prefix)
    args.resultdir = os.path.join(rootdir, args.resultdir)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resultdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    filedir_list = sorted(glob.glob(os.path.join(args.filedir,'**', f'*.ply'), recursive=True))

    ################# test #################
    if args.mode=='lossless':
        all_results = test_dense_lossless(ckptdir=args.ckptdir, filedir_list=filedir_list)

    if args.mode=='lossy':
        all_results = test_dense_lossy(ckptdir=args.ckptdir,
                                    ckptdir_sr=args.ckptdir_sr, 
                                    ckptdir_ae=args.ckptdir_ae,
                                    filedir_list=filedir_list)