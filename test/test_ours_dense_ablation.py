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
from coder import BasicCoder

# coder1

def test_dense_lossless(ckptdir, filedir_list, enc_type='pooling'):
    model = PCCModel(stage=args.stage, kernel_size=args.kernel_size, channels=args.channels, enc_type=enc_type).to(device)
    assert os.path.exists(ckptdir)
    ckpt = torch.load(ckptdir)
    model.load_state_dict(ckpt['model'])
    basic_coder = BasicCoder(model, device=device)

    for idx_file, filedir in enumerate(tqdm.tqdm(filedir_list)):
        print('DBG!!!', filedir)
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir",type=str, default='../../dataset/sparsepcgc_testdata/8iVFB/')
    parser.add_argument("--mode", default='lossless')# lossy, lossless
    parser.add_argument("--stage", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--outdir",type=str, default='output')
    parser.add_argument("--ckptdir",type=str, default='../../dataset/testdata/8iVFB/8iVFB/')
    parser.add_argument("--resultdir",type=str, default='results')
    parser.add_argument("--prefix",type=str, default='ours_lossless')
    args = parser.parse_args()
    args.outdir = os.path.join(rootdir, args.outdir, args.prefix)
    args.resultdir = os.path.join(rootdir, args.resultdir)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resultdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filedir_list = sorted(glob.glob(os.path.join(args.filedir,'**', f'*.ply'), recursive=True))

    ################# test #################
    all_results = test_dense_lossless(ckptdir=args.ckptdir, filedir_list=filedir_list, enc_type='pooling')

