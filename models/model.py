import os, sys
rootdir = os.path.split(__file__)[0]
sys.path.append(rootdir)
rootdir = os.path.split(rootdir)[0]
sys.path.append(rootdir)
import time
import torch
import MinkowskiEngine as ME
from extension.metrics import pc_error
from data_utils.data_loader import load_sparse_tensor
from data_utils.quantize import quantize_sparse_tensor
from data_utils.inout import write_ply_o3d
from data_utils.sparse_tensor import sort_sparse_tensor


class PCCModel(torch.nn.Module):
    def __init__(self, channels=32, kernel_size=3, block_layers=3, stage=8, scale=1, enc_type='pooling'):
        super().__init__()
        self.scale = scale
        self.stage = stage
        self.enc_type = enc_type
        if enc_type=='pooling': 
            self.downsampler = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
            latent_channels=1
        elif enc_type=='ae':
            from autoencoder import AutoEncoder
            self.downsampler = AutoEncoder(
                in_channels=1, channels=channels, out_channels=channels, kernel_size=kernel_size, 
                block_layers=block_layers)
            latent_channels=channels
        if stage==1: from decoder import UpSampler1Stage as UpSampler
        elif stage==0: from decoder_mask import UpSamplerMask as UpSampler
        else: from decoder import UpSampler8Stage as UpSampler
        self.upsampler = UpSampler(
            in_channels=latent_channels, channels=channels, kernel_size=kernel_size, 
            block_layers=block_layers, stage=stage)

    def forward(self, ground_truth, training=True):
        out_set_list = []
        for idx in range(self.scale):
            x = quantize_sparse_tensor(ground_truth, factor=1/(2**idx), quant_mode='floor')
            # print('DBG!!!', idx, '\t', len(x), x.C.max().cpu().numpy(), x.C.min().cpu().numpy(), x.C.max().cpu().numpy() - x.C.min().cpu().numpy())
            if self.enc_type=='pooling':
                x_low = self.downsampler(x)
            else:
                enc_set = self.downsampler(x, training=training)
                x_low = enc_set['x_low']
            out_set = self.upsampler(x_low, x_high=x)
            if self.enc_type!='pooling': out_set.update(enc_set)
            out_set_list.append(out_set)

        return out_set_list

    @torch.no_grad()
    def encode(self, x, scale=4):
        """lossless encode the input data
        """
        if x.C.min() < 0:
            ref_point = x.C.min(axis=0)[0]
            x = ME.SparseTensor(features=x.F, coordinates=x.C - ref_point, 
                                tensor_stride=x.tensor_stride, device=x.device)
        else: ref_point = None
        #
        bitstream_AE_list = []
        bitstream_list = []
        for idx in range(scale):
            if self.enc_type=='pooling': 
                x_low = self.downsampler(x)
            elif self.enc_type=='ae':
                x_low, bitstream_AE = self.downsampler.encode(x, return_one=False)
                bitstream_AE_list.append(bitstream_AE)
            bitstream = self.upsampler.encode(x_low, x)
            bitstream_list.append(bitstream)
            x = ME.SparseTensor(features=torch.ones((len(x_low),1)).float(),
                                coordinates=torch.div(x_low.C,2,rounding_mode='floor'), 
                                device=x_low.device)
            if x.shape[0]<64: break

        return {'ref_point':ref_point, 'bitstream_AE_list':bitstream_AE_list, 
                'bitstream_list':bitstream_list, 'x':x}

    @torch.no_grad()
    def decode(self, input_set):
        ref_point = input_set['ref_point']
        bitstream_AE_list = input_set['bitstream_AE_list'][::-1]
        bitstream_list = input_set['bitstream_list'][::-1]
        x = input_set['x']
        for idx, bitstream in enumerate(bitstream_list):
            x = ME.SparseTensor(features=x.F, coordinates=x.C*2,
                                tensor_stride=2, device=x.device)
            if self.enc_type=='ae':
                x = self.downsampler.decode(x, bitstream_AE_list[idx])
            x = self.upsampler.decode(x, bitstream)
        if ref_point is not None:
            x = ME.SparseTensor(features=x.F, coordinates=x.C + ref_point, 
                                tensor_stride=x.tensor_stride, device=x.device)

        return x

    @torch.no_grad()
    def test(self, x, scale=4):
        start = time.time()
        enc_set = self.encode(x, scale=scale)
        print('enc time:\t', round(time.time() - start, 3))
        
        start = time.time()
        x_dec = self.decode(enc_set)
        print('dec time:\t', round(time.time() - start, 3))
        assert (sort_sparse_tensor(x).C==sort_sparse_tensor(x_dec).C).all()
        bits = sum([len(bitstream) for bitstream in enc_set['bitstream_list']])*8
        if self.enc_type=='ae':
            bits_AE = sum([len(bitstream) for bitstream in enc_set['bitstream_AE_list']])*8
            # print('DBG!!!', bits_AE/x.shape[0])
            bits += bits_AE
        bpp = round(bits / x.shape[0], 3)
        print('bpp:\t', bpp, x.shape[0])
        print("memoey:\t", round(torch.cuda.max_memory_allocated()/1024**3,2),'GB')

        return 


if __name__ == '__main__':
    """
    python model.py --stage=8 --scale=8 --kernel_size=5 --ckptdir='ckpts/anchor/vcn2/epoch_last.pth' --filedir='testdata/kitti_seqs11_000000.bin' --voxel_size=0.02 --posQuantscale=1
    python model.py --stage=1 --enc_type='ae' --ckptdir='ckpts/anchor/ae/epoch_last.pth' --filedir='testdata/longdress_vox10_1300.ply' --scale=4 --voxel_size=1 --posQuantscale=1
    python model.py --stage=8 --ckptdir='ckpts/anchor/dense/epoch_last.pth' --filedir='testdata/longdress_vox10_1300.ply' --scale=4 --voxel_size=1 --posQuantscale=1
    python model.py --stage=1 --ckptdir='ckpts/anchor/sr/epoch_last.pth' --filedir='testdata/longdress_vox10_1300.ply' --scale=4 --voxel_size=1 --posQuantscale=1
    """
    import argparse 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model settings
    parser.add_argument("--stage", type=int, default=8)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--enc_type", type=str, default='pooling')
    #
    parser.add_argument("--voxel_size", type=float, default=1)
    parser.add_argument("--ckptdir", default='')
    parser.add_argument("--outdir",type=str, default='output')
    parser.add_argument("--filedir", default='../dataset/KITTI/pc_q1mm/test100/11/velodyne/000040.h5')
    parser.add_argument("--resolution", type=int, default=30000)
    parser.add_argument("--posQuantscale", type=int, default=16)
    args = parser.parse_args()
    args.outdir = os.path.join(rootdir, args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    # load model
    model = PCCModel(stage=args.stage, kernel_size=args.kernel_size, 
                    scale=args.scale, enc_type=args.enc_type).to(device)
    assert os.path.exists(args.ckptdir)
    ckpt = torch.load(args.ckptdir)
    model.load_state_dict(ckpt['model'])
    # print(model)

    # load data 
    assert os.path.exists(args.filedir)
    x = load_sparse_tensor(args.filedir, voxel_size=args.voxel_size, device=device)
    # downsacle
    x_down = quantize_sparse_tensor(x, factor=1/args.posQuantscale, quant_mode='round')
    # test
    model.test(x_down, scale=args.scale)
    x_dec = quantize_sparse_tensor(x_down, factor=args.posQuantscale)
    # psnr
    write_ply_o3d(os.path.join(args.outdir, 'input.ply'), coords=x.C.detach().cpu().numpy()[:,1:])
    write_ply_o3d(os.path.join(args.outdir, 'dec.ply'), coords=x_dec.C.detach().cpu().numpy()[:,1:])
    results_pc_error = pc_error(os.path.join(args.outdir,'input.ply'), 
                                os.path.join(args.outdir,'dec.ply'), resolution=args.resolution, show=False)
    print('PSNR:\t', results_pc_error['mseF,PSNR (p2point)'])