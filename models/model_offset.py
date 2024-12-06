import os, sys
rootdir = os.path.split(__file__)[0]
sys.path.append(rootdir)
rootdir = os.path.split(rootdir)[0]
sys.path.append(rootdir)
import torch
import MinkowskiEngine as ME
from resnet import ResNetBlock
from extension.tools import pc_error
from data_utils.quantize import quantize_sparse_tensor
from data_utils.inout import write_ply_o3d
from data_utils.data_loader import load_sparse_tensor


class OffsetDecoder(torch.nn.Module):
    """
    """
    def __init__(self, in_channels=1, channels=128, out_channels=3, kernel_size=3, block_layers=3):
        super().__init__()
        self.block = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels, out_channels=channels,
                kernel_size=kernel_size, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=True),
            ResNetBlock(
                block_layers=block_layers, channels=channels, 
                kernel_size=kernel_size),
             ME.MinkowskiConvolution(
                in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, stride=1, bias=True, dimension=3))
        #
        self.fc = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=channels, out_channels=channels,
                kernel_size=1, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                in_channels=channels, out_channels=channels,
                kernel_size=1, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                in_channels=channels, out_channels=out_channels,
                kernel_size=1, stride=1, bias=True, dimension=3))
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        out = self.fc(out)

        return out


class OffsetModel(torch.nn.Module):
    def __init__(self, channels=128, kernel_size=3, block_layers=3, posQuantscale_list=None):
        super().__init__()
        self.posQuantscale_list = posQuantscale_list
        self.offset_decoder = OffsetDecoder(
            in_channels=1, channels=channels, out_channels=3, kernel_size=kernel_size,
            block_layers=block_layers)

    def forward(self, x, posQuantscale_list=None):
        out_set_list = []
        if posQuantscale_list is not None: 
            self.posQuantscale_list = posQuantscale_list
        for idx, posQuantscale in enumerate(self.posQuantscale_list):
            x_offset = quantize_sparse_tensor(x, factor=1/posQuantscale, 
                                            return_offset=True, quant_mode='round')
            x_one = ME.SparseTensor(
                features=torch.ones([x_offset.C.shape[0], 1]), 
                coordinate_map_key=x_offset.coordinate_map_key, 
                coordinate_manager=x_offset.coordinate_manager, 
                device=x_offset.device)
            out = self.offset_decoder(x_one)
            # print('DBG!!!', idx, '\t', posQuantscale, len(x_offset), x_offset.C.max().cpu().numpy())
            out_set_list.append({'ground_truth':x_offset, 'out':out})

        return out_set_list

    @torch.no_grad()
    def downscale(self, ground_truth, posQuantscale=None):
        out = quantize_sparse_tensor(ground_truth, factor=1/posQuantscale, 
                                    return_offset=True, quant_mode='round')
        out = ME.SparseTensor(
            features=torch.ones([out.C.shape[0], 1]), 
            coordinate_map_key=out.coordinate_map_key, 
            coordinate_manager=out.coordinate_manager, 
            device=out.device)

        return out

    @torch.no_grad()
    def upscale(self, x, posQuantscale=1):
        out = self.offset_decoder(x)
        offset = out.F.float().cpu()
        coords = out.C[:,1:].float().cpu()
        coords = coords + offset
        coords = coords * posQuantscale

        return coords.numpy()



class OffsetModelVCN(torch.nn.Module):
    def __init__(self, channels=128, kernel_size=3, block_layers=3, posQuantscale_list=None):
        super().__init__()
        self.offset_decoder = OffsetDecoder(
            in_channels=1, channels=channels, out_channels=3, kernel_size=kernel_size,
            block_layers=block_layers)

    def forward(self, x):
        x_one = ME.SparseTensor(
            features=torch.ones([x.C.shape[0], 1]), 
            coordinate_map_key=x.coordinate_map_key, 
            coordinate_manager=x.coordinate_manager, 
            device=x.device)
        out = self.offset_decoder(x_one)
        out_set_list =[{'ground_truth':x, 'out':out}]

        return out_set_list

    # @torch.no_grad()
    # def downscale(self, ground_truth, posQuantscale=None):
    #     out = quantize_sparse_tensor(ground_truth, factor=1/posQuantscale, 
    #                                 return_offset=True, quant_mode='round')
    #     out = ME.SparseTensor(
    #         features=torch.ones([out.C.shape[0], 1]), 
    #         coordinate_map_key=out.coordinate_map_key, 
    #         coordinate_manager=out.coordinate_manager, 
    #         device=out.device)

    #     return out

    # @torch.no_grad()
    # def upscale(self, x, posQuantscale=1):
    #     out = self.offset_decoder(x)
    #     offset = out.F.float().cpu()
    #     coords = out.C[:,1:].float().cpu()
    #     coords = coords + offset
    #     coords = coords * posQuantscale

    #     return coords.numpy()


if __name__ == '__main__':
    import argparse, os
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model settings
    parser.add_argument("--ckptdir", default='../ckpts/anchor/offset/epoch_last.pth')
    parser.add_argument("--filedir", default='../testdata/kitti_q1mm_000040.h5')
    parser.add_argument("--outdir",type=str, default='output')
    parser.add_argument("--resolution", type=int, default=30000)
    parser.add_argument("--posQuantscale", type=int, default=64)
    parser.add_argument("--voxel_size", type=int, default=1)
    args = parser.parse_args()
    args.outdir = os.path.join(rootdir, args.outdir)
    os.makedirs(args.outdir, exist_ok=True)
    
    # load model
    model = OffsetModel(channels=128, block_layers=3, kernel_size=5,
                        posQuantscale_list=None).to(device)
    assert os.path.exists(args.ckptdir)
    ckpt = torch.load(args.ckptdir)
    model.load_state_dict(ckpt['model'])
    
    # load data
    assert os.path.exists(args.filedir)
    x = load_sparse_tensor(args.filedir, voxel_size=args.voxel_size, device=device)
    x_down = model.downscale(x, args.posQuantscale)
    x_up = quantize_sparse_tensor(x_down, factor=args.posQuantscale)

    coords_dec = model.upscale(x_down, args.posQuantscale)
    print("memoey!:\t", round(torch.cuda.max_memory_allocated()/1024**3,3),'GB')
    input_dir = os.path.join(args.outdir, 'input.ply')
    up_dir = os.path.join(args.outdir, 'up.ply')
    dec_dir = os.path.join(args.outdir, 'dec.ply')
    write_ply_o3d(input_dir, coords=x.C.detach().cpu().numpy()[:,1:])
    write_ply_o3d(up_dir, coords=x_up.C.detach().cpu().numpy()[:,1:])
    write_ply_o3d(dec_dir, coords=coords_dec)
    results_pc_error = pc_error(input_dir, up_dir, resolution=args.resolution, show=False)
    print('PSNR (baseline):\t', results_pc_error['mseF,PSNR (p2point)'])
    results_pc_error = pc_error(input_dir, dec_dir, resolution=args.resolution, show=False)
    print('PSNR:\t', results_pc_error['mseF,PSNR (p2point)'])
