import os, sys
rootdir = os.path.split(__file__)[0]
sys.path.append(rootdir)
rootdir = os.path.split(rootdir)[0]
sys.path.append(rootdir)
import os, time
import torch
import MinkowskiEngine as ME
import numpy as np
from data_utils.data_loader import load_sparse_tensor
from data_utils.quantize import quantize_sparse_tensor
from data_utils.quantize import quantize_precision, dequantize_precision
from data_utils.quantize import quantize_resolution, dequantize_resolution
from data_utils.quantize import quantize_octree, dequantize_octree
from data_utils.inout import read_ply_o3d, write_ply_o3d, read_coords
from data_utils.sparse_tensor import sort_sparse_tensor
from extension.metrics import pc_error, get_PSNR_VCN, get_PSNR_attn


class BasicCoder():
    """basic lossless coder
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        assert self.model.enc_type=='pooling'
        self.min_num = 32
        self.device = device
    
    def load_data(self, filedir, voxel_size=1, posQuantscale=1):
        """load data & pre-quantize if posQuantscale>1
        """
        x_raw = load_sparse_tensor(filedir, voxel_size=voxel_size, device=self.device)
        x = quantize_sparse_tensor(x_raw, factor=1/posQuantscale, quant_mode='round')
        if x.C.min() < 0:
            ref_point = x.C.min(axis=0)[0]
            print('DBG!!! min_points', ref_point.cpu().numpy())
            x = ME.SparseTensor(features=x.F, coordinates=x.C - ref_point, 
                                tensor_stride=x.tensor_stride, device=x.device)
        else: ref_point = None
        # self.filename = os.path.split(filedir)[-1].split('.')[0]
        self.ref_point = ref_point# TODO

        return x

    @torch.no_grad()
    def encode(self, x):
        bitstream_list = []
        while len(x) > self.min_num:
            x_low = self.model.downsampler(x)
            bitstream = self.model.upsampler.encode(x_low, x)
            # print('DBG!!! bitstream:\t', len(bitstream)*8/len(x))
            bitstream_list.append(bitstream)
            x = ME.SparseTensor(features=torch.ones((len(x_low),1)).float(),
                                coordinates=torch.div(x_low.C,2,rounding_mode='floor'), 
                                device=x_low.device)
        coords = x.C[:,1:].cpu().numpy().astype('int16')
        bitstream_coords = self.num2bits(coords)
        bitstream_list.append(bitstream_coords)

        return bitstream_list

    @torch.no_grad()
    def decode(self, bitstream_list):
        bitstream_list = bitstream_list[::-1]
        bitstream_coords = bitstream_list[0]
        coords = self.bit2num(bitstream_coords).astype('int')
        coords = torch.tensor(coords).int()
        feats = torch.ones((len(coords),1)).float()
        coords, feats = ME.utils.sparse_collate([coords], [feats])
        x = ME.SparseTensor(features=feats, coordinates=coords, 
                            tensor_stride=1, device=self.device)
        for idx, bitstream in enumerate(bitstream_list[1:]):
            x = ME.SparseTensor(features=x.F, coordinates=x.C*2,
                                tensor_stride=2, device=x.device)
            x = self.model.upsampler.decode(x, bitstream)
        # if self.ref_point is not None:
        #     x = ME.SparseTensor(features=x.F, coordinates=x.C + self.ref_point, 
        #                         tensor_stride=x.tensor_stride, device=x.device)

        return x

    def num2bits(self, coords, dtype='int16'):
        min_v = coords.min(axis=0)
        coords = coords - min_v
        bitstream = np.array(min_v, dtype=dtype).tobytes()
        bitstream += coords.tobytes()

        return bitstream

    def bit2num(self, bitstream, dtype='int16'):
        s = 0
        min_v = np.frombuffer(bitstream[s:s+3*2], dtype=dtype).reshape(-1,3)
        s += 3*2
        coords = np.frombuffer(bitstream[s:], dtype=dtype).reshape(-1,3)
        coords = coords + min_v

        return coords

    def pack_bitstream(self, bitstream_list, bin_dir, dtype='uint32'):
        bitstream_all = np.array(len(bitstream_list), dtype=dtype).tobytes()
        bitstream_all += np.array([len(bitstream) for bitstream in bitstream_list], dtype=dtype).tobytes()
        for bitstream in bitstream_list:
            assert len(bitstream)<2**32-1
            bitstream_all += bitstream
        with open(bin_dir, 'wb') as f: f.write(bitstream_all)

        return os.path.getsize(bin_dir)*8
        
    def unpack_bitstream(self, bin_dir, dtype='uint32'):
        with open(bin_dir, 'rb') as fin: bitstream_all = fin.read()
        s = 0
        num = np.frombuffer(bitstream_all[s:s+1*4], dtype=dtype)[0]
        s += 1*4
        lengths = np.frombuffer(bitstream_all[s:s+num*4], dtype=dtype)
        s += num*4
        bitstream_list = []
        for l in lengths:
            bitstream = bitstream_all[s:s+l]
            bitstream_list.append(bitstream)
            s += l

        return bitstream_list

    def test(self, filedir, bin_dir, dec_dir, voxel_size=1, posQuantscale=1):
        start = time.time()
        x = self.load_data(filedir, voxel_size, posQuantscale)
        num_points_raw = x.shape[0]
        start_enc = time.time()
        bitstream_list = self.encode(x)
        enc_time = round(time.time() - start_enc, 3)
        # print('DBG!!! bpp_list', [round(8*len(bitstream)/num_points_raw, 3) for bitstream in bitstream_list])
        bits = self.pack_bitstream(bitstream_list, bin_dir)
        all_enc_time = round(time.time() - start, 3)
        # print('DBG!!! all_enc_time', enc_time, all_enc_time)
        bpp = round(bits / num_points_raw, 3)

        if self.model.stage==0:
            return {'filedir':filedir, 'num_points_raw':num_points_raw, 'file_size':bits, 'bpp':bpp, 'enc_time':enc_time, 'all_enc_time':all_enc_time}

        start = time.time()
        bitstream_list_dec = self.unpack_bitstream(bin_dir)
        start_dec = time.time()
        x_dec = self.decode(bitstream_list_dec)
        dec_time = round(time.time() - start_dec, 3)
        write_ply_o3d(dec_dir, x_dec.C[:,1:].cpu().numpy(), dtype='int32')
        all_dec_time = round(time.time() - start, 3)
        # print('DBG!!! all_dec_time', dec_time, all_dec_time)
        # assert (sort_sparse_tensor(x).C==sort_sparse_tensor(x_dec).C).all()
        if not sort_sparse_tensor(x).C.shape[0]==sort_sparse_tensor(x_dec).C.shape[0]:
            print('MISMATCH!!!!!\nMISMATCH\n!!!!!MISMATCH\n!!!!!MISMATCH!!!!!')
            print(x.C.shape, x_dec.C.shape)
        # else:
        #     print('True\nTrue\nTrue\nTrue\nTrue')

        print("DBG!!! GPU memoey:\t", round(torch.cuda.max_memory_allocated()/1024**3,2),'GB')

        results = {'filedir':filedir, 'num_points_raw':num_points_raw, 'file_size':bits, 'bpp':bpp, 
                'enc_time':enc_time, 'all_enc_time':all_enc_time, 'dec_time':dec_time, 'all_dec_time':all_dec_time}

        return results


class BasicCoder2(BasicCoder):
    def __init__(self, model_low, model_high, device='cuda'):
        self.model_low = model_low
        self.model_high = model_high
        self.min_num = 32
        self.threshold = 7000#!!!
        self.device = device
    
    @torch.no_grad()
    def encode(self, x):
        bitstream_list = []
        while len(x) > self.min_num:
            max_value = (x.C.max(dim=0)[0] - x.C.min(dim=0)[0]).max().cpu()
            if max_value > self.threshold:
                self.model = self.model_high
            else:
                self.model = self.model_low
            x_low = self.model.downsampler(x)
            bitstream = self.model.upsampler.encode(x_low, x)
            bitstream_list.append(bitstream)
            x = ME.SparseTensor(features=torch.ones((len(x_low),1)).float(),
                                coordinates=torch.div(x_low.C,2,rounding_mode='floor'), 
                                device=x_low.device)
        coords = x.C[:,1:].cpu().numpy().astype('int16')
        bitstream_coords = self.num2bits(coords)
        bitstream_list.append(bitstream_coords)

        return bitstream_list

    @torch.no_grad()
    def decode(self, bitstream_list):
        bitstream_list = bitstream_list[::-1]
        bitstream_coords = bitstream_list[0]
        coords = self.bit2num(bitstream_coords).astype('int')
        coords = torch.tensor(coords).int()
        feats = torch.ones((len(coords),1)).float()
        coords, feats = ME.utils.sparse_collate([coords], [feats])
        x = ME.SparseTensor(features=feats, coordinates=coords, 
                            tensor_stride=1, device=self.device)
        for idx, bitstream in enumerate(bitstream_list[1:]):
            x = ME.SparseTensor(features=x.F, coordinates=x.C*2,
                                tensor_stride=2, device=x.device)
            max_value = (x.C.max(dim=0)[0] - x.C.min(dim=0)[0]).max().cpu()
            if max_value > self.threshold: 
                self.model = self.model_high
            else:
                self.model = self.model_low
            x = self.model.upsampler.decode(x, bitstream)
        # if self.ref_point is not None:
        #     x = ME.SparseTensor(features=x.F, coordinates=x.C + self.ref_point, 
        #                         tensor_stride=x.tensor_stride, device=x.device)

        return x


class LossyCoderSparse():
    """for sparse point clouds
    """
    def __init__(self, basic_coder, model_offset=None, device='cuda'):
        self.basic_coder = basic_coder
        self.model_offset = model_offset
        self.device = device

    @torch.no_grad()
    def downscale(self, x, posQuantscale=1):
        x_down = quantize_sparse_tensor(x, factor=1/posQuantscale, quant_mode='round')
        
        return x_down
    
    @torch.no_grad()
    def upscale(self, x, posQuantscale):
        if self.model_offset is None:
            x_dec = quantize_sparse_tensor(x, factor=posQuantscale)
            coords_dec = x_dec.C[:,1:].cpu().numpy()
        else:
            coords_dec = self.model_offset.upscale(x, posQuantscale=posQuantscale)

        return coords_dec

    def prequantize(self, points_raw, quant_mode=None, quant_factor=1):
        if quant_mode is None:
            points_q = points_raw
        elif quant_mode=='precision':
            points_q = quantize_precision(points_raw, precision=quant_factor, return_offset=False)
            meta_info = {}
        elif quant_mode=='resolution':
            points_q, max_bound, min_bound = quantize_resolution(points_raw, resolution=quant_factor, return_offset=False)
            meta_info = {'max_bound':max_bound, 'min_bound':min_bound}
        elif quant_mode=='octree':
            points_q, min_bound, max_bound, centroid = quantize_octree(points_raw, qlevel=quant_factor, return_offset=False)
            meta_info = {'min_bound':min_bound, 'max_bound':max_bound, 'centroid':centroid}
        points_q = np.unique(points_q, axis=0).astype('int32')
        if points_q.min() < 0:
            ref_point = points_q.min(axis=0)
            print('DBG!!! min_points', ref_point)
            points_q = points_q - ref_point
        else: ref_point = None
        meta_info['ref_point'] = ref_point

        return points_q, meta_info

    def postquantize(self, points_q, meta_info, quant_mode=None, quant_factor=1):
        ref_point = meta_info['ref_point']
        if ref_point is not None:
            points_q = points_q + ref_point
        if quant_mode is None:
            points = points_q
        elif quant_mode=='precision':
            points = dequantize_precision(points_q, precision=quant_factor)
        elif quant_mode=='resolution':
            max_bound, min_bound = meta_info['max_bound'], meta_info['min_bound']
            points = dequantize_resolution(points_q, max_bound=max_bound, min_bound=min_bound, resolution=quant_factor)        
        elif quant_mode=='octree':
            min_bound, max_bound, centroid = meta_info['min_bound'], meta_info['max_bound'], meta_info['centroid']
            points = dequantize_octree(points_q, min_bound=min_bound, max_bound=max_bound, centroid=centroid, qlevel=quant_factor)

        return points

    def test(self, filedir, bin_dir, dec_dir, posQuantscale=1, quant_mode=None, quant_factor=1, psnr_mode='gpcc', test_d2=False):
        start = time.time()
        points_raw = read_coords(filedir)
        num_points_raw = points_raw.shape[0]
        # preprocessing: quantize if input raw float points
        points_q, meta_info = self.prequantize(points_raw, quant_mode=quant_mode, quant_factor=quant_factor)
        start_enc = time.time()
        # sparse tensor
        coords = torch.tensor(points_q).int()
        feats = torch.ones((len(points_q),1)).float()
        coords, feats = ME.utils.sparse_collate([coords], [feats])
        x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=self.device)
        # downscale
        x_down = self.downscale(x, posQuantscale)
        # encode
        bitstream_list = self.basic_coder.encode(x_down)
        # print('DBG!!! bpp_list', [round(8*len(bitstream)/num_points_raw, 3) for bitstream in bitstream_list])
        enc_time = round(time.time() - start_enc, 3)
        bits = self.basic_coder.pack_bitstream(bitstream_list, bin_dir)
        all_enc_time = round(time.time() - start, 3)
        # print('enc time:\t', round(time.time() - start, 3))
        start = time.time()
        # decode
        bitstream_list_dec = self.basic_coder.unpack_bitstream(bin_dir)
        start_dec = time.time()
        x_dec = self.basic_coder.decode(bitstream_list_dec)
        # assert (sort_sparse_tensor(x_down).C==sort_sparse_tensor(x_dec).C).all()
        # upscale
        coords_dec = self.upscale(x_dec, posQuantscale)
        # postprocessing: dequantize if input raw float points
        points_dec = self.postquantize(coords_dec, meta_info, quant_mode=quant_mode, quant_factor=quant_factor)
        # print('DBG!!! points_dec', points_dec.max().round(), points_dec.min().round())
        # print('DBG!!! points_ori', points_raw.max().round(), points_raw.min().round())
        dec_time = round(time.time() - start_dec, 3)
        write_ply_o3d(dec_dir, points_dec, dtype='float32')
        all_dec_time = round(time.time() - start, 3)
        bpp = round(bits / num_points_raw, 3)
        print('bpp:\t', bpp, num_points_raw)
        print("memoey:\t", round(torch.cuda.max_memory_allocated()/1024**3,2),'GB')
        # metric psnr
        ref_dir = dec_dir[:-4]+'_ref.ply'
        if test_d2:
            write_ply_o3d(ref_dir, points_raw, normal=True, knn=16)# test d2
        else:
            write_ply_o3d(ref_dir, points_raw, dtype='float32')

        if psnr_mode=='vcn':
            psnr_results = get_PSNR_VCN(ref_dir, dec_dir)
        elif psnr_mode=='attn':
            psnr_results = get_PSNR_attn(ref_dir, dec_dir, test_d2=test_d2)
        elif psnr_mode=='gpcc':
            psnr_results = pc_error(ref_dir, dec_dir, resolution=30000, normal=test_d2, show=False)
        print('psnr:\t', psnr_results)

        results = {'filedir':filedir, 'num_points_raw':num_points_raw, 'file_size':bits, 'bpp':bpp, 'num_points':points_dec.shape[0],
                'enc_time':enc_time, 'all_enc_time':all_enc_time, 'dec_time':dec_time, 'all_dec_time':all_dec_time}
        results.update(psnr_results)

        return results
    

class LossyCoderDense():
    """for dense point clouds
    """
    def __init__(self, basic_coder, model_AE, model_SR, device='cuda'):
        self.basic_coder = basic_coder
        self.model_AE = model_AE
        self.model_SR = model_SR
        self.device = device

    @torch.no_grad()
    def downscale(self, x, scale_AE=1, scale_SR=2):
        num_points_raw = x.shape[0]
        # Lossy
        num_points_list = []
        for idx in range(scale_SR):
            num_points_list.append(x.shape[0])
            x = self.model_SR.downsampler(x)
            x = ME.SparseTensor(features=torch.ones((len(x),1)).float(),
                                coordinates=torch.div(x.C,2,rounding_mode='floor'), device=x.device)
        if scale_AE==1:
            num_points_list.append(x.shape[0])
            x, bitstream = self.model_AE.downsampler.encode(x, return_one=True)
            x = ME.SparseTensor(features=torch.ones((len(x),1)).float(),
                                coordinates=torch.div(x.C,2,rounding_mode='floor'), device=x.device)
        else: bitstream = None

        return x, bitstream, num_points_list

    @torch.no_grad()
    def upscale(self, x, bitstream, num_points_list):
        num_points_list = num_points_list[::-1]
        if bitstream is not None:
            x = ME.SparseTensor(features=x.F, coordinates=x.C*2,
                                tensor_stride=2, device=x.device)
            x = self.model_AE.downsampler.decode(x, bitstream)
            num_points = num_points_list[0]
            num_points_list = num_points_list[1:]
            x = self.model_AE.upsampler.upsample(x, num_points)
        # print('DBG!!! AE', x.C.max(), num_points_list)

        for idx, num_points in enumerate(num_points_list):
            x = ME.SparseTensor(features=x.F, coordinates=x.C*2,
                                tensor_stride=2, device=x.device)
            x = self.model_SR.upsampler.upsample(x, num_points)
            # print('DBG!!! SR', x.C.max(), num_points)

        return x

    def test(self, filedir, bin_dir, dec_dir, voxel_size=1, posQuantscale=1, scale_AE=1, scale_SR=2, psnr_resolution=1023):
        start = time.time()
        x_raw = self.basic_coder.load_data(filedir, voxel_size, posQuantscale)
        num_points_raw = x_raw.shape[0]
        start_enc = time.time()
        # downscale
        x, bitstream_AE, num_points_list = self.downscale(x_raw, scale_AE=scale_AE, scale_SR=scale_SR)
        if bitstream_AE is not None: 
            print('DBG! bpp (AE)\t', round(8*len(bitstream_AE)/num_points_raw, 3))
        # encode
        bitstream_list = self.basic_coder.encode(x)
        enc_time = round(time.time() - start_enc, 3)
        # print('DBG!!! bpp_list', [round(8*len(bitstream)/num_points_raw, 3) for bitstream in bitstream_list])
        bits = self.basic_coder.pack_bitstream(bitstream_list, bin_dir)
        all_enc_time = round(time.time() - start, 3)
        # print('enc time:\t', round(time.time() - start, 3))
        start = time.time()
        # decode
        bitstream_list_dec = self.basic_coder.unpack_bitstream(bin_dir)
        start_dec = time.time()
        x_dec = self.basic_coder.decode(bitstream_list_dec)
        # assert (sort_sparse_tensor(x).C==sort_sparse_tensor(x_dec).C).all()
        x_dec = self.upscale(x, bitstream_AE, num_points_list)
        dec_time = round(time.time() - start_dec, 3)
        # print('dec time:\t', round(time.time() - start, 3))
        points_dec = x_dec.C[:,1:].detach().cpu().numpy()
        write_ply_o3d(dec_dir, points_dec, dtype='int32')
        all_dec_time = round(time.time() - start, 3)
        bits += len(num_points_list)*4
        if bitstream_AE is not None:
            bits += len(bitstream_AE)*8
        bpp = round(bits / num_points_raw, 3)
        # print('bpp:\t', bpp, num_points_raw)
        print("memoey:\t", round(torch.cuda.max_memory_allocated()/1024**3,2),'GB')
        # metric psnr
        ref_dir = dec_dir[:-4]+'_ref.ply'
        points_raw = x_raw.C[:,1:].detach().cpu().numpy()
        # print('DBG!!!', points_raw.max(), points_dec.max())
        write_ply_o3d(ref_dir, points_raw, dtype='int32', normal=True, knn=16)
        psnr_results = pc_error(ref_dir, dec_dir, resolution=psnr_resolution, normal=True, show=False)
        # print('psnr:\t', psnr_results)

        results = {'filedir':filedir, 'num_points_raw':num_points_raw, 'file_size':bits, 'bpp':bpp, 'num_points':points_dec.shape[0],
                'enc_time':enc_time, 'all_enc_time':all_enc_time, 'dec_time':dec_time, 'all_dec_time':all_dec_time}
        results.update(psnr_results)

        return results


if __name__ == '__main__':
    """
    python coder.py --kernel_size=3 --ckptdir='ckpts/anchor/dense/epoch_last.pth' --filedir='testdata/longdress_vox10_1300.ply' --voxel_size=1 --posQuantscale=1
    python coder.py --kernel_size=5 --ckptdir='ckpts/anchor/vcn/epoch_last.pth' --filedir='testdata/kitti_seqs11_000000.bin' --voxel_size=0.02 --posQuantscale=1
    
    python coder.py --kernel_size=5 --ckptdir_low='ckpts/anchor/vcn2/epoch_last.pth' --ckptdir_high='ckpts/anchor/sparse_high/epoch_last.pth' \
        --filedir='testdata/kitti_seqs11_000000.bin' --voxel_size=0.001 --posQuantscale=1
    python coder.py --kernel_size=5 --ckptdir_low='ckpts/anchor/vcn2/epoch_last.pth' --ckptdir_high='ckpts/anchor/sparse_high/epoch_last.pth' \
        --filedir='testdata/Ford_01_vox1mm-0100.ply' --voxel_size=1 --posQuantscale=1
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
    parser.add_argument("--posQuantscale", type=int, default=1)
    parser.add_argument("--ckptdir", default='')
    parser.add_argument("--filedir", default='')
    parser.add_argument("--resolution", type=int, default=30000)
    parser.add_argument("--outdir",type=str, default='output')
    #
    parser.add_argument("--ckptdir_low", default='')
    parser.add_argument("--ckptdir_high", default='')
    parser.add_argument("--ckptdir_offset", default='')
    parser.add_argument("--ckptdir_ae", default='')
    parser.add_argument("--ckptdir_sr", default='')

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    #################################### basic coder ####################################
    # from model import PCCModel
    # model = PCCModel(stage=args.stage, kernel_size=args.kernel_size, enc_type='pooling').to(device)
    # assert os.path.exists(args.ckptdir)
    # ckpt = torch.load(args.ckptdir)
    # model.load_state_dict(ckpt['model'])

    # basic_coder = BasicCoder(model, device=device)
    # filename = os.path.split(args.filedir)[-1].split('.')[0]
    # bin_dir = os.path.join(args.outdir, filename+'.bin')
    # basic_coder.test(args.filedir, bin_dir, voxel_size=args.voxel_size, posQuantscale=args.posQuantscale)

    #################################### basic coder2 ####################################
    # from model import PCCModel
    # model_low = PCCModel(stage=8, kernel_size=5, enc_type='pooling').to(device)
    # assert os.path.exists(args.ckptdir_low)
    # ckpt = torch.load(args.ckptdir_low)
    # model_low.load_state_dict(ckpt['model'])

    # from model import PCCModel
    # model_high = PCCModel(stage=8, kernel_size=5, enc_type='pooling').to(device)
    # assert os.path.exists(args.ckptdir_high)
    # ckpt = torch.load(args.ckptdir_high)
    # model_high.load_state_dict(ckpt['model'])

    # basic_coder2 = BasicCoder2(model_low, model_high, device=device)
    # filename = os.path.split(args.filedir)[-1].split('.')[0]
    # bin_dir = os.path.join(args.outdir, filename+'.bin')
    # basic_coder2.test(args.filedir, bin_dir, voxel_size=args.voxel_size, posQuantscale=args.posQuantscale)

    #################################### LossyCoderSparse ####################################
    # from model_offset import OffsetModel
    # model_offset = OffsetModel(kernel_size=5).to(device)
    # assert os.path.exists(args.ckptdir_offset)
    # ckpt = torch.load(args.ckptdir_offset)
    # model_offset.load_state_dict(ckpt['model'])

    # # lossy_coder_sparse = LossyCoderSparse(basic_coder2, model_offset=None)
    # lossy_coder_sparse = LossyCoderSparse(basic_coder2, model_offset=model_offset)
    ########################## 
    # mode 1: input Ford_q1mm, adjust posQuantScale
    """    python coder.py --kernel_size=5 --ckptdir_low='ckpts/anchor/vcn2/epoch_last.pth' \
                                            --ckptdir_high='ckpts/anchor/vcn2/epoch_last.pth' \
                                            --ckptdir_offset='ckpts/anchor/offset/epoch_last.pth' \
                                            --filedir='testdata/Ford_01_vox1mm-0100.ply'
    """
    
    # assert args.filedir.endswith('ply')
    # posQuantscale_list = [4, 8, 32, 64, 256, 512]
    # filename = os.path.split(args.filedir)[-1].split('.')[0]
    # results_list = []
    # for i, posQuantscale in enumerate(posQuantscale_list):
    #     bin_dir = os.path.join(args.outdir, filename+'_R'+str(i)+'.bin')
    #     dec_dir = os.path.join(args.outdir, filename+'_R'+str(i)+'.ply')
    #     results = lossy_coder_sparse.test(args.filedir, bin_dir, dec_dir, posQuantscale=posQuantscale, 
    #                             quant_mode='precision', quant_factor=1, psnr_mode='gpcc')
    #     results_list.append(results)
    # print(results_list)


    ########################## 
    # mode 2: input kitti_raw, adjust resolution
    """    python coder.py --kernel_size=5 --ckptdir_low='ckpts/anchor/vcn2/epoch_last.pth' \
                                            --ckptdir_high='ckpts/anchor/vcn2/epoch_last.pth' \
                                            --ckptdir_offset='ckpts/anchor/offset/epoch_last.pth' \
                                            --filedir='testdata/kitti_seqs11_000000.bin'
    """
    # assert args.filedir.endswith('bin')
    # resolution_list = [2**12-1, 2**11-1, 2**10-1, 2**9-1]
    # filename = os.path.split(args.filedir)[-1].split('.')[0]
    # results_list = []
    # for i, resolution in enumerate(resolution_list):
    #     bin_dir = os.path.join(args.outdir, filename+'_R'+str(i)+'.bin')
    #     dec_dir = os.path.join(args.outdir, filename+'_R'+str(i)+'.ply')
    #     results = lossy_coder_sparse.test(args.filedir, bin_dir, dec_dir, posQuantscale=1, 
    #                             quant_mode='resolution', quant_factor=resolution, psnr_mode='vcn')
    #     results_list.append(results)
    # print(results_list)


    ########################## 
    # mode 3: input kitti_raw, adjust bits
    """    python coder.py --kernel_size=5 --ckptdir_low='ckpts/anchor/vcn2/epoch_last.pth' \
                                            --ckptdir_high='ckpts/anchor/vcn2/epoch_last.pth' \
                                            --ckptdir_offset='ckpts/anchor/offset/epoch_last.pth' \
                                            --filedir='testdata/kitti_seqs11_000000.bin'
    """
    ##########################
    # assert args.filedir.endswith('bin')
    # qlevel_list = [12,11,10,9]
    # filename = os.path.split(args.filedir)[-1].split('.')[0]
    # results_list = []
    # for i, qlevel in enumerate(qlevel_list):
    #     bin_dir = os.path.join(args.outdir, filename+'_R'+str(i)+'.bin')
    #     dec_dir = os.path.join(args.outdir, filename+'_R'+str(i)+'.ply')
    #     results = lossy_coder_sparse.test(args.filedir, bin_dir, dec_dir, posQuantscale=1, 
    #                             quant_mode='octree', quant_factor=qlevel, psnr_mode='attn')
    #     results_list.append(results)
    # print(results_list)

    ################################################# lossy Dense #################################################
    """python coder.py --ckptdir='ckpts/anchor/dense/epoch_last.pth' \
                        --ckptdir_sr='ckpts/anchor/sr/epoch_last.pth' \
                        --ckptdir_ae='ckpts/anchor/ae/epoch_last.pth' \
                        --filedir='testdata/longdress_vox10_1300.ply'
    """
    from models.model import PCCModel
    model = PCCModel(stage=8, kernel_size=3, enc_type='pooling').to(device)
    assert os.path.exists(args.ckptdir)
    ckpt = torch.load(args.ckptdir)
    model.load_state_dict(ckpt['model'])

    basic_coder = BasicCoder(model, device=device)
    filename = os.path.split(args.filedir)[-1].split('.')[0]
    bin_dir = os.path.join(args.outdir, filename+'.bin')
    basic_coder.test(args.filedir, bin_dir, voxel_size=args.voxel_size, posQuantscale=args.posQuantscale)

    from models.model import PCCModel
    model_SR = PCCModel(stage=1, kernel_size=3, enc_type='pooling').to(device)
    assert os.path.exists(args.ckptdir_sr)
    ckpt = torch.load(args.ckptdir_sr)
    model_SR.load_state_dict(ckpt['model'])

    from models.model import PCCModel
    model_AE = PCCModel(stage=1, kernel_size=3, enc_type='ae').to(device)
    assert os.path.exists(args.ckptdir_ae)
    ckpt = torch.load(args.ckptdir_ae)
    model_AE.load_state_dict(ckpt['model'])

    lossy_coder_dense = LossyCoderDense(basic_coder, model_AE, model_SR, device=device)
    filename = os.path.split(args.filedir)[-1].split('.')[0]
    results_list = []
    scale_AE_list = [1,0,1,0,1,0]
    scale_SR_list = [0,1,1,2,2,3]
    idx_rate = 0
    for scale_AE, scale_SR in zip(scale_AE_list, scale_SR_list):
        bin_dir = os.path.join(args.outdir, filename+'_R'+str(idx_rate)+'.bin')
        dec_dir = os.path.join(args.outdir, filename+'_R'+str(idx_rate)+'.ply')
        idx_rate += 1
        results = lossy_coder_dense.test(args.filedir, bin_dir, dec_dir,
                            scale_AE=scale_AE, scale_SR=scale_SR, psnr_resolution=1023)
        results_list.append(results)
    print(results_list)