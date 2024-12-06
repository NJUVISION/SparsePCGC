import os, sys
rootdir = os.path.split(__file__)[0]
sys.path.append(rootdir)
rootdir = os.path.split(rootdir)[0]
sys.path.append(rootdir)
import torch
import MinkowskiEngine as ME
import numpy as np
from resnet import ResNetBlock
from data_utils.sparse_tensor import isin, istopk, create_new_sparse_tensor, sort_sparse_tensor
import torchac


class UpSampler(torch.nn.Module):
    def __init__(self, in_channels=1, channels=32, out_channels=32, kernel_size=3, 
                block_layers0=3, block_layers1=3,  
                deconv_kernel_size=2, stride=2, expand_coordinates=True):
        super().__init__()
        self.block0 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels, out_channels=channels,
                kernel_size=kernel_size, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=True),
            ResNetBlock(channels=channels, kernel_size=kernel_size, block_layers=block_layers0))
        self.deconv = ME.MinkowskiConvolutionTranspose(
            in_channels=channels, out_channels=channels,
            kernel_size=deconv_kernel_size, stride=stride, expand_coordinates=expand_coordinates, 
            bias=True, dimension=3)
        self.block1 = torch.nn.Sequential(
            ResNetBlock(channels=channels, kernel_size=kernel_size, block_layers=block_layers1),
            ME.MinkowskiConvolution(
                in_channels=channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=1, bias=True, dimension=3))
        self.relu = ME.MinkowskiReLU(inplace=True)
    
    def forward(self, x):
        out = self.block0(x)
        out = self.relu(self.deconv(out))
        out = self.block1(out)

        return out

class Classifier(torch.nn.Module):
    """calculate the probability of voxel occupied.
        Todo: rename to MLP or FC or Linear Layer.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels, out_channels=channels,
            kernel_size=1, stride=1, bias=True, dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels, out_channels=channels,
            kernel_size=1, stride=1, bias=True, dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels, out_channels=1,
            kernel_size=1, stride=1, bias=True, dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.relu(self.conv1(out))
        out = self.conv2(out)

        return out


class UpSampler1Stage(torch.nn.Module):
    def __init__(self, in_channels=1, channels=32, kernel_size=3, block_layers=3, stage=1):
        super().__init__()
        self.upsampler = UpSampler(
                in_channels=in_channels, channels=channels, out_channels=channels, kernel_size=kernel_size, 
                block_layers0=block_layers, block_layers1=block_layers, 
                deconv_kernel_size=[2,2,2], stride=[2,2,2])
        self.classifier = Classifier(channels)
        self.pruning = ME.MinkowskiPruning()


    def forward(self, x_low, x_high):
        out = self.upsampler(x_low)
        out_cls = self.classifier(out)

        return {'out_cls_list': [out_cls], 
                'ground_truth_list': [x_high]}
    
    @torch.no_grad()
    def upsample(self, x_low, num_points):
        out = self.upsampler(x_low)
        out_cls = self.classifier(out)
        mask = istopk(out_cls, [num_points])
        out = self.pruning(out, mask.to(out.device))
        out = ME.SparseTensor(torch.ones([out.shape[0], 1]).float(), 
                            coordinate_map_key=out.coordinate_map_key, 
                            coordinate_manager=out.coordinate_manager, 
                            device=out.device)

        return out

    @torch.no_grad()
    def encode(self, x_low, x_high, DBG=True):
        out_set = self.forward(x_low, x_high)
        out_cls = out_set['out_cls_list'][0]
        out_cls = sort_sparse_tensor(out_cls)
        prob = torch.sigmoid(out_cls.F).detach().cpu()
        occupancy = isin(out_cls.C, x_high.C).short().cpu()
        BAC = BinaryArithmeticCoding()
        bitstream = BAC.encode(prob, occupancy)
        
        return bitstream

    @torch.no_grad()
    def decode(self, x_low, bitstream):
        out = self.upsampler(x_low)
        out_cls = self.classifier(out)
        out_cls = sort_sparse_tensor(out_cls)
        prob = torch.sigmoid(out_cls.F).detach().cpu()
        BAC = BinaryArithmeticCoding()
        occupancy = BAC.decode(prob, bitstream)
        ground_truth = self.pruning(out_cls, occupancy.bool().to(out_cls.device))
        ground_truth = ME.SparseTensor(torch.ones([ground_truth.F.shape[0], 1]).float(), 
            coordinate_map_key=ground_truth.coordinate_map_key, 
            coordinate_manager=ground_truth.coordinate_manager, 
            device=ground_truth.device)

        return ground_truth

    @torch.no_grad()
    def test(self, x_low, x_high):
        # real bitstream
        bitstream = self.encode(x_low, x_high)
        bits = len(bitstream)*8
        x_high_rec = self.decode(x_low, bitstream)
        assert (sort_sparse_tensor(x_high).C==sort_sparse_tensor(x_high_rec).C).all()
        # print('DBG!!!', bits)

        return bits



########################################################################
class UpSampler8Stage(torch.nn.Module):
    def __init__(self, in_channels=1, channels=32, kernel_size=3, block_layers=3, stage=8):
        super().__init__()
        self.stage = stage
        self.block_in = torch.nn.Sequential(
            self.make_block(in_channels=in_channels, channels=channels, out_channels=channels, 
                            kernel_size=kernel_size, block_layers=block_layers),
            ME.MinkowskiConvolutionTranspose(in_channels=channels, out_channels=channels, kernel_size=[2,2,2], 
                                            stride=[2,2,2], expand_coordinates=True, bias=True, dimension=3))
        self.block_list = torch.nn.ModuleList([
            self.make_block(in_channels=channels, channels=channels, out_channels=channels, 
                            kernel_size=kernel_size, block_layers=block_layers) for i in range(self.stage)])
        self.classifier = Classifier(channels)
        self.pruning = ME.MinkowskiPruning()

    def make_block(self, in_channels=32, channels=32, out_channels=32, kernel_size=3, block_layers=3):
        return torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels, out_channels=channels,
                kernel_size=kernel_size, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=True),
            ResNetBlock(block_layers=block_layers, channels=channels, kernel_size=kernel_size),
            ME.MinkowskiConvolution(
                in_channels=channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=1, bias=True, dimension=3))
    
    def split(self, x):
        device = x.device
        slice_list = []
        octant = torch.sum((x.C[:,1:]%torch.tensor(2).to(device))*torch.tensor([1,2,4]).to(device), axis=1)
        if self.stage==8:
            offset_list = [[0],[7],[1],[6],[5],[2],[3],[4]]
        elif self.stage==3:
            offset_list = [[0,1], [6,7], [2,3,4,5]]
        elif self.stage==2:
            offset_list = [[0,1,6,7], [2,3,4,5]]
        for _, offsets  in enumerate(offset_list):
            mask = torch.sum(torch.stack([octant==a for a in offsets]), axis=0).bool()
            curr_slice = self.pruning(x, mask)
            slice_list.append(curr_slice)

        return slice_list

    def concat(self, x0, x1):
        feats = torch.cat([x0.F, x1.F], dim=0)
        coords=torch.cat([x0.C, x1.C], dim=0)
        out = create_new_sparse_tensor(
            features=feats, 
            coordinates=coords, 
            tensor_stride=x0.tensor_stride, 
            dimension=x0.D, device=x0.device)

        return out

    def basic_module(self, block, classifier, x_in, siblings, ground_truth):
        # concat  x_in and siblings
        if siblings is None:
            inputs = x_in
        else: 
            inputs = self.concat(siblings, x_in)
        # & extract feature
        out = block(inputs)
        # estimate probability
        if siblings is None:
            x_out = out
        else:
            mask_in = isin(out.C, x_in.C)
            x_out = self.pruning(out, mask_in.to(out.device))
        # assert len(x_out)==len(x_in)
        # assert (sort_sparse_tensor(x_out).C==sort_sparse_tensor(x_in).C).all()
        out_cls = classifier(x_out)
        # output ground truth
        mask_out = isin(out.C, ground_truth.C).to(out.device)
        out = self.pruning(out, mask_out)

        return out, out_cls

    def forward(self, x_low, x_high):
        x_low = self.block_in(x_low)
        slice_list = self.split(x_low)
        out_cls_list = []
        out = None
        for block, curr_slice, in zip(self.block_list, slice_list):
            out, out_cls = self.basic_module(block=block, classifier=self.classifier, 
                                            x_in=curr_slice, siblings=out, ground_truth=x_high)
            # print('DBG!!!', out.shape, out_cls.shape)
            out_cls_list.append(out_cls)

        return {'out_cls_list': out_cls_list, 
                'ground_truth_list': [x_high]*8}

    @torch.no_grad()
    def encode(self, x_low, x_high, DBG=True):
        out_set = self.forward(x_low, x_high)
        bitstream_list = []
        for idx, out_cls in enumerate(out_set['out_cls_list']):
            out_cls = sort_sparse_tensor(out_cls)
            prob = torch.sigmoid(out_cls.F).detach().cpu()
            # prob = torch.round(prob*1e3)/1e3
            occupancy = isin(out_cls.C, x_high.C).short().cpu()
            BAC = BinaryArithmeticCoding()
            bitstream = BAC.encode(prob, occupancy)
            bitstream_list.append(bitstream)
            if DBG:
                occupancy_dec = BAC.decode(prob, bitstream)
                assert (occupancy == occupancy_dec).all()

        bitstream = self.pack_bitstream(bitstream_list)        

        return bitstream

    def pack_bitstream(self, bitstream_list, dtype='uint32'):
        bitstream_all = np.array(len(bitstream_list), dtype=dtype).tobytes()
        bitstream_all += np.array([len(bitstream) for bitstream in bitstream_list], dtype=dtype).tobytes()
        for bitstream in bitstream_list:
            assert len(bitstream)<2**32-1
            bitstream_all += bitstream

        return bitstream_all

    @torch.no_grad()
    def decode(self, x_low, bitstream):
        bitstream_list = self.unpack_bitstream(bitstream)
        x_low = self.block_in(x_low)
        slice_list = self.split(x_low)
        out = None
        for block, curr_slice, bitstream in zip(self.block_list, slice_list, bitstream_list):
            out = self.basic_decode_module(
                block=block, classifier=self.classifier, 
                x_in=curr_slice, siblings=out, bitstream=bitstream)
        out = ME.SparseTensor(torch.ones([out.F.shape[0], 1]).float(), 
            coordinate_map_key=out.coordinate_map_key, 
            coordinate_manager=out.coordinate_manager, 
            device=out.device)

        return out

    def unpack_bitstream(self, bitstream_all, dtype='uint32'):
        s = 0
        num = np.frombuffer(bitstream_all[s:s+1*4], dtype=dtype)[0]
        s += 1*4
        lengths = np.frombuffer(bitstream_all[s:s+num*4], dtype=dtype)
        s += num*4
        # print('DBG!!!', num, lengths)
        bitstream_list = []
        for l in lengths:
            bitstream = bitstream_all[s:s+l]
            bitstream_list.append(bitstream)
            s += l
        
        return bitstream_list

    @torch.no_grad()
    def basic_decode_module(self, block, classifier, x_in, siblings, bitstream):
        # concat  x_in and siblings
        if siblings is None:
            inputs = x_in
        else: 
            inputs = self.concat(siblings, x_in)
        # & extract feature
        out = block(inputs)
        # estimate probability
        if siblings is None:
            x_out = out
        else:
            mask_in = isin(out.C, x_in.C)
            x_out = self.pruning(out, mask_in.to(out.device))
        out_cls = classifier(x_out)
        # out
        out_cls = sort_sparse_tensor(out_cls)
        prob = torch.sigmoid(out_cls.F).detach().cpu()
        # prdob = torch.round(prob*1e3)/1e3
        BAC = BinaryArithmeticCoding()
        occupancy = BAC.decode(prob, bitstream)
        ground_truth = self.pruning(out_cls, occupancy.bool().to(out_cls.device))# prune NOVs & keep POVs
        #
        mask_out = isin(out.C, ground_truth.C)
        if siblings is not None:
            mask_out += isin(out.C, siblings.C)
        out = self.pruning(out, mask_out.to(out.device))

        return out

    @torch.no_grad()
    def test(self, x_low, x_high):
        # real bitstream
        bitstream = self.encode(x_low, x_high)
        bits = len(bitstream)*8
        x_high_rec = self.decode(x_low, bitstream)
        assert (sort_sparse_tensor(x_high).C==sort_sparse_tensor(x_high_rec).C).all()
        # entropy
        # out_set = self.forward(x_low, x_high)
        # curr_bits = []
        # for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
        #     bce = get_bce(out_cls, ground_truth).item()
        #     curr_bits.append(bce)
        # estimated_bits = round(sum(curr_bits))
        # print('test bits:\t', bits, estimated_bits, round(bits/estimated_bits, 3))

        return bits


class BinaryArithmeticCoding():
    """arithmetic coding for occupancy status.
    """
    def _get_cdf(self, prob):
        zeros = torch.zeros(prob.shape, dtype=prob.dtype, device=prob.device)
        ones = torch.ones(prob.shape, dtype=prob.dtype, device=prob.device)
        cdf = torch.cat([zeros, 1-prob, ones], dim=-1)
        
        return cdf

    def estimate_bitrate(self, prob, occupancy):
        pmf = torch.cat([1-prob, prob], axis=-1)
        prob_true = pmf[torch.arange(0,len(occupancy)).tolist(), occupancy.tolist()]
        entropy = -torch.log2(prob_true)
        bits = torch.sum(entropy).tolist()
        
        return bits

    def encode(self, prob, occupancy):
        cdf = self._get_cdf(prob)
        bitstream = torchac.encode_float_cdf(cdf, occupancy)
        # with open(filename+'.bin', 'wb') as fout:
        #     fout.write(bitstream)

        return bitstream

    def decode(self, prob, bitstream):
        cdf = self._get_cdf(prob)
        # with open(filename+'.bin', 'rb') as fin:
        #     bitstream = fin.read()
        occupancy = torchac.decode_float_cdf(cdf, bitstream)
        
        return occupancy


if __name__ == '__main__':
    model = UpSampler8Stage(channels=32, block_layers=3)
    print(model)