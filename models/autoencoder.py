import os, sys
rootdir = os.path.split(__file__)[0]
sys.path.append(rootdir)
rootdir = os.path.split(rootdir)[0]
sys.path.append(rootdir)
import torch
import MinkowskiEngine as ME
import numpy as np
from resnet import ResNetBlock
from entropy_model import EntropyBottleneck
from decoder import UpSampler
from data_utils.sparse_tensor import array2vector, sort_sparse_tensor


class DownSampler(torch.nn.Module):
    def __init__(self, in_channels=1, channels=32, out_channels=32, block_layers0=0, block_layers1=0, kernel_size=3):
        super().__init__()
        self.block0 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels, out_channels=channels,
                kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=True),
            ResNetBlock(
                block_layers=block_layers0, channels=channels, kernel_size=kernel_size))
        self.conv = ME.MinkowskiConvolution(
            in_channels=channels, out_channels=channels,
            kernel_size=2, stride=2, bias=True, dimension=3)
        self.block1 = torch.nn.Sequential(
            ResNetBlock(
                block_layers=block_layers1, channels=channels, kernel_size=kernel_size),
            ME.MinkowskiConvolution(
                in_channels=channels, out_channels=out_channels,
                kernel_size=3, stride=1, bias=True, dimension=3))
        self.relu = ME.MinkowskiReLU(inplace=True)
    
    def forward(self, x):
        out = self.block0(x)
        out = self.relu(self.conv(out))
        out = self.block1(out)

        return out


class AutoEncoder(torch.nn.Module):
    def __init__(self, in_channels=1, channels=32, out_channels=32, block_layers=3, kernel_size=3):
        super().__init__()
        self.downsampler0 = DownSampler(
            in_channels=in_channels, channels=channels, out_channels=channels*2, kernel_size=kernel_size, 
            block_layers0=block_layers, block_layers1=block_layers)
        self.downsampler1 = DownSampler(
            in_channels=channels*2, channels=channels*2, out_channels=out_channels, kernel_size=kernel_size, 
            block_layers0=0, block_layers1=block_layers)
        self.upsampler0 = UpSampler(
            in_channels=out_channels, channels=channels*2, out_channels=channels, kernel_size=kernel_size,
            block_layers0=block_layers, block_layers1=0, 
            expand_coordinates=False)
        self.entropy_bottleneck = EntropyBottleneck(channels=out_channels)

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F, quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x, training=True):
        x_low = self.downsampler0(x)
        latent = self.downsampler1(x_low)
        latentQ, likelihood = self.get_likelihood(latent, 
            quantize_mode="noise" if training else "symbols")
        x_low_dec = self.upsampler0(latentQ)

        return {'likelihood': likelihood,
                'latent':latentQ, 
                'x_low':x_low_dec}

    @torch.no_grad()
    def encode(self, x, return_one=True):
        out_set = self.forward(x, training=False)
        latent = sort_sparse_tensor(out_set['latent'])
        x_low = out_set['x_low']
        feats = latent.F
        strings, min_v, max_v = self.entropy_bottleneck.compress(feats)
        shape = feats.shape
        bitstream = self.pack_bitstream(shape, min_v, max_v, strings)
        if return_one: 
            x_low = ME.SparseTensor(
                torch.ones([x_low.F.shape[0], 1]).float(), 
                coordinate_map_key=x_low.coordinate_map_key, 
                coordinate_manager=x_low.coordinate_manager, 
                device=x_low.device)
        
        return x_low, bitstream

    def pack_bitstream(self, shape, min_v, max_v, strings, dtype='int32'):
        bitstream = np.array(shape, dtype=dtype).tobytes()
        bitstream += np.array(min_v, dtype=dtype).tobytes()
        bitstream += np.array(max_v, dtype=dtype).tobytes()
        bitstream += strings

        return bitstream

    @torch.no_grad()
    def decode(self, x_low, bitstream):
        shape, min_v, max_v, strings = self.unpack_bitstream(bitstream)
        feats = self.entropy_bottleneck.decompress(strings, min_v, max_v, shape, channels=shape[-1])
        downsampler = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3).to(x_low.device)
        latent = downsampler(x_low)
        index = array2vector(latent.C, latent.C.max()+1).sort()[1]
        inverse_index = index.sort()[1]
        latent = ME.SparseTensor(
            features=feats[inverse_index], 
            coordinate_map_key=latent.coordinate_map_key, 
            coordinate_manager=latent.coordinate_manager, 
            device=latent.device)
        x_low = self.upsampler0(latent)

        return x_low

    def unpack_bitstream(self, bitstream, dtype='int32'):
        s = 0
        shape = np.frombuffer(bitstream[s:s+2*4], dtype=dtype)
        s += 2*4
        min_v = np.frombuffer(bitstream[s:s+1*4], dtype=dtype)
        s += 1*4
        max_v = np.frombuffer(bitstream[s:s+1*4], dtype=dtype)
        s += 1*4
        strings = bitstream[s:]

        return shape, min_v, max_v, strings

    @torch.no_grad()
    def test(self, x):
        # real bitstream
        x_low, bitstream = self.encode(x, return_one=False)
        x_low_one = ME.SparseTensor(torch.ones([x_low.F.shape[0], 1]).float(), 
            coordinate_map_key=x_low.coordinate_map_key, 
            coordinate_manager=x_low.coordinate_manager, 
            device=x_low.device)
        bits = len(bitstream)*8
        x_low_dec = self.decode(x_low_one, bitstream)
        assert (sort_sparse_tensor(x_low).F==sort_sparse_tensor(x_low_dec).F).all()
        # enc_set = self.forward(x, training=False)
        # estimated_bits = round(get_bits(enc_set['likelihood']).item())
        # print('DBG! feat_bpp\t', bits, estimated_bits, round(bits/estimated_bits,3))

        return x_low_dec, bits

if __name__ == '__main__':
    model = AutoEncoder(channels=32)
    print(model)