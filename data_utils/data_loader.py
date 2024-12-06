import os, sys
rootdir = os.path.split(__file__)[0]
sys.path.append(rootdir)
rootdir = os.path.split(rootdir)[0]
sys.path.append(rootdir)
import glob, os, time
from tqdm import tqdm
import numpy as np
import torch
import MinkowskiEngine as ME
from inout import read_coords
from quantize import quantize_precision, quantize_resolution, quantize_octree, random_quantize, merge_points
from partition import kdtree_partition
import random


def load_sparse_tensor(filedir, voxel_size=1, resolution=None, qlevel=None, quant_mode='round', device='cuda'):
    coords = read_coords(filedir)
    assert voxel_size is None or resolution is None
    if voxel_size is not None:
        coords = quantize_precision(coords, precision=voxel_size, quant_mode=quant_mode, return_offset=False)
    elif resolution is not None:
        coords, _, _ = quantize_resolution(coords, resolution=resolution, quant_mode=quant_mode, return_offset=False)
    elif qlevel is not None:
        coords, _, _ = quantize_octree(coords, qlevel=qlevel, quant_mode=quant_mode, return_offset=False)
    # coords -= coords.min(axis=0)
    coords = np.unique(coords.astype('int'), axis=0).astype('int')
    coords = torch.tensor(coords).int()
    feats = torch.ones((len(coords),1)).float()
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, 
                        tensor_stride=1, device=device)

    return x

class InfSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        else:
            perm = torch.range(0, perm-1).long()
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)

######################## static point cloud ############################
def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1
    list_data = new_list_data
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    coords, feats = list(zip(*list_data))
    coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)

    return coords_batch, feats_batch


class PCDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None, voxel_size=1, resolution=None, qlevel=None, max_num=1e7, augment=False):
        self.files = []
        self.cache = {}
        self.files = files
        self.transforms = transforms
        assert voxel_size is None or resolution is None
        self.voxel_size = voxel_size
        self.resolution = resolution
        self.qlevel = qlevel
        self.max_num = max_num
        self.augment = augment

    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):
        filedir = self.files[idx]
        if filedir.endswith('bin'):
            self.voxel_size = None
            self.resolution = None
            self.qlevel = 12
        if idx in self.cache:
            coords, feats = self.cache[idx]
        else:
            # import time
            # start = time.time()
            coords = read_coords(filedir)
            # coords = quantize_precision(coords, precision=self.voxel_size, quant_mode='round', return_offset=False)
            if self.voxel_size is not None:
                coords = quantize_precision(coords, precision=self.voxel_size, quant_mode='round', return_offset=False)
            elif self.resolution is not None:
                coords, _, _ = quantize_resolution(coords, resolution=self.resolution, quant_mode='round', return_offset=False)
            elif self.qlevel is not None:
                coords, _, _, _ = quantize_octree(coords, qlevel=self.qlevel, quant_mode='round', return_offset=False)
            coords = np.unique(coords.astype('int'), axis=0).astype('int')
            # print('DBG!!! loading time', round(time.time() - start, 4), filedir, len(coords), coords.max() - coords.min())
            # print('DBG!!! loading', len(coords), coords.max() - coords.min())
            if self.augment: 
                coords = random_quantize(coords)
                # print('DBG!!! augment', coords.max() - coords.min())
            if len(coords) > self.max_num:
                print('DBG', len(coords), self.max_num)
                parts = kdtree_partition(coords, max_num=self.max_num)
                coords = random.sample(parts, 1)[0]
                print('DBG!!! partition', len(parts), len(coords))
            # # transform
            # if self.transforms is not None:
            #     for trans in self.transforms:
            #         coords = trans(coords)
            feats = np.ones([len(coords), 1]).astype('bool')
            self.cache[idx] = (coords, feats)
        feats = feats.astype("float32")

        return (coords, feats)


class PCDatasetOffset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None, voxel_size=None, resolution=1023, qlevel=None, max_num=1e7, augment=False):
        self.files = []
        self.cache = {}
        self.files = files
        self.transforms = transforms
        assert voxel_size is None or resolution is None
        self.voxel_size = voxel_size
        self.resolution = resolution
        self.qlevel = qlevel
        self.max_num = max_num
        self.augment = augment

    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):
        filedir = self.files[idx]
        if filedir.endswith('bin'):
            self.voxel_size = None
            # self.resolution = 2**12-1
            self.qlevel = None
        if idx in self.cache:
            coords, feats = self.cache[idx]
        else:
            coords = read_coords(filedir)
            if self.voxel_size is not None:
                coords, offset = quantize_precision(coords, precision=self.voxel_size, quant_mode='round', return_offset=True)
            elif self.resolution is not None:
                coords, _, _, offset = quantize_resolution(coords, resolution=self.resolution, quant_mode='round', return_offset=True)
            elif self.qlevel is not None:
                coords, _, _, _, offset = quantize_octree(coords, qlevel=self.qlevel, quant_mode='round', return_offset=True)
            coords, offset = merge_points(coords, offset)
            # coords = np.unique(coords.astype('int'), axis=0).astype('int')
            # print('DBG!!! loading time', round(time.time() - start, 4), filedir, len(coords), coords.max() - coords.min())
            # print('DBG!!! loading', len(coords), coords.max() - coords.min())
            # if self.augment: 
            #     coords = random_quantize(coords)
            #     # print('DBG!!! augment', coords.max() - coords.min())
            # if len(coords) > self.max_num:
            #     print('DBG', len(coords), self.max_num)
            #     parts = kdtree_partition(coords, max_num=self.max_num)
            #     coords = random.sample(parts, 1)[0]
            #     print('DBG!!! partition', len(parts), len(coords))
            # # transform
            # if self.transforms is not None:
            #     for trans in self.transforms:
            #         coords = trans(coords)
            feats = offset.astype("float32")
            self.cache[idx] = (coords, feats)

        return (coords, feats)


######################## data loader ############################
def make_data_loader(dataset, batch_size=1, shuffle=True, num_workers=1, repeat=False, 
                    collate_fn=collate_pointcloud_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", default='../testdata/kitti_seqs11_000000.bin')
    parser.add_argument("--rootdir", default='../../dataset/sparsepcgc_testdata/Ford/')
    parser.add_argument("--voxel_size", type=float, default=1)
    parser.add_argument("--resolution", type=float, default=4096)
    parser.add_argument("--qlevel", type=float, default=12)
    parser.add_argument("--max_num", type=int, default=10000000)
    parser.add_argument("--augment", action="store_true", help="test or not.")# random_quantize
    args = parser.parse_args()
    filedirs = sorted(glob.glob(os.path.join(args.rootdir,'**', f'*.h5'), recursive=True) + \
                    glob.glob(os.path.join(args.rootdir,'**', f'*.ply'), recursive=True) + \
                    glob.glob(os.path.join(args.rootdir,'**', f'*.bin'), recursive=True))

    #################### static point cloud ###################
    # x = load_sparse_tensor(args.filedir, voxel_size=args.voxel_size)
    x = load_sparse_tensor(args.filedir, voxel_size=None, resolution=args.resolution)
    # print(x)
    print('x:\t', len(x), x.C.max().item() - x.C.min().item())

    # test_dataset = PCDataset(filedirs, transforms=None, voxel_size=args.voxel_size, resolution=None,
    #                         max_num=args.max_num, augment=args.augment)
    # test_dataset = PCDataset(filedirs, transforms=None, voxel_size=None, resolution=args.resolution, 
    #                         qlevel=args.qlevel, max_num=args.max_num, augment=args.augment)

    test_dataset = PCDatasetOffset(filedirs, transforms=None, voxel_size=None, resolution=args.resolution, 
                            qlevel=None, max_num=args.max_num, augment=args.augment)

    test_dataloader = make_data_loader(
        dataset=test_dataset, batch_size=1, shuffle=True, num_workers=1, repeat=False,
        collate_fn=collate_pointcloud_fn)
    for idx, (coords, feats) in enumerate(tqdm(test_dataloader)):
        print("idx:", len(coords), coords.max().item() - coords.min().item())
        print(feats.shape, feats)