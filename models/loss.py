import os, sys
rootdir = os.path.split(__file__)[0]
sys.path.append(rootdir)
rootdir = os.path.split(rootdir)[0]
sys.path.append(rootdir)
import torch
import numpy as np
from collections import Counter
from data_utils.sparse_tensor import isin, istopk
criterion = torch.nn.BCEWithLogitsLoss()


def get_bce(data, groud_truth):
    """ Input data and ground_truth are sparse tensor.
    """
    mask = isin(data.C, groud_truth.C)
    bce = criterion(data.F.squeeze(), mask.type(data.F.dtype))
    bce /= torch.log(torch.tensor(2.0)).to(bce.device)
    sum_bce = bce * data.shape[0]
    
    return sum_bce

def get_bits(likelihood):
    bits = -torch.sum(torch.log2(likelihood))

    return bits

def get_metrics(data, groud_truth):
    mask_real = isin(data.C, groud_truth.C)
    nums = [len(C) for C in groud_truth.decomposed_coordinates]
    mask_pred = istopk(data, nums, rho=1.0)
    metrics = get_cls_metrics(mask_pred, mask_real)

    return metrics[0]

def get_cls_metrics(pred, real):
    TP = (pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FN = (~pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FP = (pred * ~real).cpu().nonzero(as_tuple=False).shape[0]
    TN = (~pred * ~real).cpu().nonzero(as_tuple=False).shape[0]

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    IoU = TP / (TP + FP + FN + 1e-7)

    return [round(precision, 4), round(recall, 4), round(IoU, 4)]

# metric: shannon entropy
def get_entropy2(error):
    bits = 0
    for idx_ch in range(error.shape[-1]):
        bits += get_entropy(error[:,idx_ch])
    return bits

def get_entropy(error):
    # normalization
    # print("=========== entropy ===========")
    data = error.reshape(-1)
    data = data.astype('int')
    keys = np.sort(np.unique(data))
    dataN = data.copy()
    for i, k in enumerate(keys):
        dataN[data==k] = i
    # data = dataN.copy()

    statistic = Counter(dataN)
    freq_table = {}
    for _, k in enumerate(sorted(statistic)):
        freq_table[k]=statistic[k]/sum(statistic.values())
    pmf = np.array([p for p in freq_table.values()])
    pmf = pmf.astype('float32').round(8)

    probs = pmf[dataN]
    bits = -np.log2(np.array(probs))
    bits = bits.reshape(error.shape)

    return bits.sum()

from pytorch3d.loss import chamfer_distance
def get_chamfer_distance(coords0, coords1):
    """input: coords0, coords1, [N,3];  batch size should be one.
    """
    # curWarp = motion.C[:,1:].float() + motion.F[:,0:3]
    # ref = feat0.C[:,1:].float()
    chamferLoss, _ = chamfer_distance(coords0.float().unsqueeze(0), coords1.float().unsqueeze(0))

    return chamferLoss

from pytorch3d.ops.knn import knn_points
def get_smoothness_loss(data, knn=8):
    """input: sparse tensor;  batch size should be one.
    """
    _, knnIdxs, _ = knn_points(data.C[:,1:].float().unsqueeze(0), data.C[:,1:].float().unsqueeze(0), K=knn)
    smoothLoss = torch.norm(data.F.cpu()[knnIdxs.squeeze(0)] - data.F.cpu().unsqueeze(1), dim=-1).mean()

    return smoothLoss