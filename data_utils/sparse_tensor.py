import torch
import numpy as np
import MinkowskiEngine as ME


def array2vector(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long(), step.long()
    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])

    return vector

def isin(data, ground_truth):
    """ Input data and ground_truth are torch tensor of shape [N, D].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is in `ground_truth` and False otherwise.
    """
    device = data.device
    if len(ground_truth)==0:
        return torch.zeros([len(data)]).bool().to(device)
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = torch.isin(data.to(device), ground_truth.to(device))

    return mask

def istopk(data, nums, rho=1.0):
    """ Input data is sparse tensor and nums is a list of shape [batch_size].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is the top k (=nums*rho) value and False otherwise.
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N*rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)# must CPU.
        mask[row_indices[indices]]=True

    return mask.bool().to(data.device)

def create_new_sparse_tensor(coordinates, features, tensor_stride, dimension, device):
    sparse_tensor = ME.SparseTensor(features=features, 
                                coordinates=coordinates,
                                tensor_stride=tensor_stride,
                                device=device)
    # manager = ME.CoordinateManager(D=dimension)
    # key, _ = manager.insert_and_map(coordinates.to(device), tensor_stride)
    # sparse_tensor = ME.SparseTensor(features=features, 
    #                                 coordinate_map_key=key, 
    #                                 coordinate_manager=manager, 
    #                                 device=device)

    return sparse_tensor

def sort_sparse_tensor(sparse_tensor):
    """ Sort points in sparse tensor according to their coordinates.
    """
    indices = torch.argsort(array2vector(sparse_tensor.C, 
                                           sparse_tensor.C.max()+1))
    sparse_tensor = create_new_sparse_tensor(coordinates=sparse_tensor.C[indices], 
                                            features=sparse_tensor.F.cpu()[indices], 
                                            tensor_stride=sparse_tensor.tensor_stride, 
                                            dimension=sparse_tensor.D, 
                                            device=sparse_tensor.device)

    return sparse_tensor



if __name__ == '__main__':
    array = torch.randint(0, 10, [12]).reshape(-1,3).to('cuda')
    step = array.max() + 1
    vector = array2vector(array, step)
    print("\narray:\n", array, 
        "\nstep:\n", step, 
        "\nvector:\n", vector)

    data = torch.randint(0, 2, [16]).reshape(-1,4).to('cuda')
    ground_truth = torch.randint(0, 2, [16]).reshape(-1,4).to('cuda')
    mask = isin(data, ground_truth)
    print("\ndata\n:", data, 
        "\nground trurh\n:", data, 
        "\nmask\n", mask)

    n_points = 10
    batch_indexes = torch.zeros(n_points).int()
    batch_indexes[len(batch_indexes)//2:] = 1
    coords = torch.randint(0, 8, [n_points*3]).reshape(-1,3).int()
    batch_coords = torch.cat([batch_indexes.reshape(-1,1), coords], dim=-1)
    
    data = ME.SparseTensor(
        features=torch.rand(n_points).reshape(-1, 1).float(),
        coordinates=batch_coords, 
        device='cuda')
    nums = np.random.randint(1,n_points//2,2).tolist()
    mask = istopk(data, nums)
    print("\ndata\n:", data, 
    "\nnums\n:", nums, 
    "\nmask\n", mask)