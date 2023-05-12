import torch

# def white_noise(z, snr_db):
#     z_clone = torch.detach(z)
#     x_power = torch.Tensor([1.]) / torch.numel(z_clone)
#     snr_lin = 10 ** (snr_db / 10)
#     var = x_power / snr_lin
#     std = torch.sqrt(var)
#     return torch.randn_like(z_clone) * std

def white_noise(z, snr_db):
    z_clone = torch.detach(z)
    x_power = torch.sum(z_clone ** 2) / torch.numel(z_clone)
    snr_lin = 10 ** (snr_db / 10)
    var = x_power / snr_lin
    std = torch.sqrt(var)
    return torch.randn_like(z_clone) * std

def channel_fading(adj):
    """function to generate channel fading"""
    s_air = torch.randn_like(adj, dtype=torch.complex64) * torch.sqrt(torch.tensor(0.5, dtype=torch.float32))
    s_air = torch.abs(s_air)
    return s_air

# def k_hop_adjacency_matrix(adj_matrix, k):
#     n = adj_matrix.shape[0]
#     adj_tensor = torch.zeros((k, n, n))
#     adj_tensor[0] = adj_matrix
#     for i in range(1, k):
#         adj_tensor[i] = torch.matmul(adj_matrix, adj_tensor[i-1])
#     return adj_tensor

def k_hop_adjacency_matrix(x, k):
    """
    Function to compute the k matrix: A^k

    :param x: input matrix
    :param k: number of hops

    :return A
    """
    if k == 0:
        return torch.eye(x.shape[0])
    shape_a = x.shape[0]
    A = torch.empty((k, shape_a, shape_a))

    for i in range(k):
        A[i,:,:] = torch.matrix_power(x, i+1)

    return A