import torch

def white_noise(z, snr_db):
    snr_lin = 10 ** (snr_db / 10)  # SNRdb to linear
    var = 1 / snr_lin
    std = torch.sqrt(torch.Tensor([var]))
    return torch.randn_like(z) * std

def k_hop_adjacency_matrix(adj_matrix, k):
    n = adj_matrix.shape[0]
    adj_tensor = torch.zeros((k, n, n)).double()
    adj_tensor[0] = adj_matrix
    for i in range(1, k):
        adj_tensor[i] = torch.matmul(adj_matrix, adj_tensor[i-1])
    return adj_tensor