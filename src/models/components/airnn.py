import torch
import torch.nn as nn
from torch.nn import Linear

# Assuming that batch_mm and full_channel_fading are already defined or imported

class AirNN(nn.Module):

    """TODO: create another class NN. AirNN ineherits from NN batch_mm"""

    def white_noise(self, z, snr_lin):
        """function to add white noise to the input signal"""
        with torch.no_grad():
            z_clone = torch.detach(z[0,:,:])
            x_power = torch.sum(z_clone ** 2) / torch.numel(z_clone)
            var = x_power / snr_lin
            std = torch.sqrt(var)
            return torch.randn_like(z_clone) * std
    
    def channel_fading(self, adj, delta):
        """function to generate channel fading"""
        with torch.no_grad():
            s_air = torch.randn(size=adj.shape, dtype=torch.complex64, device=adj.device) * delta
            s_air = torch.abs(s_air)
            return s_air
        
    def batch_mm(self,matrix, matrix_batch):
        """
        :param matrix: Sparse or dense matrix, size (m, n).
        :param matrix_batch: Batched dense matrices, size (b, n, k).
        :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
        """
        batch_size = matrix_batch.shape[0]
        # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
        vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

        # A matrix-matrix product is a batched matrix-vector product of the columns.
        # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
        return torch.sparse.mm(matrix, vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)