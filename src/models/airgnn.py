import torch
import torch.nn as nn


class AirGNN(nn.Module):

    def _white_noise(self, z):
        """function to add white noise to the input signal"""
        with torch.no_grad():
            z_clone = torch.detach(z)
            snr_lin = 10 ** (self.snr_db / 10)  # SNRdb to linear
            x_power = torch.sum(z_clone ** 2) / torch.numel(z_clone)
            var = x_power / snr_lin
            std = torch.sqrt(var)
            return torch.randn_like(z_clone) * std
        
    def _channel_fading(self, adj):
        """function to generate channel fading"""
        with torch.no_grad():
            s_air = torch.randn_like(adj, dtype=torch.complex64)
            s_air = torch.abs(s_air)
            return s_air

    def __init__(self, c_in, c_out, snr_db = 10):
        super(AirGNN, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.snr_db = snr_db
        self.weight = nn.Parameter(torch.Tensor(c_in, c_out))

    def forward(self, x_in, adj):
        """
        1. multiply pairwise A with S
        2. apply the shift operator to x
        3. add white noise
        4. multiply with the weight matrix
        """
        S = adj * self._channel_fading(adj) 
        x = torch.einsum("ij,kl->il", (S,x_in))
        x = x + self._white_noise(x)
        x = torch.einsum("ij,kl->il", (x, self.weight))
        return x



if __name__ == '__main__':
    # test
    x = torch.randn(10, 3) + 10
    adj = (torch.randn(10, 10) + .3).clamp(min = 0, max = 1)
    model = AirGNN(3, 4)
    y = model(x, adj)
