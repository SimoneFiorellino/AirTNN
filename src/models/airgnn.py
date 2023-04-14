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

    def __init__(self, c_in, c_out, k = 1, snr_db = 10):
        super(AirGNN, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.snr_db = snr_db
        self.k = k
        self.weight = nn.Parameter(torch.Tensor(c_in, c_out, k))

    def shift(self, x, adj):
        """function to shift the input signal:
        1. multiply pairwise A with S
        2. apply the shift operator to x
        3. add white noise"""
        S = adj * self._channel_fading(adj) 
        x = torch.einsum("ij,kl->il", (S,x))
        x = x + self._white_noise(x)
        return x

    def forward(self, x, adj):
        """function to forward the input signal:
        1. shift the input signal
        2. apply the weight matrix to x
        3. sum the output of each shift"""
        out = torch.zeros(x.shape[0], self.c_out)
        for i in range(self.k):
            x = self.shift(x, adj)
            out += torch.einsum("ij,kl->il", (x, self.weight[:,:,i]))
        return out

if __name__ == '__main__':
    # test
    x = torch.randn(10, 3)
    adj = torch.randn(10, 10)
    model = AirGNN(3, 3, k = 1)
    out = model(x, adj)
