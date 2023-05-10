import torch
import torch.nn as nn
from torch.nn import Linear

class AirGNN(nn.Module):

    def _white_noise(self, z):
        """function to add white noise to the input signal"""
        with torch.no_grad():
            z_clone = torch.detach(z[0,:,:])
            x_power = torch.sum(z_clone ** 2) / torch.numel(z_clone)
            var = x_power / self.snr_lin
            std = torch.sqrt(var)
            return torch.randn_like(z_clone) * std

    def _channel_fading(self, adj):
        """function to generate channel fading"""
        with torch.no_grad():
            adj_clone = torch.detach(adj[0,:,:])
            s_air = torch.randn_like(adj_clone, dtype=torch.complex64) * self.fad_std
            s_air = torch.abs(s_air)
            return s_air

    def __init__(self, c_in, c_out, k = 1, snr_db = 10, bias: bool = True):
        super(AirGNN, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.snr_db = snr_db
        self.snr_lin = 10 ** (self.snr_db / 10)  # SNRdb to linear
        self.k = k
        self.fad_std = torch.sqrt(torch.tensor(0.5, dtype=torch.float32))
        self.lins = torch.nn.ModuleList([
            Linear(c_in, c_out, bias=False) for _ in range(k + 1)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def shift(self, x, S):
        """function to shift the input signal:
        1. multiply pairwise A with S
        2. apply the shift operator to x
        3. add white noise"""
        S = S * self._channel_fading(S)[None,:,:]
        x = torch.bmm(S,x)
        x = x + self._white_noise(x)[None,:,:]
        return x

    def forward(self, x, adj):
        """function to forward the input signal:
        1. shift the input signal
        2. apply the weight matrix to x
        3. sum the output of each shift"""
        x = self.shift(x, adj)
        out = self.lins[0].forward(x)
        for lin in self.lins[1:]:
            x = self.shift(x, adj)
            out = out + lin.forward(x)
        return out

if __name__ == '__main__':
    # test
    x = torch.randn(10, 3)
    adj = torch.randn(10, 10)
    model = AirGNN(3, 3, k = 1)
    out = model(x, adj)
