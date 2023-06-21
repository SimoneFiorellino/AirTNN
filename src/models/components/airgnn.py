import torch
import torch.nn as nn
from torch.nn import Linear

from models.components.components_utilts import *

class AirGNN(nn.Module):

    def __init__(self, c_in, c_out, k = 1, snr_db = 10, delta = 1):
        super(AirGNN, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.snr_db = snr_db
        self.snr_lin = 10 ** (self.snr_db / 10)  # SNRdb to linear
        self.k = k
        self.delta = torch.tensor(delta, dtype=torch.float32)

        self.lins = torch.nn.ModuleList([
            Linear(c_in, c_out, bias=False) for _ in range(k + 1)
        ])
        self.h_lin = Linear(c_in, c_out, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def shift(self, x, S):
        """function to shift the input signal:
        1. multiply pairwise A with S
        2. apply the shift operator to x
        3. add white noise"""
        if self.snr_db == 100:
            return batch_mm(S,x)
        
        fading = full_channel_fading(S, self.delta)
        x = batch_mm(S * fading, x) + white_noise(x, self.snr_lin)[None,:,:]
        
        return x

    def forward(self, x_in, adj):
        """function to forward the input signal:
        1. shift the input signal
        2. apply the weight matrix to x
        3. sum the output of each shift"""
        x = self.shift(x_in, adj)
        out = self.lins[0].forward(x)
        for lin in self.lins[1:]:
            x = self.shift(x, adj)
            out = out + lin.forward(x)
        return out + self.h_lin.forward(x_in)

if __name__ == '__main__':
    # test
    x = torch.randn(10, 3)
    adj = torch.randn(10, 10)
    model = AirGNN(3, 3, k = 1)
    out = model(x, adj)
