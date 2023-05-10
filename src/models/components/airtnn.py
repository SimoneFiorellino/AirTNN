import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class AirTNN(nn.Module):

    def _white_noise(self, z):
        """function to add white noise to the input signal"""
        with torch.no_grad():
            z_clone = torch.detach(z[0,:,:])
            x_power = torch.sum(z_clone ** 2) / torch.numel(z_clone)
            var = x_power / self.snr_lin
            std = torch.sqrt(var)
            return torch.randn_like(z_clone) * std

    def _channel_fading(self, s):
        """function to generate channel fading"""
        with torch.no_grad():
            s_clone = torch.detach(s[0,:,:])
            s_air = torch.randn_like(s_clone, dtype=torch.complex64) * self.cf_std
            s_air = torch.abs(s_air)
            return s_air

    def __init__(self, c_in, c_out, k = 1, snr_db = 10):
        super(AirTNN, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.snr_db = snr_db
        self.snr_lin = 10 ** (self.snr_db / 10)  # SNRdb to linear        
        self.k = k
        self.cf_std = torch.sqrt(torch.tensor(0.5, dtype=torch.float32))
        self.up_lins = torch.nn.ModuleList([
            Linear(c_in, c_out, bias=False) for _ in range(k + 1)
        ])
        self.low_lins = torch.nn.ModuleList([
            Linear(c_in, c_out, bias=False) for _ in range(k + 1)
        ])
        self.h_lin = Linear(c_in, c_out, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.up_lins:
            lin.reset_parameters()
        for lin in self.low_lins:
            lin.reset_parameters()

    def shift(self, x_up, x_low, upper_lp, lower_lp):
        """function to shift the input signal:
        1. multiply pairwise A with S
        2. apply the shift operator to x
        3. add white noise"""
        if self.snr_db == 1000:
            x_up  = torch.bmm(upper_lp, x_up )
            x_low = torch.bmm(lower_lp, x_low)
        else:
            fading = self._channel_fading(upper_lp)[None,:,:]
            noise  = self._white_noise(x_up)[None,:,:]
            x_up  = torch.bmm(upper_lp * fading, x_up ) + noise
            x_low = torch.bmm(lower_lp * fading, x_low) + noise
        return x_up, x_low

    def forward(self, x, lower_lp, upper_lp):
        """function to forward the input signal:
        1. shift the input signal
        2. apply the weight matrix to x
        3. sum the output of each shift"""

        x_up, x_low = self.shift(x, x, upper_lp, lower_lp)
        out_up = self.up_lins[0].forward(x_up)
        out_low = self.low_lins[0].forward(x_low)

        for up_lin, low_lin in zip(self.up_lins[1:], self.low_lins[1:]):
            x_up, x_low = self.shift(x_up, x_low, upper_lp, lower_lp)
            out_up = out_up + up_lin.forward(x_up)
            out_low = out_low + low_lin.forward(x_low)

        return out_up + out_low + self.h_lin.forward(x)