import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from models.components.airnn import AirNN

class AirTNN(AirNN):

    def __init__(self, c_in, c_out, k = 1, snr_db = 10, delta = 1):
        super(AirNN, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.snr_db = snr_db
        self.snr_lin = 10 ** (self.snr_db / 10)  # SNRdb to linear        
        self.k = k
        self.delta = torch.tensor(delta, dtype=torch.float32)
            
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
        self.h_lin.reset_parameters()

    def shift(self, x_up, x_low, upper_lp, lower_lp):
        """function to shift the input signal:
        0. sample S and noise
        1. multiply pairwise A with S
        2. apply the shift operator to x
        3. add white noise"""
        if self.snr_db == 100:
            x_up = self.batch_mm(upper_lp, x_up)
            x_low = self.batch_mm(lower_lp, x_low)
        else:
            x_up  = self.batch_mm(upper_lp * self.channel_fading(upper_lp, self.delta), x_up ) + self.white_noise(x_up, self.snr_lin)[None,:,:]
            x_low = self.batch_mm(lower_lp * self.channel_fading(lower_lp, self.delta), x_low) + self.white_noise(x_low, self.snr_lin)[None,:,:]
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