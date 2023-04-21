import torch
import torch.nn as nn


class TAirGNN(nn.Module):

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
        super(TAirGNN, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.snr_db = snr_db
        self.k = k
        self.up_weight = nn.Parameter(torch.Tensor(c_in, c_out, k))
        self.low_weight = nn.Parameter(torch.Tensor(c_in, c_out, k))

    def reset_parameters(self):
        """function to initialize the weight matrix"""
        nn.init.xavier_uniform_(self.up_weight)
        nn.init.xavier_uniform_(self.low_weight)

    def shift(self, x_up, x_low, upper_lp, lower_lp):
        """function to shift the input signal:
        1. multiply pairwise A with S
        2. apply the shift operator to x
        3. add white noise"""
        fading = self._channel_fading(upper_lp)
        noise  = self._white_noise(x_up)
        x_up  = torch.einsum("ij,jk->ik", (upper_lp * fading, x_up )) + noise # (S_up @ x_up + n)
        x_low = torch.einsum("ij,jk->ik", (lower_lp * fading, x_low)) + noise # (S_low @ x_low + n)
        return x_up, x_low

    def forward(self, x, upper_lp, lower_lp):
        """function to forward the input signal:
        1. shift the input signal
        2. apply the weight matrix to x
        3. sum the output of each shift"""

        x_up, x_low = self.shift(x, x, upper_lp, lower_lp)
        out_up  = torch.einsum("ij,jk->ik", (x_up, self.up_weight[:,:,0]))
        out_low = torch.einsum("ij,jk->ik", (x_low, self.low_weight[:,:,0]))

        for i in range(1, self.k):
            x_up, x_low = self.shift(x_up, x_low, upper_lp, lower_lp)
            out_up += torch.einsum("ij,jk->ik", (x_up, self.up_weight[:,:,i]))
            out_low += torch.einsum("ij,jk->ik", (x_low, self.low_weight[:,:,i]))

        return out_up + out_low

if __name__ == '__main__':
    # test
    x = torch.randn(10, 3)
    upper_lp = torch.randn(10, 10)
    lower_lp = torch.randn(10, 10)
    model = TAirGNN(3, 4, k = 10)
    out = model(x, upper_lp, lower_lp)
