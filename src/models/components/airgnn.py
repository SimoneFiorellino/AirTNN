import torch
import torch.nn as nn


class AirGNN(nn.Module):

    def _white_noise(self, z):
        """function to add white noise to the input signal"""
        with torch.no_grad():
            z_clone = torch.detach(z)
            x_power = torch.sum(z_clone ** 2) / torch.numel(z_clone)
            var = x_power / self.snr_lin
            std = torch.sqrt(var)
            return torch.randn_like(z_clone) * std

    def _channel_fading(self, adj):
        """function to generate channel fading"""
        with torch.no_grad():
            s_air = torch.randn_like(adj, dtype=torch.complex64) * torch.sqrt(torch.tensor(0.5, dtype=torch.float32))
            s_air = torch.abs(s_air)
            return s_air

    def __init__(self, c_in, c_out, k = 1, snr_db = 10):
        super(AirGNN, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.snr_db = snr_db
        self.snr_lin = 10 ** (self.snr_db / 10)  # SNRdb to linear
        self.k = k
        self.weight = nn.Parameter(torch.Tensor(c_in, c_out, k), requires_grad=True)

    def reset_parameters(self):
        """function to initialize the weight matrix as glorot"""
        nn.init.xavier_uniform_(self.weight)

    def shift(self, x, S):
        """function to shift the input signal:
        1. multiply pairwise A with S
        2. apply the shift operator to x
        3. add white noise"""
        # check if x and S have Nan values with try except
        # if torch.isnan(x).any():
        #     raise ValueError("x has Nan values")
        if torch.isnan(S).any():
            raise ValueError("S has Nan values")

        #S = S * self._channel_fading(adj) 
        x = torch.einsum("bin,bnc->bic", (S,x)) # S @ x.T
        #x = x + self._white_noise(x)
        if torch.isnan(x).any():
            print("x has Nan values")
            #raise ValueError("post shift x has Nan values")
        return x

    def forward(self, x, adj):
        """function to forward the input signal:
        1. shift the input signal
        2. apply the weight matrix to x
        3. sum the output of each shift"""
        x = self.shift(x, adj)
        out = x @ self.weight[:,:,0] # torch.einsum("bnu,buo->bno", (x, self.weight[None,:,:,0])) # x @ self.weight[:,:,0]
        for i in range(1, self.k):
            #x = self.shift(x, adj)
            out += x @ self.weight[:,:,0] # torch.einsum("bnu,uo->bno", (x, self.weight[:,:,i])) # x @ self.weight[:,:,i]
        return out

if __name__ == '__main__':
    # test
    x = torch.randn(10, 3)
    adj = torch.randn(10, 10)
    model = AirGNN(3, 3, k = 1)
    out = model(x, adj)
