import os
import torch
import torch.nn.functional as F

try:
    from models.components.airtnn import AirTNN
except:
    from components.airtnn import AirTNN

class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128, k=1, snr_db=100, p=.0):
        super().__init__()
        self.p = p
        self.l1 = AirTNN(1, hidden_dim, k, snr_db)
        self.l2 = AirTNN(hidden_dim, hidden_dim, k, snr_db)
        self.enc = torch.nn.LazyLinear(128)
        self.out = torch.nn.LazyLinear(10)

    def forward(self, x, lower, upper):
        # # AirTNN
        x = F.relu(self.l1(x, lower, upper))
        x = F.relu(self.l2(x, lower, upper))
        # MLP
        x = self.out(F.relu(self.enc(x))) 
        # Average pooling -> [batch_size, 10] -> max vote
        x = x.mean(dim=1)
        return x