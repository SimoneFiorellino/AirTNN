import os
import torch
import torch.nn.functional as F
import torch.nn as nn

try:
    from models.components.airtnn import AirTNN
except:
    from components.airtnn import AirTNN

class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128, hidden_dim_ffnn=1024, n_classes=11, k=1, snr_db=100, p=.0, delta=1):
        super().__init__()
        self.p = p

        # AirTNN
        self.l1 = AirTNN(1, hidden_dim, k, snr_db, delta)
        self.l2 = AirTNN(hidden_dim, hidden_dim, k, snr_db, delta)
        # MLP
        self.enc = torch.nn.LazyLinear(hidden_dim_ffnn)
        self.out = torch.nn.LazyLinear(n_classes)

    def forward(self, x, lower, upper):
        # # AirTNN
        x = F.relu(self.l1(x, lower, upper))
        x = F.relu(self.l2(x, lower, upper))
        x = F.dropout(x, p=self.p, training=self.training)
        # Max pooling: [batch_size, nodes, F] -> [batch_size, F]
        x = x.max(dim=1)[0]
        # MLP
        x = self.out(F.relu(self.enc(x))) 
        return x