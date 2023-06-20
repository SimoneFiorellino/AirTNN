import os
import torch
import torch.nn.functional as F
import torch.nn as nn

try:
    from models.components.airtnn import AirTNN
except:
    from components.airtnn import AirTNN

class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128, hidden_dim_ffnn=1024, n_layers=2, n_classes=11, k=1, snr_db=100, p=.0, delta=1):
        super().__init__()
        self.p = p
        if n_layers < 2:
            raise ValueError("Number of layers must be greater than 1.")

        # AirTNN
        layers = []
        layers += [AirTNN(1, hidden_dim, k, snr_db, delta)]
        for _ in range(n_layers - 2):
            layers += [
                AirTNN(hidden_dim, hidden_dim, k, snr_db, delta)
            ]
        layers += [AirTNN(hidden_dim, 64, k, snr_db, delta)]
        self.layers = nn.ModuleList(layers)
        # MLP
        self.enc = torch.nn.LazyLinear(hidden_dim_ffnn)
        self.out = torch.nn.LazyLinear(n_classes)

    def forward(self, x, lower, upper):
        # # AirTNN
        for i in range(len(self.layers)):
            x = self.layers[i](x, lower, upper)
            x = F.relu(x)
            x = F.dropout(x, p=self.p)
        # Max pooling: [batch_size, nodes, F] -> [batch_size, F]
        x = x.max(dim=1)[0]
        # MLP
        x = self.out(F.relu(self.enc(x))) 
        return x