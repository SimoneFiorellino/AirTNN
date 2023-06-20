import os
import torch
import torch.nn.functional as F
import torch.nn as nn

try:
    from models.components.tnn import TNN
except:
    from components.tnn import TNN

class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128, hidden_dim_ffnn=1024, n_layers=2, n_classes=11, k=1, snr_db=100, p=.0, delta=1):
        super().__init__()
        self.p = p
        if n_layers < 2:
            raise ValueError("Number of layers must be greater than 1.")

        # TNN
        layers = []
        layers += [TNN(1, hidden_dim, k, snr_db, delta)]
        for _ in range(n_layers - 2):
            layers += [
                TNN(hidden_dim, hidden_dim, k, snr_db, delta),
                nn.ReLU(inplace=True),
                nn.Dropout(p),
            ]
        layers += [TNN(hidden_dim, 64, k, snr_db, delta)]
        self.layers = nn.ModuleList(layers)
        # MLP
        self.enc = torch.nn.LazyLinear(hidden_dim_ffnn)
        self.out = torch.nn.LazyLinear(n_classes)

    def forward(self, x, lower, upper):
        # # TNN
        for i in range(len(self.layers)):
            x = self.layers[i](x, lower)
        # Max pooling: [batch_size, nodes, F] -> [batch_size, F]
        x = x.max(dim=1)[0]
        # MLP
        x = self.out(F.relu(self.enc(x))) 
        return x