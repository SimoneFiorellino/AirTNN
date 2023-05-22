import os
import torch
import torch.nn.functional as F

try:
    from models.components.airtnn import AirTNN
except:
    from components.airtnn import AirTNN

class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128, hidden_dim_ffnn=1024, n_classes=11, k=1, snr_db=100, p=.0):
        super().__init__()
        self.p = p
        self.l1 = AirTNN(1, hidden_dim, k, snr_db)
        self.l2 = AirTNN(hidden_dim, hidden_dim, k, snr_db)
        self.enc = torch.nn.LazyLinear(hidden_dim_ffnn)
        self.out = torch.nn.LazyLinear(n_classes)

    def forward(self, x, lower, upper, hodge):
        # # AirTNN
        x = F.relu(self.l1(x, lower, upper))
        x = F.relu(self.l2(x, lower, upper))
        x = F.dropout(x, p=self.p, training=self.training)
        # Average pooling: [batch_size, nodes, F] -> [batch_size, F]
        x = x.mean(dim=1)
        # MLP
        x = self.out(F.relu(self.enc(x))) 
        return x