import os
import torch
import torch.nn.functional as F

try:
    from models.components.airgnn import AirGNN
except:
    from components.airgnn import AirGNN

class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128, k=1, snr_db=100, p=.0):
        super().__init__()
        self.p = p
        self.l1 = AirGNN(1, hidden_dim, k, snr_db)
        self.l2 = AirGNN(hidden_dim, hidden_dim, k, snr_db)
        self.enc = torch.nn.LazyLinear(128)
        self.out = torch.nn.LazyLinear(10)

    def forward(self, x, low, up):
        # AirGNN
        x = F.relu(self.l1(x, low))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.l2(x, low))
        x = F.dropout(x, p=self.p, training=self.training)
        # MLP
        x = self.out(F.relu(self.enc(x))) 
        # Average pooling -> [batch_size, 10] -> max vote
        x = x.mean(dim=1)
        return x