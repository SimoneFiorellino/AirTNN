import os
import torch
import torch.nn.functional as F

try:
    from models.components.airgnn import AirGNN
except:
    from components.airgnn import AirGNN

class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128, k=1, snr_db=100, p=0.5):
        super().__init__()
        self.p = p
        self.l1 = AirGNN(1, hidden_dim, k, snr_db)
        self.l2 = AirGNN(hidden_dim, hidden_dim, k, snr_db)
        self.l3 = AirGNN(hidden_dim, hidden_dim, k, snr_db)
        self.out = torch.nn.LazyLinear(10)

    def forward(self, x, A):
        x = torch.relu(self.l1(x, A))
        x = F.dropout(x, p=self.p, training=self.training)
        x = torch.relu(self.l2(x, A))
        x = F.dropout(x, p=self.p, training=self.training)
        x = torch.relu(self.l3(x, A))
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.out(x)
        #x = torch.softmax(x, dim=-1)
        x = x.mean(dim=1)
        return x