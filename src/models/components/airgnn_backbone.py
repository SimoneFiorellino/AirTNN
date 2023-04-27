import os
import torch
import torch.nn.functional as F

try:
    from models.components.airgnn import AirGNN
except:
    from components.airgnn import AirGNN

class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128, k=1, snr_db=100):
        super().__init__()
        self.l1 = AirGNN(1, hidden_dim, k, snr_db)
        self.l2 = AirGNN(hidden_dim, hidden_dim, k, snr_db)
        #self.l3 = AirGNN(hidden_dim, 10, k, snr_db)
        self.l3 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x, A):
        x = torch.relu(self.l1(x, A))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = torch.relu(self.l2(x, A))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = torch.relu(self.l3(x))
        x = x.mean(dim=1)
        return x