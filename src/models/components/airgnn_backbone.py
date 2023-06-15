import os
import torch
import torch.nn.functional as F

try:
    from models.components.airgnn import AirGNN
except:
    from components.airgnn import AirGNN

class Backbone(torch.nn.Module):

    def __init__(self, hidden_dim=128, hidden_dim_ffnn=1024, n_classes=11, k=1, snr_db=100, p=.0, delta=1):
        super().__init__()
        self.p = p
        self.l1 = AirGNN(1, hidden_dim, k, snr_db, delta)
        self.l2 = AirGNN(hidden_dim, hidden_dim, k, snr_db, delta)
        self.enc = torch.nn.LazyLinear(hidden_dim_ffnn)
        self.out = torch.nn.LazyLinear(n_classes)

    def forward(self, x, low, up):
        # AirGNN
        x = F.relu(self.l1(x, low))
        x = F.relu(self.l2(x, low))
        x = F.dropout(x, p=self.p, training=self.training)
        # Average pooling: [batch_size, nodes, F] -> [batch_size, F]
        #x = x.mean(dim=1)
        x = x.max(dim=1)[0]
        # MLP
        x = self.out(F.relu(self.enc(x))) 
        return x