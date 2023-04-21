"""Model that use two layer of AirGNN"""

import torch
from torch import nn
from torch.nn import functional as F

from models.components.airgnn import AirGNN

class AirGNN2Layer(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, snr_db: int, A: torch.Tensor):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.k = k
        self.snr_db = snr_db
        self.airgnn1 = AirGNN(c_in, c_out, k, snr_db)
        self.airgnn2 = AirGNN(c_out, c_out, k, snr_db)
        self.fc = nn.Linear(100, 10, dtype=torch.float32)

        # compute symmetric normalization of adjacency matrix
        D = torch.diag(torch.sum(A, dim=1))
        D = torch.inverse(torch.sqrt(D))
        A = torch.matmul(torch.matmul(D, A), D)

        self.A = A[None,:,:].cuda()



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.airgnn1(x, self.A)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        # x = self.airgnn2(x, self.A)
        # x = F.elu(x)
        x = x.mean(dim=2)
        #x = x.reshape(x.shape[0],-1)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        return x