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
        self.fc = nn.Linear(c_out, 10, dtype=torch.float32)
        self.A = A.cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.airgnn1(x, self.A)

        x = F.relu(x)
        x = self.airgnn2(x, self.A)

        x = F.relu(x)
        x = x.mean(dim=1)

        x = self.fc(x)

        return x