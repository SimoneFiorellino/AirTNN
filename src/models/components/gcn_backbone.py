import torch
import torch.nn.functional as F

class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128, p=0.5):
        super().__init__()
        self.p = p
        self.l1 = torch.nn.Linear(1, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l3 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x, A):
        x = torch.relu(self.l1(A@x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = torch.relu(self.l2(A@x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = torch.relu(self.l3(x))
        x = x.mean(dim=1)
        return x