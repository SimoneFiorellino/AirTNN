"""
Lightning module for MLP
"""

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(1, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.relu(self.l2(x))
        x = x.mean(dim=1)
        return x

class MLPLitModule(LightningModule):
    def __init__(self, hidden_dim=32, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = Backbone(hidden_dim=hidden_dim)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding
    
    def model_step(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(-1)
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.model_step(batch, batch_idx)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc(y_hat, y), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.model_step(batch, batch_idx)
        self.log("val/loss", loss, on_epoch=True)
        self.log("val/acc", self.val_acc(y_hat, y), on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.model_step(batch, batch_idx)
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", self.test_acc(y_hat, y), on_epoch=True)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
