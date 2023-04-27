import os
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy



class LitModule(LightningModule):
    def __init__(
            self, 
            backbone: torch.nn.Module, 
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x, A):
        embedding = self.backbone(x, A)
        return embedding
    
    def model_step(self, batch, batch_idx):
        x, y, A = batch
        y = y.reshape(-1)
        y_hat = self.forward(x, A)
        loss = F.cross_entropy(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.model_step(batch, batch_idx)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.model_step(batch, batch_idx)
        self.log("val/loss", loss, on_epoch=True)
        self.log("val/acc", self.val_acc(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.model_step(batch, batch_idx)
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", self.test_acc(y_hat, y), on_epoch=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}