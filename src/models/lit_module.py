import os
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MaxMetric, MeanMetric, ConfusionMatrix
from typing import Any, List

from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from data.cell_sbm_dataset import CellDataset
    

class LitModule(LightningModule):
    def __init__(
            self, 
            backbone: torch.nn.Module, 
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            n_classes: int = 10,
        ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(logger=True, ignore=["backbone"])
        self.backbone = backbone

        # print the used GPU device
        print("Using GPU device: {}".format(torch.cuda.current_device()))

        # load dataset
        self.dataset = torch.load('./datasets/sbm/cell_dataset.pt')
        print("cell dataset loaded")

        # load the laplacians on the gpu:0
        self.hodge = self.dataset.sparse_hodge_laplacian.to(torch.cuda.current_device())
        self.low = self.dataset.sparse_lower_laplacian.to(torch.cuda.current_device())
        self.up = self.dataset.sparse_upper_laplacian.to(torch.cuda.current_device())

        # split dataset into train, val, test
        self.data_train, self.data_val, self.data_test = random_split(
            self.dataset, [10000, 2500, 2500]) 

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)

        # metric objects for calculating and averaging confusion matrix across batches for test set
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=n_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x):
        embedding = self.backbone(x, self.low, self.up, self.hodge)
        return embedding
    
    def model_step(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(-1)
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.model_step(batch, batch_idx)
        self.train_loss(loss)
        self.train_acc(y_hat, y)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.model_step(batch, batch_idx)
        self.val_loss(loss)
        self.val_acc(y_hat, y)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.model_step(batch, batch_idx)
        self.test_loss(loss)
        self.test_acc(y_hat, y)
        self.test_confusion_matrix(y_hat, y)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_epoch_end(self, outputs: List[Any]):
        print(self.test_confusion_matrix.compute())

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
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )