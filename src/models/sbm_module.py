from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from models.components.airgnn_model import AirGNN2Layer

import os

class SBMLitModule(LightningModule):
    def __init__(self,
            c_in: int,
            c_out: int,
            k: int,
            snr_db: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
        ):
        super().__init__()

        # Load adjacency matrix in ./datasets/sbm/sbm_adj_matrix.pt
        # the actual file is in ./src/models/sbm_module.py
        # avoid: FileNotFoundError: [Errno 2] No such file or directory: 'sbm_adj_matrix.pt'
        path = os.path.dirname(os.path.abspath(__file__))
        # compute two step back in the path directory
        path = os.path.dirname(os.path.dirname(path))

        self.adj_matrix = torch.load(path + '/datasets/sbm/sbm_adj_matrix.pt')

        self.model = AirGNN2Layer(
            c_in = c_in,
            c_out = c_out,
            k = k,
            snr_db = snr_db,
            A = self.adj_matrix
        )
        self.save_hyperparameters(logger=False)

        # Metrics
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

        # save optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler       

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def model_step(self, batch: Any):
        x, y = batch
        y = y.reshape(-1)
        logits = self.forward(x)
        #print(logits, y)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y
    
    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val_acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.9)
    
if __name__ == "__main__":
    _ = SBMLitModule(0,0,0,0,None,None)