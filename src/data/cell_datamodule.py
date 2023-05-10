"""
Datamodule for the SBM dataset.
"""
from typing import Any, Dict, Optional, Tuple
from os import path
import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from data.cell_sbm_dataset import CellDataset


class CellDataModule(LightningDataModule):
    def __init__(self, batch_size=64, num_workers=0, pin_memory=False):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset = torch.load('./datasets/sbm/cell_dataset.pt')
        print("cell dataset loaded")
        
        self.data_train, self.data_val, self.data_test = random_split(
            self.dataset, [10000, 2500, 2500]) 

    @property
    def num_classes(self):
        return 10
    
    def frequency_labels(self):
        """Return the frequency of the labels in the dataset."""
        labels = torch.cat([y for _, y, _ in self.dataset])
        return torch.bincount(labels)

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
    
if __name__ == "__main__":
    _ = CellDataModule()