"""
Datamodule for the SBM dataset.
"""
from typing import Any, Dict, Optional, Tuple
from os import path
import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from data.sbm_dataset import SBMDataset


class SBMDataModule(LightningDataModule):
    def __init__(self, batch_size=64, num_workers=0, pin_memory=False):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Load the dataset
        path = os.path.dirname(os.path.abspath(__file__))
        # compute two step back in the path directory
        path = os.path.dirname(os.path.dirname(path))

        self.dataset = torch.load(path + '/datasets/sbm/sbm_dataset.pt')

        self.data_train, self.data_val, self.data_test = random_split(
            self.dataset, [10000, 2500, 2500]) 

    @property
    def num_classes(self):
        return 10

    # def setup(self, stage=None):
    #     self.train_set, self.val_set, self.test_set = random_split(
    #         self.dataset, [10000, 2500, 2500]) 

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
    _ = SBMDataModule()
