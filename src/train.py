import logging
import warnings
from pytorch_lightning import seed_everything

import hydra
import os
import torch

from data.cell_datamodule import CellDataModule
from data.cell_sbm_dataset import CellDataset
os.environ['HYDRA_FULL_ERROR'] = '1'

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg):
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    # seed_everything(cfg.seed, workers=True)

    # check if the dataset do not exists
    # if not os.path.exists('./datasets/sbm/cell_dataset.pt'):
    dataset = CellDataset(
        n_nodes = 70,
        n_community = 10, 
        p_intra = .9, 
        p_inter = .1/10, 
        num_samples = 15000, 
        k_diffusion = 5,
        spike = 1000,
        snr_db = 60,
    )
    # Save the dataset
    torch.save(dataset, './datasets/sbm/cell_dataset.pt')

    print("Step 1: Create the logger")
    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_hyperparams(cfg)

    print("Step 2: Create the model")
    model = hydra.utils.instantiate(cfg.model)

    # print("Step 3: Create the datamodule")
    # datamodule = hydra.utils.instantiate(cfg.datamodule)

    print("Step 4: Create the trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    print("Step 5: Train the model")
    trainer.fit(model=model)

    print("Step 6: Test the model")
    trainer.test(model=model)


if __name__ == "__main__":
    main()