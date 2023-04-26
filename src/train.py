import logging
import warnings
from pytorch_lightning import seed_everything

import hydra
import os
import torch

from data.sbm_datamodule import SBMDataModule
from data.sbm_dataset import SBMDataset


log = logging.getLogger(__name__)
os.environ['HYDRA_FULL_ERROR'] = '1'


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg):
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    seed_everything(cfg.seed, workers=True)

    ## If exist the dataset, load it. Otherwise, create it.
    if not os.path.exists('./topological_air_nn/datasets/sbm/sbm_dataset.pt'):
        print("Create the dataset")
        dataset = SBMDataset(
            n_nodes=100,
            n_community=10,
            p_intra=.8,
            p_inter=.2,
            num_samples=15000,
            k_diffusion=100
        )
        #print(z)
        # Save the dataset
        print("Save the dataset")
        torch.save(dataset, './topological_air_nn/datasets/sbm/sbm_dataset.pt')
        # Sace the adjacency matrix
        print("Save the adjacency matrix")
        torch.save(dataset.get_adj_matrix(), './topological_air_nn/datasets/sbm/sbm_adj_matrix.pt')

    print("Step 1: Create the logger")
    logger = hydra.utils.instantiate(cfg.logger)

    print("Step 2: Create the model")
    model = hydra.utils.instantiate(cfg.net)

    print("Step 3: Create the datamodule")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    print("Step 4: Create the trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    print("Step 5: Train the model")
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()