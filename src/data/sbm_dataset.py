import torch
from torch.utils.data import Dataset
import networkx as nx
import numpy as np

import os

# change the root directory to the project root
os.chdir('..')

from data.data_utils import white_noise, k_hop_adjacency_matrix


class SBMDataset(Dataset):

    """A dataset of stochastic block model graphs."""
    torch.manual_seed(42)

    def _create_adj_matrix(self, community_size, p_intra, p_inter):
        """Create a stochastic block model adjacency matrix."""
        sizes = [self.n_for_each_community for _ in range(community_size)]
        p = np.full((community_size, community_size), p_inter)
        np.fill_diagonal(p, p_intra)
        G = nx.stochastic_block_model(sizes, p, seed=42)
        out = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
        max_eig = torch.linalg.eigh(out)[0][-1]
        return out / max_eig
    
    def _create_diffused_impulse(self, source, k):
        """Create a diffused impulse from a source node."""
        impulse = torch.zeros(self.n_nodes)
        impulse[source] = 1
        noise = white_noise(impulse, 40)
        new_impulse = self.S[k,:,:] @ impulse + noise
        # return the transpose of new_impulse as torch.float32
        return new_impulse.reshape(self.n_nodes,1).type(torch.float32)

    def __init__(self, n_nodes, community_size, p_intra, p_inter, num_samples):
        """Initialize the dataset."""
        self.n_nodes = n_nodes
        self.num_samples = num_samples
        self.n_for_each_community = n_nodes // community_size
        self.adj_matrix = self._create_adj_matrix(community_size, p_intra, p_inter)
        self.S = k_hop_adjacency_matrix(self.adj_matrix, 100)

        list_communities = torch.arange(0, 10)
        sources = list_communities * 10

        self.samples = []
        
        """Generate the samples"""
        for _ in range(num_samples):
            k = torch.randint(0, 100, (1,))
            label = torch.randint(0, len(sources), (1,))
            impulse = self._create_diffused_impulse(sources[label], k)
            self.samples.append((impulse, label))
            

    def __len__(self):
        """Return the length of the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """Return the sample and label at the given index."""
        sample, label = self.samples[idx]
        return sample, label
    
    def get_adj_matrix(self):
        """Return the adjacency matrix."""
        return self.adj_matrix
    
    def get_n_nodes(self):
        """Return the number of nodes."""
        return self.n_nodes
    
# test SBMDataset with main
if __name__ == '__main__':
    dataset = SBMDataset(
        n_nodes=100,
        community_size=10,
        p_intra=0.8,
        p_inter=0.1,
        num_samples=15000
    )
    # Save the dataset
    torch.save(dataset, './datasets/sbm/sbm_dataset.pt')

    # Load the saved dataset and get the adjacency matrix
    saved_dataset = torch.load('./datasets/sbm/sbm_dataset.pt')
    saved_adj_matrix = saved_dataset.get_adj_matrix()

    # save the adjacency matrix as a tensor
    torch.save(saved_adj_matrix, './datasets/sbm/sbm_adj_matrix.pt')

    print(saved_adj_matrix)

    # Check if the loaded adjacency matrix is the same as the original one
    print(torch.allclose(dataset.adj_matrix, saved_adj_matrix))