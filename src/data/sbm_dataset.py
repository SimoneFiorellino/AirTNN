import torch
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
import random

import os

# # change the root directory to the project root
# os.chdir('..')

from data.data_utils import white_noise, k_hop_adjacency_matrix


class SBMDataset(Dataset):

    """A dataset of stochastic block model graphs."""
    torch.manual_seed(42)

    def _create_adj_matrix(self, community_size, p_intra, p_inter):
        """Create a stochastic block model adjacency matrix."""
        # creating the SBM graph
        sizes = [self.n_nodes_for_each_community for _ in range(community_size)]
        p = np.full((community_size, community_size), p_inter)
        np.fill_diagonal(p, p_intra)
        G = nx.stochastic_block_model(sizes, p, seed=1418)

        # detecting communities
        communities = {i: [] for i in range(community_size)}
        for node in G.nodes():
            community = G.nodes[node]["block"]
            communities[community].append(node)


        # Check if the graph is connected
        print("Graph connection: ", nx.is_connected(G))
        # convert the graph to a torch tensor
        out = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
        # normalize the adjacency matrix
        out.fill_diagonal_(1)
        max_eig = torch.linalg.eigh(out)[0][-1]
        out = out / max_eig

        if torch.isnan(out).any():
            print("Nan values in the impulse")

        return out, communities, G
    
    def _create_diffused_impulse(self, source, k):
        """Create a diffused impulse from a source node."""
        impulse = torch.zeros(self.n_nodes,1)
        impulse[source] = 1
        new_impulse = self.S[k,:,:].reshape(self.n_nodes,self.n_nodes) @ impulse
        new_impulse = new_impulse + white_noise(impulse, 40)

        # check if the impulse has Nan values
        if torch.isnan(new_impulse).any():
            print("Nan values in the impulse")

        # return the transpose of new_impulse as torch.float32
        return new_impulse.type(torch.float32)

    def _choose_source_nodes(self, G, use_max_degree=True):
        """Choose source nodes for each community."""
        # initialize tensor to store chosen source nodes
        source_nodes = torch.zeros((len(set(nx.get_node_attributes(G, "block").values()))), dtype=torch.long)
        for community in set(nx.get_node_attributes(G, "block").values()):
            # get nodes in community
            nodes_in_community = [n for n in G.nodes() if G.nodes[n]["block"] == community]
            if use_max_degree:
                # choose node with maximum degree in community as source
                max_degree = -1
                max_degree_node = None
                for node in nodes_in_community:
                    degree = G.degree(node)
                    if degree > max_degree:
                        max_degree = degree
                        max_degree_node = node
                source_nodes[community] = max_degree_node
            else:
                # choose node randomly from community as source
                source_nodes[community] = random.choice(nodes_in_community)
        return source_nodes
    
    def _sample_source_and_label(self, communities):
        """Sample a source node and a label."""
        # sample a source node
        source_community = np.random.randint(len(communities))
        source_node = np.random.choice(communities[source_community])
        
        return source_node, source_community

    def __init__(self, n_nodes, n_community, p_intra, p_inter, num_samples, k_diffusion):
        """Initialize the dataset."""
        self.k_diffusion = k_diffusion
        self.n_nodes = n_nodes
        self.num_samples = num_samples
        self.n_nodes_for_each_community = n_nodes // n_community
        self.adj_matrix, self.communities, self.G = self._create_adj_matrix(n_community, p_intra, p_inter)
        print(self.adj_matrix)

        # check if the adjacency matrix has Nan values
        if torch.isnan(self.adj_matrix).any():
            print("Nan values in the adjacency matrix")

        # create the S matrix
        self.S = k_hop_adjacency_matrix(self.adj_matrix, self.k_diffusion)
        # check if the S matrix has Nan values
        if torch.isnan(self.S).any():
            print("Nan values in the S matrix")

        # choose source nodes
        sources = self._choose_source_nodes(self.G, use_max_degree=False)

        self.samples = []
        """Generate the samples"""
        for _ in range(num_samples):
            k = torch.randint(0, self.k_diffusion, (1,))
            n_community = torch.randint(0, len(sources), (1,))
            impulse = self._create_diffused_impulse(sources[n_community], k)
            self.samples.append((impulse, n_community))

    def __len__(self):
        """Return the length of the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """Return the sample and label at the given index."""
        sample, label = self.samples[idx]
        return sample, label, self.adj_matrix
    
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
        p_inter=0.2,
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