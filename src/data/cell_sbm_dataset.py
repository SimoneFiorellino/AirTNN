import torch
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
import random
from tqdm import tqdm

from scipy import sparse

import os

try: 
    from data.data_utils import white_noise, k_hop_adjacency_matrix
    from data.cell_utils import get_incidences
except:
    from data_utils import white_noise, k_hop_adjacency_matrix
    from cell_utils import get_incidences 


class CellDataset(Dataset):

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
        # out.fill_diagonal_(1)
        out = self._normalize_adj_matrix(out)

        if torch.isnan(out).any():
            print("Nan values in the impulse")

        return out, communities, G

    def _normalize_adj_matrix(self, adj_matrix):
        """Normalize the adjacency matrix."""
        # normalize the adjacency matrix
        max_eig = torch.linalg.eigh(adj_matrix)[0][-1]
        adj_matrix = adj_matrix / max_eig
        return adj_matrix

    def _get_random_edge(self, node_set, edge_list):
        valid_edges = [edge for edge in edge_list if edge[0] in node_set and edge[1] in node_set]
        if len(valid_edges) == 0:
            raise ValueError("No valid edges found with the given node set.")
        random_edge = random.choice(valid_edges)
        return random_edge
    
    def _create_diffused_impulse(self, source, k):
        """Create a diffused impulse from a source node."""
        impulse = torch.zeros(self.len_edge_list,1)
        impulse[source] = 1
        new_impulse = self.S[k,:,:].reshape(self.len_edge_list,self.len_edge_list) @ impulse
        new_impulse = new_impulse + white_noise(impulse, 40)

        # check if the impulse has Nan values
        if torch.isnan(new_impulse).any():
            print("Nan values in the impulse")

        # return the transpose of new_impulse as torch.float32
        return new_impulse.type(torch.float32)
    
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
        # ---------------------
        # Generate the SBM graph
        # ---------------------
        self.adj_matrix, self.communities, self.G = self._create_adj_matrix(n_community, p_intra, p_inter)
        # check if the adjacency matrix has Nan values
        if torch.isnan(self.adj_matrix).any():
            print("Nan values in the adjacency matrix")

        node_list = torch.tensor([i for i in range(len(list(self.G.nodes)))])
        edge_list = [(i, j) for i, j in self.G.edges] # list of edges in the graph (edges are not duplicated for undirected graphs)
        self.len_edge_list = len(edge_list)
        # ---------------------
        # Compute the upper and lower laplacians
        # ---------------------
        lower_incidence, upper_incidence, n_cycles = get_incidences(node_list, edge_list, self.G)
        print("Number of edges: ", self.len_edge_list)
        print("Number of cycles: ", n_cycles)
        self.lower_laplacian = lower_incidence.T @ lower_incidence
        self.upper_laplacian = upper_incidence @ upper_incidence.T
        assert (self.upper_laplacian.shape == self.lower_laplacian.shape)
    
        self.hodge_laplacian = self.lower_laplacian + self.upper_laplacian

        # normalize the laplacians
        self.hodge_laplacian = self._normalize_adj_matrix(self.hodge_laplacian)
        self.lower_laplacian = self._normalize_adj_matrix(self.lower_laplacian)
        self.upper_laplacian = self._normalize_adj_matrix(self.upper_laplacian)

        self.sparse_hodge_laplacian = self.hodge_laplacian.to_sparse()
        self.sparse_lower_laplacian = self.lower_laplacian.to_sparse()
        self.sparse_upper_laplacian = self.upper_laplacian.to_sparse()

        # normalized edge adjacency matrix
        # self.edge_adj_matrix = self._normalize_adj_matrix(torch.abs(self.lower_laplacian))

        # ---------------------
        # Generate the S matrix
        # ---------------------
        self.S = k_hop_adjacency_matrix(self.hodge_laplacian, self.k_diffusion)
        # check if the S matrix has Nan values
        if torch.isnan(self.S).any():
            print("Nan values in the S matrix")

        # ---------------------
        # Source edges
        # ---------------------
        sources = []
        for i in range(n_community):
            sources.append(self._get_random_edge(self.communities[i], edge_list))

        # return the indices of the source edges in the edge_list
        source_idxs = [edge_list.index(sources[i]) for i in range(n_community)]
        self.samples = []
        # ---------------------
        # Generate the samples
        # ---------------------
        for _ in tqdm(range(num_samples)):
            k = torch.randint(0, self.k_diffusion, (1,))
            n_community = torch.randint(0, len(source_idxs), (1,))
            impulse = self._create_diffused_impulse(source_idxs[n_community], k)
            self.samples.append((impulse, n_community))

    def __len__(self):
        """Return the length of the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """Return the sample and label at the given index."""
        sample, label = self.samples[idx]
        return sample, label, self.sparse_lower_laplacian, self.sparse_upper_laplacian, self.sparse_hodge_laplacian

# test main function
if __name__ == "__main__":
    dataset = CellDataset(
        n_nodes = 100,
        n_community = 10, 
        p_intra = 0.8, 
        p_inter = 0.2, 
        num_samples = 15000, 
        k_diffusion = 100,
    )

    # Save the dataset
    torch.save(dataset, './datasets/sbm/cell_dataset_demo.pt')