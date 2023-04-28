import networkx as nx
import numpy as np
import torch

def detect_communities(community_size = 10, p_intra = 0.8, p_inter = 0.2):
    """Detect communities in a stochastic block model graph."""
    # creating the SBM graph
    n_nodes_for_each_community = 100 // community_size
    sizes = [n_nodes_for_each_community for _ in range(community_size)]
    p = np.full((community_size, community_size), p_inter)
    np.fill_diagonal(p, p_intra)
    G = nx.stochastic_block_model(sizes, p, seed=1418)
    
    # detecting communities
    communities = {i: [] for i in range(community_size)}
    for node in G.nodes():
        community = G.nodes[node]["block"]
        communities[community].append(node)
    
    return G, communities

def sample_source_and_label(communities):
    """Sample a source node and a label."""
    # sample a source node
    source_community = np.random.randint(len(communities))
    source_node = np.random.choice(communities[source_community])
    
    return source_node, source_community

import random

def choose_source_nodes(G, use_max_degree=True):
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

# main
if __name__ == "__main__":
    G, communities = detect_communities()
    max_nodes, rand_nodes = choose_source_nodes(G, use_max_degree=True), choose_source_nodes(G, use_max_degree=False)
    print("max_nodes", max_nodes)
    print("rand_nodes", rand_nodes)
    print(max_nodes[3])


    # source_node, label = sample_source_and_label(communities)
    # print("source_node", source_node)
    # print("label", label)