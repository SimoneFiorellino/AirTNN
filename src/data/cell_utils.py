import networkx as nx
import torch


def edge_to_node_matrix(edges, nodes, one_indexed=True):
    sigma1 = torch.zeros((len(nodes), len(edges)), dtype=torch.float)
    offset = int(one_indexed)
    j = 0
    for edge in edges:
        x, y = edge
        sigma1[x - offset][j] -= 1
        sigma1[y - offset][j] += 1
        j += 1
    return sigma1


def edge_to_cycle_matrix(edges, cycles):
    sigma2 = torch.zeros((len(edges), len(cycles)), dtype=torch.float)
    edges = [e for e in edges]
    edges = {edges[i]: i for i in range(len(edges))}
    for idx in range(len(cycles)):
        cycle_length = len(cycles[idx])
        for i in range(cycle_length):
            if (cycles[idx][i - 1], cycles[idx][i]) in edges:
                sigma2[edges[(cycles[idx][i - 1], cycles[idx][i])]][idx] += 1
            else:
                sigma2[edges[(cycles[idx][i], cycles[idx][i - 1])]][idx] -= 1
    return sigma2


def build_upper_features(lower_features, simplex_list):
    new_features = []
    for i in range(len(simplex_list)):
        if isinstance(simplex_list[i], tuple):
            idx = list(simplex_list[i])
        else:
            idx = simplex_list[i]
        new_features.append(lower_features[idx].mean(axis=0))
    return torch.stack(new_features)


def get_incidences(nodes, edge_list, G):
    cycle_list = nx.cycle_basis(G)
    lower_incidence = edge_to_node_matrix(list(edge_list), nodes, one_indexed=False)
    upper_incidence = edge_to_cycle_matrix(edge_list, cycle_list)
    return lower_incidence, upper_incidence, len(cycle_list)