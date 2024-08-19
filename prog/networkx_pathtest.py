import torch
import networkx as nx

edge_index = torch.Tensor([[0,1,1,1,2,3,3,3,3,4,4,4,5,5,6,7,7,7,7,8,8,8,9,10,10,11,11,11,12,12,12,13,13,14,15,15,16,16,18,19],
                           [1,0,2,3,1,1,4,8,7,3,6,5,4,19,4,3,11,10,12,3,9,10,8,8,7,7,13,14,7,15,16,11,18,11,12,17,12,17,13,5]])

edges = edge_index.t().tolist()

G = nx.Graph()
G.add_edges_from(edges)


def has_multiple_paths_within_length(G, source, target, max_length):
    all_paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=max_length))
    return len(all_paths) > 1, all_paths

max_length = 3


print(has_multiple_paths_within_length(G, 10, 4, max_length))
print(has_multiple_paths_within_length(G, 10, 12, max_length))
print(has_multiple_paths_within_length(G, 10, 18, max_length))