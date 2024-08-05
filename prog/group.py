import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import argparse
import os.path as osp
import time
import networkx as nx

from utils_group import group_nodes_weighted_by_degree, create_group_edge_indices_list,make_group_graph, convert_to_pyg_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

edge_index1 = data.edge_index



edge_index2 = torch.Tensor([[0,1,2,2,2,2,3,4,4,5,5,5,6,7,7,8,9,9],
                           [2,2,0,1,6,5,4,3,5,2,4,9,2,8,9,7,5,7]])

edge_index3 = torch.Tensor([[0,1,1,1,2,3,3,3,3,4,4,4,5,5,6,7,7,7,7,8,8,8,9,10,10,11,11,11,12,12,12,13,13,14,15,15,16,16,18,19],
                           [1,0,2,3,1,1,4,8,7,3,6,5,4,19,4,3,11,10,12,3,9,10,8,8,7,7,13,14,7,15,16,11,18,11,12,17,12,17,13,5]])

edge_index = edge_index3


#nxに使える形にedge_indexを変換
edges = edge_index.t().tolist()
# print("edges", edges)

#グラフ作成
G = nx.Graph()
G.add_edges_from(edges)
# print("G", G)

# グループ分け
groups, target_node = group_nodes_weighted_by_degree(G)
print("groups", groups)
print("target_nodes", target_node)


# # グループのedge_indexをリストで
# group_edge_indices_list = create_group_edge_indices_list(G, groups)
# group_edge_indices_tensors = []
# for group, (edge_start, edge_end) in group_edge_indices_list.items():
#     edge_indices = torch.tensor([edge_start, edge_end], dtype=torch.long)
#     group_edge_indices_tensors.append(edge_indices)

# # print("group list", group_edge_indices_list)
# # print("group tensor", group_edge_indices_tensors)

# #グループ内のノードのリスト



# #グループをノードとする新しいグラフ
# new_G = make_group_graph(G, target_node)
# print("new G", new_G)

# new_edge = new_G.edges()
# print("new edge", new_edge)








