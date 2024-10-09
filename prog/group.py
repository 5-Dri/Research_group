import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import argparse
import os.path as osp
import time
import networkx as nx
from matplotlib import pyplot as plt

from utils_group import group_nodes_weighted_by_degree, get_eigen_zeros, filter_single_element_lists, filter_adjacent_nodes, compute_group_features_mean, create_group_edge_indices_list,make_group_graph, convert_pygdata

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CiteSeer')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(args.dataset)


edge_index1 = data.edge_index
print(data.edge_index.shape)
print(data.x.shape)

#print(len(data.x[0]))

edge_index2 = torch.Tensor([[0,1,2,2,2,2,3,4,4,5,5,5,6,7,7,8,9,9],
                           [2,2,0,1,6,5,4,3,5,2,4,9,2,8,9,7,5,7]])

edge_index3 = torch.Tensor([[0,1,1,1,2,3,3,3,3,4,4,4,5,5,6,7,7,7,7,8,8,8,9,10,10,11,11,11,12,12,12,13,13,14,15,15,16,16,18,19],
                           [1,0,2,3,1,1,4,8,7,3,6,5,4,19,4,3,11,10,12,3,9,10,8,8,7,7,13,14,7,15,16,11,18,11,12,17,12,17,13,5]])


data_hoge = torch.Tensor([[0,0,0,0,0],
                          [1,1,1,1,1],
                         [2,2,2,2,2],
                         [3,3,3,3,3],
                         [4,4,4,4,4],
                         [5,5,5,5,5],
                         [6,6,6,6,6],
                         [7,7,7,7,7],
                         [8,8,8,8,8],
                         [9,9,9,9,9],
                         [10,10,10,10,10],
                         [11,11,11,11,11],
                         [12,12,12,12,12],
                         [13,13,13,13,13],
                         [14,14,14,14,14],
                         [15,15,15,15,15],
                         [16,16,16,16,16],
                         [17,17,17,17,17],
                         [18,18,18,18,18],
                         [19,19,19,19,19]])



edge_index = edge_index1
feature = data.x

# edge_index = edge_index3
# feature = data_hoge


#nxに使える形にedge_indexを変換
edges = edge_index.t().tolist()
# print("edges", edges)

#グラフ作成
G = nx.Graph()
G.add_edges_from(edges)
print("first G", G)

# pos = nx.spring_layout(G, seed=1) #ノードの配置を指定

# # グラフの描画
# plt.figure(figsize=(10,10)) #グラフエリアのサイズ
# nx.draw_networkx(G, pos, node_color='#87cefa', font_size=10) #グラフの描画(おまかせ)
# plt.show() #グラフの描画

# グループ分け
groups, target_node = group_nodes_weighted_by_degree(G)
# print("first_groups", groups)
print("first group len", len(groups))
# print("target_nodes", target_node)
print("first taget len", len(target_node))

# result = filter_single_element_lists(groups)
#print("長さ１のグループ",result)

final_groups, target_node2 = filter_adjacent_nodes(groups, G, target_node)
#print("最終グループ",result2)
# print("new target_node", target_node2)
print("new group len", len(final_groups))
print("new target len", len(target_node2))


# for i in range(11):
#     # print("group_index ??", final_groups[i])
#     print("group_index ??", target_node2[i])

#     # if target_node2[i] in final_groups[i]:
#     #     print("配列の中にあります。")
#     # else:
#     #     print("配列の中にありません。")


group_feature = compute_group_features_mean(final_groups, feature)


print(group_feature.shape)
# print(group_feature)





# グループのedge_indexをリストで
# group_edge_indices_list = create_group_edge_indices_list(G, groups)
# group_edge_indices_tensors = []
# for group, (edge_start, edge_end) in group_edge_indices_list.items():
#     edge_indices = torch.tensor([edge_start, edge_end], dtype=torch.long)
#     group_edge_indices_tensors.append(edge_indices)

# print("group list", group_edge_indices_list)
# print("group tensor", group_edge_indices_tensors)

# #グループ内のノードのリスト


#グループをノードとする新しいグラフ
new_G = make_group_graph(G, target_node2)
print("new G", new_G)

# print("G node", new_G.nodes())
# print(new_G.nodes())

new_edge = new_G.edges()
# print("new edge", new_edge)

# new_edge_list = create_group_edge_indices_list(new_G, final_groups)
# print(new_edge_list)

lam_zero = get_eigen_zeros(new_G)
print(lam_zero)

# pos = nx.spring_layout(new_G, seed=1) #ノードの配置を指定

# # グラフの描画
# plt.figure(figsize=(10,10)) #グラフエリアのサイズ
# nx.draw_networkx(new_G, pos, node_color='#87cefa',node_shape='D', font_size=10) #グラフの描画(おまかせ)
# plt.show() #グラフの描画


pyg_data = convert_pygdata(group_feature, new_G)
print(pyg_data)

