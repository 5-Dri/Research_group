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

from utils_group import new_graph

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


edge_index1 = data.edge_index
print(data.edge_index.shape)
print(data.x.shape)
print(data.x.dtype)
print(data.edge_index.dtype)

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

data, group_index = new_graph(feature, edge_index)

print(data)
print(data.x.dtype)
print(data.edge_index.dtype)
# print(len(group_index))
# print(group_index[0].dtype)

print(type(group_index))
print(type(group_index[0]))

# def get_group_features(x2_out, group_assignment, node_indices):
#     """
#     各ノードに対応するグループの特徴量を取得する
#     :param x2_out: グループノードの特徴量テンソル
#     :param group_assignment: 各グループに属するノードのリスト
#     :param node_indices: グラフ1のノードのインデックス
#     :return: 各ノードに対応するグループの特徴量テンソル
#     """
#     node_to_group_features = []

#     # ノードごとに、その属するグループの特徴量を取得
#     for node_idx in node_indices:
#         # group_assignmentを検索してノードaが属するグループを見つける
#         for group_idx, group in enumerate(group_assignment):
#             if node_idx in group:
#                 # グループAの特徴量を取得
#                 group_features = x2_out[group_idx]
#                 node_to_group_features.append(group_features)
#                 break

#     # テンソルに変換して返す
#     return torch.stack(node_to_group_features, dim=0)