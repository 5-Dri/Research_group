import random
import torch
import networkx as nx
from torch_geometric.data import Data
import numpy as np




# すべてのノードが属するようにグループ分けを行う関数
def group_nodes_weighted_by_degree(G):
    nodes = set(G.nodes())
    degrees = dict(G.degree())
    total_degree = sum(degrees.values())
    groups = {}
    group_id = 0
    select_node = []

    while nodes:
        group_id += 1
        # 重み付きランダムサンプリングによりノードを選択
        weights = [degrees[node] for node in nodes]
        seed_node = random.choices(list(nodes), weights=weights, k=1)[0]
        select_node.append(seed_node)
        # 1ステップのノード（隣接ノード）を取得
        neighbors = list(G.neighbors(seed_node)) + [seed_node]
        # 新しいグループに追加
        for node in neighbors:
            if node in nodes:
                nodes.remove(node)
                groups[node] = group_id

    #subgraph = make_group_graph(G, select_node)

    #return groups, subgraph
    return groups , select_node


# グループごとのエッジインデックスをリスト型で作成する関数
def create_group_edge_indices_list(G, partition):
    group_edges = {}
    for (u, v) in G.edges():
        group_u = partition[u]
        group_v = partition[v]
        if group_u == group_v:  # 同じグループのノード間のエッジのみ考慮
            if group_u not in group_edges:
                group_edges[group_u] = [[], []]  # 辺の始点リストと終点リストを初期化
            group_edges[group_u][0].append(u)
            group_edges[group_u][1].append(v)
    return group_edges


# PyTorch Geometricのデータ形式に変換する関数
# def convert_to_pyg_data(group_edge_indices, data1):
#     pyg_data_list = []
#     for group, (edge_start, edge_end) in group_edge_indices:
#         edge_index = torch.tensor([edge_start, edge_end], dtype=torch.long)
#         data = Data(x=data1.x, edge_index=edge_index, y = data1.y, train_mask = data1.train_mask, val_mask = data1.val_mask, test_mask = data1.test_mask)
#         pyg_data_list.append(data)
#     return pyg_data_list

def convert_to_pyg_data(group_edge_indices, data1):
    pyg_data_list = []
    for edges in group_edge_indices:
        edge_index = torch.tensor(edges, dtype=torch.long).to(device='cuda')
        data = Data(x=data1.x, edge_index=edge_index, y = data1.y, train_mask = data1.train_mask, val_mask = data1.val_mask, test_mask = data1.test_mask)
        pyg_data_list.append(data)
    return pyg_data_list


def make_group_graph(G, select_nodes):


    subgraph = nx.Graph()
    subgraph.add_nodes_from(select_nodes)

    for node in select_nodes:
        for target_node in select_nodes:
             if node != target_node:  # 自分自身を除外
                try:
                    path_length = nx.shortest_path_length(G, source=node, target=target_node)
                    if path_length <= 3:
                        subgraph.add_edge(node, target_node)
                except nx.NetworkXNoPath:
                    continue
    
    return subgraph


def create_weighted_groups(G, num_groups):
    # ノードリストとその次数を取得
    degree_df = G.degrees().to_pandas()
    nodes = degree_df['vertex'].values
    degrees = degree_df['degree'].values

    # 重み付きランダム選択のための確率分布を計算
    total_degree = np.sum(degrees)
    probabilities = degrees / total_degree

    # 重み付きランダムにノードを選択
    random_nodes = np.random.choice(nodes, size=num_groups, p=probabilities, replace=False)

    groups = {}
    all_assigned_nodes = set()

    for i, node in enumerate(random_nodes):
        # 1ステップの距離にあるノードを取得
        neighbors = G.get_two_hop_neighbors(cudf.Series([node])).iloc[:, 1].to_pandas().values
        # 自分自身と隣接ノードをグループに追加
        group_nodes = set(neighbors)
        group_nodes.add(node)
        groups[f"group_{i}"] = group_nodes
        all_assigned_nodes.update(group_nodes)

    return groups
